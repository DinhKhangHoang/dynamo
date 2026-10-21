//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWorkerRegistrationDoesNotRequireControllerPermission(t *testing.T) {
	t.Log("the independently verified initial world registers before any traffic target")
	fixture := newMembershipFixture(t, 2, 3)
	workers := &workerControlFixture{engine: fixture.engine, registered: map[int32]bool{0: true, 1: true}}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i))
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	membership, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	capacity := &workerTestCapacity{processes: fixture.adapter.Processes, count: 2}
	observedCapacity, err := capacity.Observe(t.Context(), groupID)
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	verifier := &WorkerVerifier{Membership: fixture.adapter}
	initialTraffic, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	require.Len(t, initialTraffic.Admitted, 2)
	require.Zero(t, initialTraffic.AppliedRevision)
	status, err := domain.NewGroupStatus(membership.CommittedTopology, observedCapacity, initialTraffic)
	require.NoError(t, err)
	target := fixture.target(t, membership.CommittedTopology, "grow-three", 10, 3, domain.PlanKindGrow)

	t.Log("reconcile until both old ranks have accepted native expansion")
	submitted := false
	for step := 0; step < 30; step++ {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		fixture.engine.mu.Lock()
		submitted = len(fixture.engine.requests[0]) > 0 && len(fixture.engine.requests[1]) > 0
		fixture.engine.mu.Unlock()
		if submitted {
			break
		}
	}
	require.True(t, submitted)
	require.NotNil(t, status.Traffic.Accepted)
	previousRevision := status.Traffic.Accepted.ControlRevision

	t.Log("stop controller reconciliation; native commit and runtime verification register the joiner")
	fixture.engine.complete(3)
	workers.mu.Lock()
	workers.registered[2] = true
	workers.mu.Unlock()
	observed, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	assert.Len(t, observed.Admitted, 3)
	assert.Equal(t, previousRevision, observed.AppliedRevision, "registration needs no new controller revision")

	t.Log("observe the already-established proof after the controller restarts")
	for step := 0; step < 20; step++ {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &target.Plan, status)
		require.NoError(t, err)
		status = result.Status
		if status.Transition.Outcome == domain.TransitionOutcomeCompleted {
			break
		}
	}
	require.Equal(t, domain.TransitionOutcomeCompleted, status.Transition.Outcome)
	assert.Zero(t, workers.mutations, "the controller must never grant serving permission or initiate warmup")
}

func TestWorkerServingEvidenceAndUnsupportedDrain(t *testing.T) {
	t.Log("observe runtime-owned verification and registration for one exact native incarnation")
	fixture := newMembershipFixture(t, 2, 2)
	workers := &workerControlFixture{engine: fixture.engine, registered: map[int32]bool{0: true, 1: true}}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i))
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	membership, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	verifier := &WorkerVerifier{Membership: fixture.adapter}
	proof, err := verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.NoError(t, err)
	require.NotNil(t, proof.Proof)

	t.Log("reject retirement because this growth-only runtime cannot durably drain")
	result, err := traffic.Apply(t.Context(), groupID, domain.TrafficTarget{
		ControlRevision: 1, TransitionID: "retire", TopologyGeneration: 1,
		Drain: []domain.TrafficDrainTarget{{Membership: membership.CommittedTopology.Replicas[1], Mode: domain.TrafficDrainModeGraceful}},
	})
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "DrainUnsupported", result.Rejection.Reason)

	t.Log("a warming runtime cannot produce proof merely because the resize completed")
	workers.failProbe = true
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
	workers.failProbe = false

	t.Log("a stale native epoch cannot be presented as current registration or serving proof")
	workers.staleProof = true
	_, err = traffic.Observe(t.Context(), groupID)
	require.ErrorContains(t, err, "lacks current native serving evidence")
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
	workers.staleProof = false

	t.Log("a new native API lifetime invalidates the previous membership and serving proof")
	fixture.engine.mu.Lock()
	fixture.engine.states[0].InstanceID = "replacement"
	fixture.engine.mu.Unlock()
	_, err = verifier.Verify(t.Context(), groupID, membership.CommittedTopology)
	require.Error(t, err)
}

type workerControlFixture struct {
	mu         sync.Mutex
	engine     *nativeMembershipEngine
	registered map[int32]bool
	failProbe  bool
	staleProof bool
	mutations  int
}

func (w *workerControlFixture) RoundTrip(request *http.Request) (*http.Response, error) {
	var rank int32
	if _, err := fmt.Sscanf(request.URL.Hostname(), "rank-%d.test", &rank); err != nil {
		return nil, err
	}
	path := strings.TrimPrefix(request.URL.Path, "/engine/control")
	if path == statusPath || path == scalePath {
		copy := request.Clone(request.Context())
		copy.URL.Path = path
		return w.engine.handle(rank, copy)
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if path != "/engine_group_status" {
		w.mutations++
		return nil, fmt.Errorf("unexpected serving mutation %s", path)
	}
	w.engine.mu.Lock()
	binding := nativeBinding(w.engine.states[rank])
	w.engine.mu.Unlock()
	if w.staleProof {
		epoch := "stale-epoch"
		binding.Epoch = &epoch
	}
	status := workerStatus{
		GroupID: "group-uid", PodUID: domain.PodUID(fmt.Sprintf("pod-%d", rank)),
		MemberID: domain.NativeMemberID(fmt.Sprintf("dp-%d", rank)), Binding: &binding,
		ListenerReady: true, Verified: !w.failProbe,
		Admitted: w.registered[rank] && !w.failProbe,
	}
	payload, err := json.Marshal(status)
	if err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(payload)))}, nil
}

type workerTestCapacity struct {
	processes []Process
	count     int32
	revision  int64
}

func (a *workerTestCapacity) Observe(_ context.Context, _ domain.GroupID) (domain.CapacityObservation, error) {
	observation := domain.CapacityObservation{AppliedRevision: a.revision}
	for rank := int32(0); rank < a.count; rank++ {
		process := a.processes[rank]
		observation.Allocations = append(observation.Allocations, domain.CapacityAllocation{Available: true, Health: domain.AllocationHealthHealthy,
			Incarnation: domain.ReplicaIncarnation{ReplicaID: process.Membership.ReplicaID, SlotID: domain.CapacitySlotID(fmt.Sprintf("slot-%d", rank)), Members: process.Membership.Members, CapacityRefs: []domain.CapacityRef{{Name: fmt.Sprintf("pod-%d", rank), UID: process.PodUID}}},
		})
	}
	return observation, nil
}

func (a *workerTestCapacity) Apply(_ context.Context, _ domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	a.count, a.revision = int32(len(target.Replicas)), target.ControlRevision
	return domain.ApplyResult{}, nil
}
