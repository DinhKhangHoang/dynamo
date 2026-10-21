/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"k8s.io/utils/ptr"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

// NativeBinding is engine-observed evidence, not a Dynamo topology generation.
type NativeBinding struct {
	InstanceID       string  `json:"instance_id"`
	OperationID      *string `json:"operation_id"`
	Epoch            *string `json:"epoch"`
	DataParallelSize int32   `json:"data_parallel_size"`
}

type workerStatus struct {
	GroupID       enginegroup.GroupID        `json:"group_id"`
	PodUID        enginegroup.PodUID         `json:"pod_uid"`
	MemberID      enginegroup.NativeMemberID `json:"member_id"`
	Binding       *NativeBinding             `json:"binding"`
	Verified      bool                       `json:"verified"`
	Admitted      bool                       `json:"admitted"`
	ListenerReady bool                       `json:"listener_ready"`
}

func (c *Client) workerRequest(ctx context.Context, path string, result any) error {
	// Serving observation remains reachable independently of inference registration.
	if c.routePrefix != "/engine/control" {
		return fmt.Errorf("worker observation requires the Dynamo worker transport")
	}
	ctx, cancel := context.WithTimeout(ctx, c.ObservationTimeout)
	defer cancel()
	payload, err := c.request(ctx, path, nil)
	if err != nil {
		return err
	}
	return json.Unmarshal(payload, result)
}

func nativeBinding(status Status) NativeBinding {
	return NativeBinding{InstanceID: status.InstanceID, OperationID: status.OperationID, Epoch: status.Epoch, DataParallelSize: status.EffectiveDataParallelSize}
}

func (a *MembershipAdapter) servingBindings(ctx context.Context, groupID enginegroup.GroupID, topology enginegroup.MembershipTopology) ([]NativeBinding, error) {
	// Correlate runtime-owned serving proof with the engine's exact committed topology.
	if _, err := a.Observe(ctx, groupID, ""); err != nil {
		return nil, err
	}
	state, _, err := a.load(ctx)
	if err != nil {
		return nil, err
	}
	if !enginegroup.SameTopology(state.Topology, topology) || (state.Target != nil && state.Result == nil) {
		return nil, fmt.Errorf("serving evidence requires exact committed membership")
	}
	statuses, err := a.readTopology(ctx, topology, state.Instances)
	if err != nil {
		return nil, err
	}
	bindings := make([]NativeBinding, len(statuses))
	for i, status := range statuses {
		if status.Phase != PhaseIdle && status.Phase != PhaseCompleted {
			return nil, ErrTopologyUnknown
		}
		bindings[i] = nativeBinding(status)
	}
	return bindings, nil
}

// WorkerTraffic observes autonomous registration in the growth-only worker profile.
// It never sends positive permissions. Retirement and drain remain unsupported
// until the runtime can durably maintain negative intent and prove completion.
type WorkerTraffic struct {
	Membership *MembershipAdapter
	Journal    kubejournal.Store
}

type workerTrafficState struct {
	Target enginegroup.TrafficTarget
}

// Observe reports current registration even before any controller traffic target exists.
func (a *WorkerTraffic) Observe(ctx context.Context, groupID enginegroup.GroupID) (enginegroup.TrafficObservation, error) {
	// Observe only committed members; a joiner need not expose serving controls before native preparation.
	membership, err := a.Membership.Observe(ctx, groupID, "")
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	state := workerTrafficState{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	observation := enginegroup.TrafficObservation{}
	if snapshot.Exists() {
		observation.AppliedRevision = state.Target.ControlRevision
	}
	binding, _, err := a.Membership.load(ctx)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	for _, member := range membership.CommittedTopology.Replicas {
		process, err := a.Membership.processForMember(member.Members[0].ID)
		if err != nil {
			return enginegroup.TrafficObservation{}, err
		}
		status, err := observeWorker(ctx, groupID, process)
		if err != nil {
			return enginegroup.TrafficObservation{}, err
		}
		if status.Admitted {
			if !status.Verified || !status.ListenerReady || status.Binding == nil ||
				status.Binding.InstanceID != binding.Instances[process.Rank] ||
				ptr.Deref(status.Binding.OperationID, "") != binding.NativeOperationID ||
				ptr.Deref(status.Binding.Epoch, "") != binding.NativeEpoch ||
				status.Binding.DataParallelSize != membership.CommittedTopology.NativeMemberCount() {
				return enginegroup.TrafficObservation{}, fmt.Errorf("registered worker lacks current native serving evidence")
			}
			observation.Admitted = append(observation.Admitted, process.Membership)
		}
	}
	return observation, nil
}

// Apply accepts clearing withdrawal intent but cannot fabricate drain support.
func (a *WorkerTraffic) Apply(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.TrafficTarget) (enginegroup.ApplyResult, error) {
	// Reject retirement rather than silently losing its durable withdrawal requirements.
	if groupID == "" || groupID != enginegroup.GroupID(a.Journal.Owner.UID) ||
		target.ControlRevision <= 0 || target.TopologyGeneration <= 0 || len(target.Drain) != 0 {
		return enginegroup.ApplyResult{Rejection: terminalFailure("DrainUnsupported", "the growth-only worker cannot maintain retirement/drain intent")}, nil
	}
	state := workerTrafficState{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if snapshot.Exists() && (target.ControlRevision < state.Target.ControlRevision ||
		target.ControlRevision == state.Target.ControlRevision && !reflect.DeepEqual(target, state.Target)) {
		return enginegroup.ApplyResult{Rejection: terminalFailure("ConflictingTrafficRevision", "traffic revision is stale or has another payload")}, nil
	}
	// Acceptance does not trigger registration; the worker already owns that lifecycle.
	_, err = a.Journal.Save(ctx, snapshot, workerTrafficState{Target: target})
	return enginegroup.ApplyResult{}, err
}

// WorkerVerifier observes proof already established by the serving runtime.
// It cannot initiate warmup or grant registration.
type WorkerVerifier struct{ Membership *MembershipAdapter }

// Verify correlates independently produced proof with pre/post native membership evidence.
func (v *WorkerVerifier) Verify(ctx context.Context, groupID enginegroup.GroupID, topology enginegroup.MembershipTopology) (enginegroup.VerificationResult, error) {
	bindings, err := v.Membership.servingBindings(ctx, groupID, topology)
	if err != nil {
		return enginegroup.VerificationResult{}, err
	}
	for rank, binding := range bindings {
		process, err := v.Membership.process(int32(rank))
		if err != nil {
			return enginegroup.VerificationResult{}, err
		}
		status, err := observeWorker(ctx, groupID, process)
		if err != nil {
			return enginegroup.VerificationResult{}, err
		}
		if !status.Verified || !reflect.DeepEqual(status.Binding, &binding) {
			return enginegroup.VerificationResult{}, fmt.Errorf("rank %d has not established serving proof for the current topology", rank)
		}
	}
	after, err := v.Membership.servingBindings(ctx, groupID, topology)
	if err != nil {
		return enginegroup.VerificationResult{}, err
	}
	if !reflect.DeepEqual(bindings, after) {
		return enginegroup.VerificationResult{}, fmt.Errorf("native topology changed during serving observation")
	}
	return enginegroup.VerificationResult{Proof: &enginegroup.ServingProof{
		TopologyGeneration: topology.Generation, RuntimeDigest: enginegroup.TopologyRuntimeDigest(topology),
	}}, nil
}

func observeWorker(ctx context.Context, groupID enginegroup.GroupID, process Process) (workerStatus, error) {
	status := workerStatus{}
	if err := process.Client.workerRequest(ctx, "/engine_group_status", &status); err != nil {
		return status, err
	}
	if status.GroupID != groupID || status.PodUID != process.PodUID ||
		status.MemberID != process.Membership.Members[0].ID || process.PodUID == "" {
		return status, fmt.Errorf("worker serving observation belongs to another incarnation")
	}
	return status, nil
}

func (a *MembershipAdapter) processForMember(member enginegroup.NativeMemberID) (Process, error) {
	rank, err := nativeRank(member)
	if err != nil {
		return Process{}, err
	}
	return a.process(rank)
}
