//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVLLMWorkerRuntimeResolvesBeforeBootstrapWithoutNativeRequests(t *testing.T) {
	t.Log("resolve the production provider from the projected Pod cache before APIs or assignments exist")
	kube, group := newVLLMRuntimeTestWorldForTransport(t, true)
	transport := &vllmNoNativeRequests{}
	provider := &VLLMWorkerRuntimeProvider{Client: kube, HTTP: &http.Client{Transport: transport}}
	runtime, err := provider.Resolve(t.Context(), group)
	require.NoError(t, err)
	require.NoError(t, validateEngineGroupRuntime(runtime))
	assert.Empty(t, transport.hosts)
	assert.Equal(t, int32(2), runtime.Profile.MinSupportedReplicas)
	assert.Equal(t, int32(4), runtime.Profile.MaxSupportedReplicas)

	t.Log("bootstrap is a separate lifecycle-fenced capacity effect, not membership observation")
	initial, ok := runtime.Capacity.(InitialCapacityReconciler)
	require.True(t, ok)
	require.NoError(t, initial.ReconcileInitialCapacity(t.Context(), domain.GroupID(group.UID)))
	assert.Empty(t, transport.hosts)

	t.Log("native-only launches cannot silently substitute logical admission for real discovery")
	nativeKube, nativeGroup := newVLLMRuntimeTestWorld(t)
	provider.Client = nativeKube
	_, err = provider.Resolve(t.Context(), nativeGroup)
	require.ErrorIs(t, err, ErrRuntimeUnavailable)

	t.Log("other backends remain explicitly unavailable instead of entering vLLM resolution")
	group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] = "sglang"
	_, err = provider.Resolve(t.Context(), group)
	require.ErrorIs(t, err, ErrRuntimeUnavailable)
}

func TestVLLMWorkerPlannerSupportsAbsoluteGrowthOnly(t *testing.T) {
	t.Log("use immutable topology identity and a fixed profile rather than a PoC-sized increment")
	planner := vllmGrowthPlanner{fingerprint: "profile"}
	status := domain.GroupStatus{Topologies: domain.TopologyHistory{
		CurrentGeneration: 1,
		Snapshots: []domain.MembershipTopology{{Generation: 1, Replicas: []domain.ReplicaMembership{
			{ReplicaID: "replica-0"}, {ReplicaID: "replica-1"},
		}}},
	}}
	for _, target := range []int32{1, 2, 3, 4} {
		t.Logf("resolve absolute target %d against the two-replica base", target)
		result, err := planner.ResolveScalePlan(t.Context(), "group", target, status)
		require.NoError(t, err)
		switch {
		case target < 2:
			require.NotNil(t, result.Rejection)
			assert.Equal(t, "ShrinkUnsupported", result.Rejection.Reason)
		case target == 2:
			assert.Nil(t, result.Plan)
			assert.Nil(t, result.Rejection)
		default:
			require.NotNil(t, result.Plan)
			assert.Len(t, result.Plan.Change.Grow.Replicas, int(target-2))
			assert.Equal(t, domain.VerificationRequirementRequired, result.Plan.VerificationRequirement)
		}
	}
}
