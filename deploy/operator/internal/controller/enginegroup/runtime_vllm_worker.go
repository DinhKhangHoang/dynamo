/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"net/http"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// VLLMWorkerRuntimeProvider composes Grove and the existing Python worker's native controls.
// Client and HTTP must not be nil. Resolution is read-only, including before first formation.
type VLLMWorkerRuntimeProvider struct {
	Client client.Client
	HTTP   *http.Client
}

// Resolve binds physical allocation and native membership to runtime-owned serving observations.
func (p *VLLMWorkerRuntimeProvider) Resolve(ctx context.Context, group *api.DynamoGraphDeploymentEngineGroup) (Runtime, error) {
	// Native-server and sidecar deployments need supported remote serving observations.
	if group.Labels[consts.KubeLabelDynamoEngineGroupRuntime] != string(dynamo.BackendFrameworkVLLM) {
		return Runtime{}, fmt.Errorf("%w: the default Grove runtime supports the vLLM Python worker only", ErrRuntimeUnavailable)
	}
	capacity, err := ResolveVLLMGroveCapacity(ctx, p.Client, group)
	if err != nil {
		return Runtime{}, err
	}
	if !dynamo.IsVLLMWorkerLaunch(capacity.Source) {
		return Runtime{}, fmt.Errorf("%w: vLLM managed growth requires the Dynamo Python worker", ErrRuntimeUnavailable)
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(capacity.Source)
	if err != nil {
		return Runtime{}, err
	}
	membership, err := ResolveVLLMMembership(ctx, capacity, p.HTTP)
	if err != nil {
		return Runtime{}, err
	}
	membership.DrainTimeout = 120

	// The traffic journal fences withdrawal revisions; it never grants registration.
	traffic := &vllmruntime.WorkerTraffic{
		Membership: membership,
		Journal:    kubejournal.NewStore(p.Client, group.Namespace, group.Name, group.UID, "vllm-traffic-"+string(capacity.Grove.CliqueUID)),
	}
	return Runtime{
		Profile: api.EngineGroupProfileStatus{
			Backend: "vllm", Fingerprint: profile.Geometry.Fingerprint,
			GPUsPerReplica: profile.Geometry.GPUsPerReplica, PodsPerReplica: profile.Geometry.PodsPerReplica,
			NativeMembersPerReplica: 1, MinSafeServingNativeMembers: profile.InitialReplicas,
			MinSupportedReplicas: profile.InitialReplicas, MaxSupportedReplicas: profile.MaximumReplicas,
		},
		Capacity: capacity, Membership: membership, Traffic: traffic,
		Verifier: &vllmruntime.WorkerVerifier{Membership: membership},
		Planner:  vllmGrowthPlanner{fingerprint: profile.Geometry.Fingerprint},
	}, nil
}

type vllmGrowthPlanner struct{ fingerprint string }

func (p vllmGrowthPlanner) ResolveScalePlan(_ context.Context, _ domain.GroupID, target int32, status domain.GroupStatus) (ScalePlanResolution, error) {
	// One immutable plan covers the entire contiguous append, not a hardcoded PoC target.
	topology, found := status.Topologies.Current()
	if !found {
		return ScalePlanResolution{}, fmt.Errorf("growth planning requires committed membership")
	}
	if target == topology.ReplicaCount() {
		return ScalePlanResolution{}, nil
	}
	if target < topology.ReplicaCount() {
		return ScalePlanResolution{Rejection: &domain.Failure{Classification: domain.FailureClassificationTerminal, Reason: "ShrinkUnsupported", Message: "the Grove vLLM worker integration currently supports growth only"}}, nil
	}
	if status.Transition != nil && status.Transition.Outcome != domain.TransitionOutcomeCompleted && status.Transition.Outcome != domain.TransitionOutcomeRolledBack {
		return ScalePlanResolution{}, nil
	}
	joining := make([]domain.ReplicaTarget, 0, target-topology.ReplicaCount())
	for rank := topology.ReplicaCount(); rank < target; rank++ {
		joining = append(joining, domain.ReplicaTarget{
			ReplicaID: domain.ReplicaID(fmt.Sprintf("replica-%d", rank)), SlotID: domain.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
			Bootstrap: domain.BootstrapModeJoin, NativeMembers: []domain.NativeMemberID{domain.NativeMemberID(fmt.Sprintf("dp-%d", rank))},
		})
	}
	return ScalePlanResolution{Plan: &domain.ResolvedPlan{
		ID: fmt.Sprintf("grow-g%d-to-%d", topology.Generation, target), ProfileFingerprint: p.fingerprint,
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator, TrafficRequirement: domain.TrafficRequirementKeepServing,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change:                  domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{Replicas: joining}},
	}}, nil
}
