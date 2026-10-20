/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// VLLMGroveCapacity composes Grove allocation with Pod-bound native launch delivery.
// Grove and Group must not be nil. Source is the immutable native launch declaration,
// not the rendered launch-only entrypoint. Effects require the existing Group and Pod lifecycle finalizers.
// This adapter does not implement traffic admission.
// Do not register a vLLM Runtime until the worker admission and serving adapters are available.
type VLLMGroveCapacity struct {
	Grove             *grovecapacity.Adapter
	Group             *api.DynamoGraphDeploymentEngineGroup
	Source            dynamo.VLLMProfileGeometrySource
	RendezvousAddress string
	RPCPort           int32
}

// ReconcileInitialCapacity delivers initial assignments under the exact logical-world authority.
func (a *VLLMGroveCapacity) ReconcileInitialCapacity(ctx context.Context, groupID domain.GroupID) error {
	// A restarted controller may replay publication, but never initialize another world's slot.
	if groupID == "" || groupID != domain.GroupID(a.Group.UID) {
		return fmt.Errorf("initial capacity belongs to another Engine Group")
	}
	return a.ReconcileLaunches(ctx, nil)
}

// Observe remains read-only and reports allocation readiness independently of native API readiness.
// The launch receipt probe makes a joiner available before its API can wait on membership preparation.
func (a *VLLMGroveCapacity) Observe(ctx context.Context, groupID domain.GroupID) (domain.CapacityObservation, error) {
	if groupID != domain.GroupID(a.Group.UID) || groupID == "" {
		return domain.CapacityObservation{}, fmt.Errorf("vLLM capacity observation belongs to a different Engine Group")
	}
	if _, err := a.launchBindings(); err != nil {
		return domain.CapacityObservation{}, err
	}
	observation, err := a.Grove.Observe(ctx, groupID)
	if err != nil {
		return domain.CapacityObservation{}, err
	}
	for _, allocation := range observation.Allocations {
		if !allocation.Available {
			continue
		}
		ref := allocation.Incarnation.CapacityRefs[0]
		pod := &corev1.Pod{}
		if err := a.Grove.Client.Get(ctx, client.ObjectKey{Namespace: a.Group.Namespace, Name: ref.Name}, pod); err != nil {
			return domain.CapacityObservation{}, err
		}
		if domain.PodUID(pod.UID) != ref.UID {
			return domain.CapacityObservation{}, fmt.Errorf("%w: vLLM Pod changed during availability observation", domain.ErrRecoveryRequired)
		}
		if err := a.validateRetainedLaunch(pod); err != nil {
			return domain.CapacityObservation{}, err
		}
	}
	return observation, nil
}

// Apply reasserts Grove's absolute capacity target and delivers only explicitly authorized joining launches.
// Publication can lag allocation: an accepted capacity revision never means every joiner has consumed its command.
func (a *VLLMGroveCapacity) Apply(ctx context.Context, groupID domain.GroupID, target domain.CapacityTarget) (domain.ApplyResult, error) {
	// Validate the immutable source before permitting a physical allocation effect.
	if groupID != domain.GroupID(a.Group.UID) || groupID == "" || !slices.Contains(a.Group.Finalizers, engineGroupFinalizer) {
		return domain.ApplyResult{}, fmt.Errorf("vLLM capacity effects require the exact lifecycle-owned Engine Group")
	}
	if _, err := a.launchBindings(); err != nil {
		return domain.ApplyResult{}, err
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(a.Source)
	if err != nil {
		return domain.ApplyResult{}, err
	}
	if target.ProfileFingerprint != profile.Geometry.Fingerprint ||
		a.Group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] != profile.Geometry.Fingerprint ||
		len(target.Replicas) > int(profile.MaximumReplicas) {
		return domain.ApplyResult{Rejection: &domain.Failure{
			Classification: domain.FailureClassificationTerminal, Reason: "InvalidVLLMCapacityTarget",
			Message: "capacity target does not match the immutable vLLM profile or reserved maximum",
		}}, nil
	}
	result, err := a.Grove.Apply(ctx, groupID, target)
	if err != nil || result.Rejection != nil {
		return result, err
	}
	return result, a.ReconcileLaunches(ctx, &target)
}

// ReconcileLaunches publishes assignments for existing Grove Pods, never creating Pods itself.
// A nil target delivers initial formation before engine observation, without a synthetic resize.
// Missing Pods are asynchronous Grove work; Pod watches and bounded reconciliation resume delivery.
func (a *VLLMGroveCapacity) ReconcileLaunches(ctx context.Context, target *domain.CapacityTarget) error {
	// Reuse the complete Grove UID/slot/restart validation before publishing any process intent.
	if !slices.Contains(a.Group.Finalizers, engineGroupFinalizer) {
		return fmt.Errorf("%w: vLLM bootstrap is waiting for Engine Group lifecycle ownership", ErrRuntimeUnavailable)
	}
	profile, err := dynamo.ResolveVLLMLaunchProfile(a.Source)
	if err != nil {
		return err
	}
	if _, err := a.launchBindings(); err != nil {
		return err
	}
	observation, err := a.Grove.Observe(ctx, domain.GroupID(a.Group.UID))
	if err != nil {
		return err
	}
	launches := make([]vllmPodLaunch, 0, len(observation.Allocations))
	for _, allocation := range observation.Allocations {
		if target == nil {
			rank, err := strconv.ParseInt(strings.TrimPrefix(string(allocation.Incarnation.SlotID), "slot-"), 10, 32)
			if err != nil {
				return fmt.Errorf("invalid Grove launch slot: %w", err)
			}
			if rank >= int64(profile.InitialReplicas) {
				continue
			}
		}
		ref := allocation.Incarnation.CapacityRefs[0]
		pod := &corev1.Pod{}
		if err := a.Grove.Client.Get(ctx, client.ObjectKey{Namespace: a.Group.Namespace, Name: ref.Name}, pod); err != nil {
			return fmt.Errorf("observe vLLM launch Pod: %w", err)
		}
		if domain.PodUID(pod.UID) != ref.UID {
			return fmt.Errorf("vLLM launch Pod changed after physical capacity observation")
		}
		if target != nil && !vllmSlotNeedsLaunch(*target, allocation.Incarnation.SlotID) {
			if err := a.validateRetainedLaunch(pod); err != nil {
				return err
			}
			continue
		}
		if !slices.Contains(pod.Finalizers, engineGroupProcessFinalizer) {
			continue // The normal Pod watch resumes launch after the Group controller protects this lifetime.
		}
		if pod.Spec.RestartPolicy != corev1.RestartPolicyNever {
			return fmt.Errorf("vLLM launch requires restartPolicy Never")
		}
		usesLaunchEntrypoint := false
		for _, container := range pod.Spec.Containers {
			if container.Name == consts.MainContainerName && slices.Equal(container.Command, []string{"python3", "-m", dynamo.VLLMElasticEPBootstrapModule}) && len(container.Args) == 0 {
				usesLaunchEntrypoint = true
			}
		}
		if !usesLaunchEntrypoint {
			return fmt.Errorf("vLLM launch Pod does not use the template-invariant native bootstrap")
		}
		assignment, err := dynamo.RenderVLLMLaunchAssignment(a.Group, pod, a.Source, a.RendezvousAddress, a.RPCPort, target)
		if err != nil {
			return err
		}
		launches = append(launches, vllmPodLaunch{pod: pod, assignment: assignment})
	}
	changed, err := a.bindLaunches(ctx, launches)
	if err != nil || changed {
		return err // The next cached Group observation carries publication authority.
	}
	for _, launch := range launches {
		if err := vllmruntime.PublishLaunch(ctx, a.Grove.Client, launch.pod, launch.assignment); err != nil {
			return err
		}
	}
	return nil
}

func (a *VLLMGroveCapacity) validateRetainedLaunch(pod *corev1.Pod) error {
	// Metadata loss is recovery-required, not permission to regenerate a command for an already-running member.
	assignment, err := vllmruntime.DecodeLaunchAssignment(pod.Annotations[vllmruntime.LaunchAnnotation])
	if err != nil {
		return fmt.Errorf("%w: Pod %s lost its vLLM launch binding: %w", domain.ErrRecoveryRequired, pod.Name, err)
	}
	if assignment.PodUID != string(pod.UID) || assignment.GroupUID != string(a.Group.UID) ||
		assignment.GroupName != a.Group.Name || assignment.ProfileFingerprint != a.Group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] ||
		strconv.FormatInt(int64(assignment.Rank), 10) != pod.Labels[grovecommon.LabelPodCliquePodIndex] {
		return fmt.Errorf("%w: Pod %s no longer matches its vLLM launch binding", domain.ErrRecoveryRequired, pod.Name)
	}
	bindings, err := a.launchBindings()
	if err != nil {
		return err
	}
	binding, err := vllmBindingForAssignment(assignment)
	if err != nil {
		return err
	}
	if bindings.Slots[domain.CapacitySlotID(fmt.Sprintf("slot-%d", assignment.Rank))] != binding {
		return fmt.Errorf("%w: Pod %s is not the accepted vLLM slot occupant", domain.ErrRecoveryRequired, pod.Name)
	}
	return nil
}

func vllmSlotNeedsLaunch(target domain.CapacityTarget, slot domain.CapacitySlotID) bool {
	for _, replica := range target.Replicas {
		if replica.SlotID == slot {
			return replica.Bootstrap != nil
		}
	}
	return false
}

var _ domain.CapacityAdapter = (*VLLMGroveCapacity)(nil)
