/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"encoding/json"
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestVLLMInitialAndJoiningLaunchAssignments(t *testing.T) {
	t.Log("resolve one immutable template for eight initial TP4 allocations")
	source := newTestVLLMProfileGeometrySource()
	source.Args = append(source.Args, "-dpa", "original.test.svc", "-dpp=29700")
	profile, err := ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	group := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
		Name: "world", Namespace: "test", UID: "group-uid",
		Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupProfile: profile.Geometry.Fingerprint},
	}}
	before, err := json.Marshal(source)
	require.NoError(t, err)

	t.Log("form all initial participants at the same DP8 size without any resize operation")
	for rank := range int32(8) {
		pod := newVLLMLaunchPod(rank)
		assignment, err := RenderVLLMLaunchAssignment(group, pod, source, "primary.test.svc", 29550, nil)
		require.NoError(t, err)
		assert.Equal(t, rank, assignment.Rank)
		assert.Equal(t, int32(8), assignment.DataParallelSize)
		assert.Equal(t, "initial", assignment.Mode)
		assert.Empty(t, assignment.TransitionID)
		assert.Equal(t, "0", assignment.Environment["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"])
		values, _, err := collectVLLMProfileOptions(assignment.Command[:2], assignment.Command[2:])
		require.NoError(t, err)
		assert.Equal(t, fmt.Sprint(rank), values["--data-parallel-rank"])
		assert.Equal(t, "8", values["--data-parallel-size"])
		assert.Equal(t, "primary.test.svc", values["--data-parallel-address"])
		assert.Equal(t, "29550", values["--data-parallel-rpc-port"])
	}

	t.Log("give both joiners the complete operation target, never rank plus one")
	target := enginegroup.CapacityTarget{
		ControlRevision: 4, TransitionID: "grow-ten", ProfileFingerprint: profile.Geometry.Fingerprint,
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas:              make([]enginegroup.CapacityReplicaTarget, 10),
	}
	for rank := range int32(10) {
		target.Replicas[rank] = enginegroup.CapacityReplicaTarget{
			ReplicaID: enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)), SlotID: enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
			Bootstrap: &enginegroup.CapacityBootstrap{
				Mode: enginegroup.BootstrapModeJoin, BaseTopologyGeneration: 2,
				NativeMembers: []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank))},
			},
		}
	}
	for _, rank := range []int32{8, 9} {
		assignment, err := RenderVLLMLaunchAssignment(group, newVLLMLaunchPod(rank), source, "primary.test.svc", 29550, &target)
		require.NoError(t, err)
		assert.Equal(t, int32(10), assignment.DataParallelSize)
		assert.Equal(t, "join", assignment.Mode)
		assert.Equal(t, target.TransitionID, assignment.TransitionID)
		assert.Equal(t, target.ControlRevision, assignment.ControlRevision)
		assert.Equal(t, int64(2), assignment.BaseTopologyGeneration)
		assert.Equal(t, "1", assignment.Environment["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"])
	}

	t.Log("retain the original profile source without modifying its startup geometry")
	after, err := json.Marshal(source)
	require.NoError(t, err)
	assert.JSONEq(t, string(before), string(after))
}

func TestVLLMLaunchRejectsUnboundCapacity(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode string
	}{
		{"a joining index cannot self-bootstrap an initial world", "unplanned"},
		{"a different profile cannot receive the launch", "profile"},
		{"the slot must be canonical", "slot"},
		{"launch must not invoke an unrelated Dynamo module", "wrapper"},
		{"projected arguments are not kubelet interpolation", "expansion"},
		{"a joining operation must carry its correlation", "correlation"},
		{"a retained incarnation cannot be relaunched", "incarnation"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("construct a fresh binding with one unsafe launch property")
			source := newTestVLLMProfileGeometrySource()
			profile, err := ResolveVLLMElasticEPProfile(source)
			require.NoError(t, err)
			group := &api.DynamoGraphDeploymentEngineGroup{ObjectMeta: metav1.ObjectMeta{
				Name: "world", Namespace: "test", UID: "group-uid",
				Annotations: map[string]string{consts.KubeAnnotationDynamoEngineGroupProfile: profile.Geometry.Fingerprint},
			}}
			pod := newVLLMLaunchPod(0)
			var target *enginegroup.CapacityTarget
			switch tc.mode {
			case "unplanned":
				pod = newVLLMLaunchPod(8)
			case "profile":
				group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] = "another-profile"
			case "slot":
				pod.Labels[grovecommon.LabelPodCliquePodIndex] = "00"
			case "wrapper":
				source.Command = []string{"python3", "-m", "dynamo.frontend"}
			case "expansion":
				source.Args = append(source.Args, "--served-model-name", "$(MODEL)")
			case "correlation", "incarnation":
				pod = newVLLMLaunchPod(8)
				target = &enginegroup.CapacityTarget{
					ProfileFingerprint: profile.Geometry.Fingerprint, ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
					Replicas: make([]enginegroup.CapacityReplicaTarget, 9),
				}
				target.Replicas[8] = enginegroup.CapacityReplicaTarget{
					ReplicaID: "replica-8", SlotID: "slot-8", Bootstrap: &enginegroup.CapacityBootstrap{
						Mode: enginegroup.BootstrapModeJoin, BaseTopologyGeneration: 1, NativeMembers: []enginegroup.NativeMemberID{"dp-8"},
					},
				}
				if tc.mode == "incarnation" {
					target.Replicas[8].Incarnation = &enginegroup.ReplicaIncarnation{ReplicaID: "replica-8"}
				}
			}

			t.Log("reject the assignment without granting launch or membership permission")
			_, err = RenderVLLMLaunchAssignment(group, pod, source, "primary.test.svc", 29550, target)
			require.Error(t, err)
		})
	}
}

func TestVLLMLaunchTemplateSeparatesAllocationAndInferenceReadiness(t *testing.T) {
	t.Log("render a single template with no role-dependent command or engine readiness probe")
	template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Name: consts.MainContainerName, Command: []string{"vllm", "serve"}, Args: []string{"model"},
	}}}}
	before := template.DeepCopy()
	rendered, err := RenderVLLMLaunchPodTemplate(template, "world", "fingerprint")
	require.NoError(t, err)
	assert.Equal(t, before, template)
	assert.Equal(t, corev1.RestartPolicyNever, rendered.Spec.RestartPolicy)
	main := rendered.Spec.Containers[0]
	assert.Equal(t, []string{"python3", "-m", VLLMElasticEPBootstrapModule}, main.Command)
	assert.Equal(t, []string{"python3", "-m", VLLMElasticEPBootstrapModule, "--check"}, main.ReadinessProbe.Exec.Command)
	assert.Nil(t, main.ReadinessProbe.HTTPGet)
	assert.Nil(t, main.LivenessProbe)
	require.Len(t, rendered.Spec.Volumes, 2)
	assert.Contains(t, rendered.Spec.Volumes[0].DownwardAPI.Items[0].FieldRef.FieldPath, "dynamo-vllm-launch")

	t.Log("reject readiness overrides that would silently conflate allocation and inference")
	template.Spec.Containers[0].ReadinessProbe = &corev1.Probe{}
	_, err = RenderVLLMLaunchPodTemplate(template, "world", "fingerprint")
	require.ErrorContains(t, err, "caller-owned probes")
}

func newVLLMLaunchPod(rank int32) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name: fmt.Sprintf("pod-%d", rank), Namespace: "test", UID: "pod-uid",
		Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: "world", grovecommon.LabelPodCliquePodIndex: fmt.Sprint(rank)},
	}}
}
