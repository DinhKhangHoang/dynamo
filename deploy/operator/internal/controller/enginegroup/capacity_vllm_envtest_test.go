//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"encoding/json"
	"fmt"
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/grovecapacity"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestVLLMGroveLaunchDeliveryUsesRealScaleAndPodUIDs(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx := t.Context()

	t.Log("declare a native TP1/DP2 world with capacity reserved for DP3")
	source := dynamo.VLLMProfileGeometrySource{
		Command: []string{"vllm", "serve", "model"},
		Args: []string{
			"--data-parallel-size=2", "--data-parallel-rank=0", "--data-parallel-size-local=1",
			"--tensor-parallel-size=1", "--distributed-executor-backend=mp", "--elastic-ep-max-dp-size=3",
			"--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep",
		},
		InitialReplicas: 2, MainContainerGPUs: 1, DedicatedMainGPUAllocation: true,
		WorkloadRevisionDigest: "test-image-model",
	}
	profile, err := dynamo.ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	dgd := &api.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "deployment", Namespace: env.Namespace(), UID: "dgd-uid"}}
	component := &api.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2}}
	group := dynamo.RenderVLLMEngineGroup(dgd, component, 0, profile)
	group.Finalizers = []string{engineGroupFinalizer}
	require.NoError(t, env.Client().Create(ctx, group))
	template, err := dynamo.RenderVLLMLaunchPodTemplate(&corev1.PodTemplateSpec{Spec: corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test-vllm:latest"}},
	}}, group.Name, profile.Geometry.Fingerprint)
	require.NoError(t, err)
	clique := &grove.PodClique{
		ObjectMeta: metav1.ObjectMeta{Name: "vllm-members", Namespace: env.Namespace(),
			Labels: map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: grove.SchemeGroupVersion.String(), Kind: "PodCliqueScalingGroup", Name: "world", UID: "world-uid", Controller: ptr.To(true),
			}},
		},
		Spec: grove.PodCliqueSpec{Replicas: 2, MinAvailable: ptr.To(int32(2)), PodSpec: template.Spec},
	}
	require.NoError(t, env.Client().Create(ctx, clique))
	capacity := &VLLMGroveCapacity{
		Grove: &grovecapacity.Adapter{
			Client: env.Client(), Clique: client.ObjectKeyFromObject(clique), CliqueUID: clique.UID, GroupName: group.Name,
			Journal: kubejournal.NewStore(env.Client(), group.Namespace, group.Name, group.UID, "grove-capacity"),
		},
		Group: group, Source: source, RendezvousAddress: "primary.test.svc", RPCPort: 29550,
	}
	pods := make([]*corev1.Pod, 2)
	for rank := range int32(2) {
		pod := newVLLMGroveLaunchPod(clique, rank)
		require.NoError(t, env.Client().Create(ctx, pod))
		pods[rank] = pod
	}

	t.Log("publish initial commands while neither engine API nor Pod readiness exists")
	require.NoError(t, capacity.ReconcileLaunches(ctx, nil))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(group), group))
	for _, pod := range pods {
		require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(pod), pod))
		assert.Empty(t, pod.Annotations[vllmruntime.LaunchAnnotation], "slot authority is persisted before command publication")
	}
	require.NoError(t, capacity.ReconcileLaunches(ctx, nil))
	for rank, pod := range pods {
		require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(pod), pod))
		var assignment vllmruntime.LaunchAssignment
		require.NoError(t, json.Unmarshal([]byte(pod.Annotations[vllmruntime.LaunchAnnotation]), &assignment))
		assert.Equal(t, int32(rank), assignment.Rank)
		assert.Equal(t, int32(2), assignment.DataParallelSize)
		assert.Equal(t, "initial", assignment.Mode)
		assert.Equal(t, string(pod.UID), assignment.PodUID)
	}
	observed, err := capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	require.Len(t, observed.Allocations, 2)
	assert.False(t, observed.Allocations[0].Available)
	assert.False(t, observed.Allocations[1].Available)
	target := domain.CapacityTarget{
		ControlRevision: 1, TransitionID: "grow-three", ProfileFingerprint: profile.Geometry.Fingerprint,
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
	}
	for _, allocation := range observed.Allocations {
		incarnation := allocation.Incarnation
		target.Replicas = append(target.Replicas, domain.CapacityReplicaTarget{
			ReplicaID: incarnation.ReplicaID, SlotID: incarnation.SlotID, Incarnation: &incarnation,
		})
	}
	target.Replicas = append(target.Replicas, domain.CapacityReplicaTarget{
		ReplicaID: "replica-2", SlotID: "slot-2", Bootstrap: &domain.CapacityBootstrap{
			Mode: domain.BootstrapModeJoin, BaseTopologyGeneration: 1, NativeMembers: []domain.NativeMemberID{"dp-2"},
		},
	})

	t.Log("scale the native member clique through its real Scale subresource without changing its template")
	result, err := capacity.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(clique), clique))
	assert.Equal(t, int32(3), clique.Spec.Replicas)
	assert.Equal(t, template.Spec, clique.Spec.PodSpec)
	joiner := newVLLMGroveLaunchPod(clique, 2)
	require.NoError(t, env.Client().Create(ctx, joiner))

	t.Log("replay after Grove creates the joiner and deliver the complete correlated DP3 launch")
	result, err = capacity.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(group), group))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(joiner), joiner))
	assert.Empty(t, joiner.Annotations[vllmruntime.LaunchAnnotation], "the joining UID is durably bound before publication")
	result, err = capacity.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(joiner), joiner))
	var assignment vllmruntime.LaunchAssignment
	require.NoError(t, json.Unmarshal([]byte(joiner.Annotations[vllmruntime.LaunchAnnotation]), &assignment))
	assert.Equal(t, int32(3), assignment.DataParallelSize)
	assert.Equal(t, "join", assignment.Mode)
	assert.Equal(t, target.ControlRevision, assignment.ControlRevision)
	assert.Equal(t, target.TransitionID, assignment.TransitionID)
	assert.Equal(t, string(joiner.UID), assignment.PodUID)
	observed, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	require.Len(t, observed.Allocations, 3)
	assert.False(t, observed.Allocations[2].Available, "metadata publication alone is not bootstrap consumption")

	t.Log("initial reconciliation and completed-target replay must not relaunch any retained process")
	version := joiner.ResourceVersion
	require.NoError(t, capacity.ReconcileLaunches(ctx, nil))
	result, err = capacity.Apply(ctx, domain.GroupID(group.UID), target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(joiner), joiner))
	assert.Equal(t, version, joiner.ResourceVersion)

	t.Log("missing launch authority for a retained process fails closed instead of reinitializing it")
	joiner.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	require.NoError(t, env.Client().Status().Update(ctx, joiner))
	observed, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.NoError(t, err)
	pinned := target
	pinned.ControlRevision = 2
	pinned.Replicas = append([]domain.CapacityReplicaTarget(nil), target.Replicas...)
	for _, allocation := range observed.Allocations {
		if allocation.Incarnation.ReplicaID == "replica-2" {
			incarnation := allocation.Incarnation
			pinned.Replicas[2].Bootstrap = nil
			pinned.Replicas[2].Incarnation = &incarnation
		}
	}
	result, err = capacity.Apply(ctx, domain.GroupID(group.UID), pinned)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(joiner), joiner))
	before := joiner.DeepCopy()
	delete(joiner.Annotations, vllmruntime.LaunchAnnotation)
	require.NoError(t, env.Client().Patch(ctx, joiner, client.MergeFrom(before)))
	_, err = capacity.Observe(ctx, domain.GroupID(group.UID))
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
	_, err = capacity.Apply(ctx, domain.GroupID(group.UID), pinned)
	require.ErrorIs(t, err, domain.ErrRecoveryRequired)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(joiner), joiner))
	assert.Empty(t, joiner.Annotations[vllmruntime.LaunchAnnotation], "an exact incarnation does not authorize another bootstrap")

	t.Log("a replacement cannot inherit the old assignment even through a stale publication")
	joiner.Finalizers = nil // No engine runs in envtest; simulate completed teardown by the lifecycle owner.
	require.NoError(t, env.Client().Update(ctx, joiner))
	old := joiner.DeepCopy()
	require.NoError(t, env.Client().Delete(ctx, joiner, client.GracePeriodSeconds(0)))
	replacement := newVLLMGroveLaunchPod(clique, 2)
	require.NoError(t, env.Client().Create(ctx, replacement))
	assert.NotEqual(t, old.UID, replacement.UID)
	old.Annotations = nil
	require.Error(t, vllmruntime.PublishLaunch(ctx, env.Client(), old, assignment))
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(replacement), replacement))
	assert.Empty(t, replacement.Annotations[vllmruntime.LaunchAnnotation])

	t.Log("startup replay cannot turn a replacement initial participant into a fresh bootstrap")
	initial := pods[1]
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(initial), initial))
	initial.Finalizers = nil
	require.NoError(t, env.Client().Update(ctx, initial))
	require.NoError(t, env.Client().Delete(ctx, initial, client.GracePeriodSeconds(0)))
	initialReplacement := newVLLMGroveLaunchPod(clique, 1)
	require.NoError(t, env.Client().Create(ctx, initialReplacement))
	require.ErrorIs(t, capacity.ReconcileLaunches(ctx, nil), domain.ErrRecoveryRequired)
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(initialReplacement), initialReplacement))
	assert.Empty(t, initialReplacement.Annotations[vllmruntime.LaunchAnnotation])

	t.Log("loss of the group's launch catalog is not interpreted as uninitialized startup")
	require.NoError(t, env.Client().Get(ctx, client.ObjectKeyFromObject(group), group))
	groupBefore := group.DeepCopy()
	delete(group.Annotations, vllmruntime.LaunchBindingsAnnotation)
	require.NoError(t, env.Client().Patch(ctx, group, client.MergeFrom(groupBefore)))
	require.ErrorIs(t, capacity.ReconcileLaunches(ctx, nil), domain.ErrRecoveryRequired)
}

func newVLLMGroveLaunchPod(clique *grove.PodClique, rank int32) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("allocation-%d", rank), Namespace: clique.Namespace,
			Finalizers: []string{engineGroupProcessFinalizer},
			Labels: map[string]string{
				grovecommon.LabelPodClique: clique.Name, grovecommon.LabelPodCliquePodIndex: fmt.Sprint(rank),
				consts.KubeLabelDynamoEngineGroup:         clique.Labels[consts.KubeLabelDynamoEngineGroup],
				consts.KubeLabelDynamoScaleRepresentative: consts.KubeLabelDynamoScaleRepresentativeYes,
			},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(clique, grove.SchemeGroupVersion.WithKind("PodClique"))},
		},
		Spec: *clique.Spec.PodSpec.DeepCopy(),
	}
}
