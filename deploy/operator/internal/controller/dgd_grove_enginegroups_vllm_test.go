/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	enginegroupcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDGDCreatesVLLMWorldAndPreservesScaleOwnership(t *testing.T) {
	for _, scenario := range []string{"create-child", "pods-ready-only", "engine-serving", "stale-health"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("declare a native DP2 world and its Grove ownership chain")
			dgd := &api.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "test", UID: "dgd-uid"},
				Spec: api.DynamoGraphDeploymentSpec{BackendFramework: string(dynamo.BackendFrameworkVLLM), Components: []api.DynamoComponentDeploymentSharedSpec{{
					ComponentName: "worker", ComponentType: api.ComponentTypeWorker, Replicas: ptr.To(int32(1)),
					EngineGroup: &api.ComponentEngineGroupSpec{InitialSize: 2},
					PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
						Name: consts.MainContainerName, Image: "vllm:native", Command: []string{"vllm", "serve", "model"},
						Args: []string{"--data-parallel-size=2", "--data-parallel-rank=0", "--data-parallel-size-local=1", "--elastic-ep-max-dp-size=4",
							"--distributed-executor-backend=mp", "--enable-expert-parallel", "--enable-eplb", "--enable-elastic-ep"},
						Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("1")}},
					}}}},
				}}}}
			before := dgd.DeepCopy()
			component := &dgd.Spec.Components[0]
			_, profile, err := dynamo.ResolveVLLMComponentEngineGroup(component, nil)
			require.NoError(t, err)
			group := dynamo.RenderVLLMEngineGroup(dgd, component, 0, profile)
			pcs := &grove.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Name: "pcs", Namespace: dgd.Namespace, UID: "pcs-uid"}}
			world := &grove.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "world", Namespace: dgd.Namespace, UID: "world-uid",
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grove.SchemeGroupVersion.WithKind("PodCliqueSet"))}}}
			clique := &grove.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "members", Namespace: dgd.Namespace, UID: "clique-uid",
				Labels:          map[string]string{consts.KubeLabelDynamoEngineGroup: group.Name, grovecommon.LabelPodCliqueScalingGroupReplicaIndex: "0"},
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(world, grove.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}}}
			objects := []client.Object{pcs, world, clique}
			if scenario != "create-child" {
				group.Spec.Replicas = 3
				group.Spec.Policy = &api.EngineGroupScalingPolicy{MinReplicas: ptr.To(int32(2)), MaxReplicas: ptr.To(int32(3))}
				group.Generation = 4
				if scenario == "engine-serving" || scenario == "stale-health" {
					generation := group.Generation
					if scenario == "stale-health" {
						generation--
					}
					group.Status.Conditions = []metav1.Condition{
						{Type: "Available", Status: metav1.ConditionTrue, ObservedGeneration: generation},
						{Type: "TopologyKnown", Status: metav1.ConditionTrue, ObservedGeneration: generation},
					}
				}
				objects = append(objects, group)
			}
			kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(objects...).Build()
			reconciler := &groveWorkloadsReconciler{engineGroups: &enginegroupcontroller.GroveChildrenReconciler{Client: kube}}
			result := ReconcileResult{State: api.DGDStateSuccessful}

			t.Log("seed the child or observe its live serving status without writing DGD status")
			require.NoError(t, reconciler.reconcileEngineGroups(t.Context(), dgd, pcs, &result))
			stored := &api.DynamoGraphDeploymentEngineGroup{}
			require.NoError(t, kube.Get(t.Context(), client.ObjectKeyFromObject(group), stored))
			assert.True(t, metav1.IsControlledBy(stored, dgd))
			assert.Equal(t, profile.Geometry.Fingerprint, stored.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile])
			assert.Equal(t, vllmruntime.InitialLaunchBindings, stored.Annotations[vllmruntime.LaunchBindingsAnnotation])
			assert.Equal(t, before, dgd, "the outer controller remains the only DGD status writer")
			assert.Equal(t, int32(1), result.ComponentStatus[component.ComponentName].Replicas)
			if scenario == "create-child" {
				assert.Equal(t, int32(2), stored.Spec.Replicas)
				assert.Empty(t, stored.Finalizers, "only the Engine Group controller owns effects")
			} else {
				assert.Equal(t, int32(3), stored.Spec.Replicas)
				assert.Equal(t, int32(3), *stored.Spec.Policy.MaxReplicas)
			}
			if scenario == "engine-serving" {
				assert.Equal(t, api.DGDStateSuccessful, result.State)
				assert.Equal(t, int32(1), *result.ComponentStatus[component.ComponentName].ReadyReplicas)
			} else {
				assert.Equal(t, api.DGDStatePending, result.State)
				assert.Equal(t, int32(0), *result.ComponentStatus[component.ComponentName].ReadyReplicas)
			}
		})
	}
}
