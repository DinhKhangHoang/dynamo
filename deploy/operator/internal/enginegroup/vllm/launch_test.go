/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPublishLaunchIsImmutableAndIncarnationBound(t *testing.T) {
	t.Log("start with an observed Pod and an exact joining assignment")
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "joiner", Namespace: "test", UID: "pod-uid"}}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod).Build()
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	assignment := LaunchAssignment{
		Version: LaunchVersion, GroupName: "world", GroupUID: "group-uid", ProfileFingerprint: "fingerprint",
		PodUID: "pod-uid", Rank: 2, DataParallelSize: 3, Mode: "join", ControlRevision: 4, TransitionID: "grow-three",
		BaseTopologyGeneration: 1, Command: []string{"vllm", "serve", "model"},
		Environment: map[string]string{
			"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_USE_RUST_FRONTEND": "0",
		},
	}

	t.Log("publish once and replay exactly after the annotation is observed")
	require.NoError(t, PublishLaunch(t.Context(), kubeClient, pod, assignment))
	require.Empty(t, pod.Annotations, "publication must not mutate the caller's cached snapshot")
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	version := pod.ResourceVersion
	require.NoError(t, PublishLaunch(t.Context(), kubeClient, pod, assignment))
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	assert.Equal(t, version, pod.ResourceVersion)

	t.Log("refuse a competing operation and a replacement inheriting old metadata")
	competing := assignment
	competing.TransitionID = "another-operation"
	require.ErrorContains(t, PublishLaunch(t.Context(), kubeClient, pod, competing), "different or invalid")
	replacement := pod.DeepCopy()
	replacement.UID = "replacement-uid"
	require.ErrorContains(t, PublishLaunch(t.Context(), kubeClient, replacement, assignment), "Pod UID")
}

func TestPublishLaunchRetriesInformerLagWithoutOverwritingNewerMetadata(t *testing.T) {
	t.Log("retain a stale cache snapshot while another writer updates the Pod")
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "initial", Namespace: "test", UID: "pod-uid"}}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pod).Build()
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	stale := pod.DeepCopy()
	pod.Labels = map[string]string{"other-writer": "preserved"}
	require.NoError(t, kubeClient.Update(t.Context(), pod))
	assignment := LaunchAssignment{
		Version: LaunchVersion, GroupName: "world", GroupUID: "group-uid", ProfileFingerprint: "fingerprint",
		PodUID: "pod-uid", Rank: 0, DataParallelSize: 2, Mode: "initial", Command: []string{"vllm", "serve", "model"},
		Environment: map[string]string{
			"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_USE_RUST_FRONTEND": "0",
		},
	}

	t.Log("a stale publication conflicts, then succeeds using the next cached observation")
	err := PublishLaunch(t.Context(), kubeClient, stale, assignment)
	require.True(t, apierrors.IsConflict(err))
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	require.NoError(t, PublishLaunch(t.Context(), kubeClient, pod, assignment))
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	assert.Equal(t, "preserved", pod.Labels["other-writer"])
	assert.NotEmpty(t, pod.Annotations[LaunchAnnotation])
}
