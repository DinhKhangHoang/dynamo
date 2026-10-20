/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type vllmLaunchBindings struct {
	Version   int                                         `json:"version"`
	CliqueUID types.UID                                   `json:"cliqueUID"`
	Slots     map[domain.CapacitySlotID]vllmLaunchBinding `json:"slots"`
}

type vllmLaunchBinding struct {
	PodUID           domain.PodUID `json:"podUID"`
	AssignmentDigest string        `json:"assignmentDigest"`
}

type vllmPodLaunch struct {
	pod        *corev1.Pod
	assignment vllmruntime.LaunchAssignment
}

func (a *VLLMGroveCapacity) launchBindings() (vllmLaunchBindings, error) {
	// A missing initialized record is lost authority, not permission to bind replacement processes.
	literal := a.Group.Annotations[vllmruntime.LaunchBindingsAnnotation]
	var bindings vllmLaunchBindings
	decoder := json.NewDecoder(strings.NewReader(literal))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&bindings); err != nil || !json.Valid([]byte(literal)) ||
		bindings.Version != 1 || bindings.Slots == nil ||
		(bindings.CliqueUID != "" && bindings.CliqueUID != a.Grove.CliqueUID) ||
		(bindings.CliqueUID == "" && len(bindings.Slots) != 0) {
		return vllmLaunchBindings{}, fmt.Errorf("%w: vLLM launch bindings are missing, invalid, or belong to another physical world", domain.ErrRecoveryRequired)
	}
	for slot, binding := range bindings.Slots {
		if slot == "" || binding.PodUID == "" || binding.AssignmentDigest == "" {
			return vllmLaunchBindings{}, fmt.Errorf("%w: incomplete vLLM slot binding", domain.ErrRecoveryRequired)
		}
	}
	return bindings, nil
}

func (a *VLLMGroveCapacity) bindLaunches(ctx context.Context, launches []vllmPodLaunch) (bool, error) {
	// Bind every exact launch before making any command visible to the processes.
	bindings, err := a.launchBindings()
	if err != nil {
		return false, err
	}
	before, err := json.Marshal(bindings)
	if err != nil {
		return false, err
	}
	bindings.CliqueUID = a.Grove.CliqueUID
	for _, launch := range launches {
		slot := domain.CapacitySlotID(fmt.Sprintf("slot-%d", launch.assignment.Rank))
		binding, err := vllmBindingForAssignment(launch.assignment)
		if err != nil {
			return false, err
		}
		if accepted, present := bindings.Slots[slot]; present && accepted != binding {
			return false, fmt.Errorf("%w: vLLM slot %s already belongs to another Pod or launch", domain.ErrRecoveryRequired, slot)
		}
		bindings.Slots[slot] = binding
	}
	after, err := json.Marshal(bindings)
	if err != nil {
		return false, err
	}
	if bytes.Equal(before, after) {
		return false, nil
	}

	// UID and resource-version preconditions fence replacement worlds and stale reconciles.
	patch, err := json.Marshal(map[string]any{"metadata": map[string]any{
		"uid": a.Group.UID, "resourceVersion": a.Group.ResourceVersion,
		"annotations": map[string]string{vllmruntime.LaunchBindingsAnnotation: string(after)},
	}})
	if err != nil {
		return false, err
	}
	if err := a.Grove.Client.Patch(ctx, a.Group.DeepCopy(), client.RawPatch(types.MergePatchType, patch)); err != nil {
		return false, fmt.Errorf("persist exact vLLM launch bindings: %w", err)
	}
	return true, nil
}

func vllmBindingForAssignment(assignment vllmruntime.LaunchAssignment) (vllmLaunchBinding, error) {
	encoded, err := json.Marshal(assignment)
	if err != nil {
		return vllmLaunchBinding{}, err
	}
	return vllmLaunchBinding{PodUID: domain.PodUID(assignment.PodUID), AssignmentDigest: fmt.Sprintf("sha256:%x", sha256.Sum256(encoded))}, nil
}
