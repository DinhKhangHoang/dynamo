/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// LaunchAnnotation carries an immutable assignment to the launch-only entrypoint through the Downward API.
	LaunchAnnotation = "nvidia.com/dynamo-vllm-launch"
	// LaunchVersion identifies the private operator/entrypoint wire contract, not an Engine Group API version.
	LaunchVersion = "vllm-launch/v1"
	// LaunchBindingsAnnotation retains first-occupant authority after a Pod disappears.
	LaunchBindingsAnnotation = "nvidia.com/dynamo-vllm-launch-bindings"
	// InitialLaunchBindings is seeded only when a new Engine Group is created. Absence is not initialization.
	InitialLaunchBindings = `{"version":1,"cliqueUID":"","slots":{}}`
)

// LaunchAssignment authorizes one native process launch in one exact Pod lifetime.
// A joining assignment retains the capacity target's correlation before the native API exists.
// Assignment acceptance is not committed membership, serving readiness, or traffic admission.
type LaunchAssignment struct {
	Version                string            `json:"version"`
	GroupName              string            `json:"groupName"`
	GroupUID               string            `json:"groupUID"`
	ProfileFingerprint     string            `json:"profileFingerprint"`
	PodUID                 string            `json:"podUID"`
	Rank                   int32             `json:"rank"`
	DataParallelSize       int32             `json:"dataParallelSize"`
	Mode                   string            `json:"mode"`
	ControlRevision        int64             `json:"controlRevision"`
	TransitionID           string            `json:"transitionID"`
	BaseTopologyGeneration int64             `json:"baseTopologyGeneration"`
	Command                []string          `json:"command"`
	Environment            map[string]string `json:"environment"`
}

// PublishLaunch binds an assignment to the observed Pod using an optimistic, UID-checked metadata write.
// pod must not be nil. A stale cache conflict is retryable; an existing assignment is never replaced.
// This method does not wait for projection, start an engine, or assert that the assignment was consumed.
func PublishLaunch(ctx context.Context, writer client.Writer, pod *corev1.Pod, assignment LaunchAssignment) error {
	// Never deliver another occupant's command or repurpose a terminating allocation.
	if pod.UID == "" || assignment.PodUID != string(pod.UID) || !pod.DeletionTimestamp.IsZero() {
		return fmt.Errorf("vLLM launch assignment does not match the live Pod UID")
	}
	if err := assignment.Validate(); err != nil {
		return err
	}
	if literal, present := pod.Annotations[LaunchAnnotation]; present {
		accepted, err := DecodeLaunchAssignment(literal)
		if err != nil || !reflect.DeepEqual(accepted, assignment) {
			return fmt.Errorf("Pod %s already has a different or invalid vLLM launch assignment; recovery is required", pod.Name)
		}
		return nil
	}

	// The resource version and UID must still name the same observed object at publication.
	if pod.ResourceVersion == "" {
		return fmt.Errorf("vLLM launch publication requires an observed Pod resource version")
	}
	encoded, err := json.Marshal(assignment)
	if err != nil {
		return fmt.Errorf("encode vLLM launch assignment: %w", err)
	}
	patch, err := json.Marshal(map[string]any{"metadata": map[string]any{
		"uid": pod.UID, "resourceVersion": pod.ResourceVersion,
		"annotations": map[string]string{LaunchAnnotation: string(encoded)},
	}})
	if err != nil {
		return fmt.Errorf("encode Pod-bound vLLM launch patch: %w", err)
	}
	if err := writer.Patch(ctx, pod.DeepCopy(), client.RawPatch(types.MergePatchType, patch)); err != nil {
		return fmt.Errorf("publish vLLM launch assignment for Pod %s: %w", pod.Name, err)
	}
	return nil
}

// DecodeLaunchAssignment reads one exact versioned launch document, rejecting unknown fields and invalid execution.
func DecodeLaunchAssignment(literal string) (LaunchAssignment, error) {
	// Missing or malformed metadata cannot be interpreted as permission to initialize a replacement.
	var assignment LaunchAssignment
	decoder := json.NewDecoder(strings.NewReader(literal))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&assignment); err != nil {
		return LaunchAssignment{}, fmt.Errorf("decode vLLM launch assignment: %w", err)
	}
	if !json.Valid([]byte(literal)) {
		return LaunchAssignment{}, fmt.Errorf("vLLM launch assignment is not one JSON document")
	}
	return assignment, assignment.Validate()
}

// Validate checks wire-level identity and operation correlation; the profile renderer owns native argv validation.
func (a LaunchAssignment) Validate() error {
	// Initial formation is observed as one world, never represented as a synthetic resize.
	if a.Version != LaunchVersion || a.GroupName == "" || a.GroupUID == "" || a.ProfileFingerprint == "" ||
		a.PodUID == "" || a.Rank < 0 || a.DataParallelSize <= 1 || a.Rank >= a.DataParallelSize ||
		len(a.Command) == 0 || a.Command[0] == "" {
		return fmt.Errorf("vLLM launch assignment requires its version, exact identities, valid DP rank/size, and native command")
	}
	switch a.Mode {
	case "initial":
		if a.ControlRevision != 0 || a.TransitionID != "" || a.BaseTopologyGeneration != 0 {
			return fmt.Errorf("initial vLLM launch must not carry resize correlation")
		}
	case "join":
		if a.ControlRevision <= 0 || a.TransitionID == "" || a.BaseTopologyGeneration <= 0 {
			return fmt.Errorf("joining vLLM launch requires the capacity revision, transition, and base topology")
		}
	default:
		return fmt.Errorf("unsupported vLLM launch mode %q", a.Mode)
	}
	return a.validateExecution()
}

func (a LaunchAssignment) validateExecution() error {
	// Keep the launcher wire contract restricted to native execution and explicit, mode-bound overrides.
	executable := filepath.Base(a.Command[0])
	native := executable == "vllm" && len(a.Command) >= 2 && a.Command[1] == "serve"
	native = native || ((executable == "python" || executable == "python3") && len(a.Command) >= 3 &&
		a.Command[1] == "-m" && (a.Command[2] == "vllm.entrypoints.openai.api_server" || a.Command[2] == "dynamo.vllm"))
	if !native {
		return fmt.Errorf("vLLM launch requires the native server or Dynamo worker entrypoint")
	}
	for _, token := range a.Command {
		if token == "" || strings.ContainsRune(token, '\x00') {
			return fmt.Errorf("vLLM launch command contains an invalid argument")
		}
	}
	joining := "0"
	if a.Mode == "join" {
		joining = "1"
	}
	if len(a.Environment) != 3 || a.Environment["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] != joining ||
		a.Environment["VLLM_ENABLE_V1_MULTIPROCESSING"] != "1" || a.Environment["VLLM_USE_RUST_FRONTEND"] != "0" {
		return fmt.Errorf("vLLM launch environment does not match its native launch mode")
	}
	return nil
}
