/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// ResolveVLLMComponentEngineGroup resolves the immutable native declaration, not the bootstrap Pod.
// component and component.EngineGroup must not be nil. parentEnv may be nil.
// Effective environment uses the same component-over-parent precedence as rendering. No input is mutated.
func ResolveVLLMComponentEngineGroup(component *api.DynamoComponentDeploymentSharedSpec, parentEnv []corev1.EnvVar) (VLLMProfileGeometrySource, ResolvedVLLMElasticEPProfile, error) {
	// Resolve inherited environment before validating visibility, ports and native geometry.
	component = component.DeepCopy()
	if main := GetMainContainer(component); main != nil {
		main.Env = MergeEnvs(parentEnv, main.Env)
	}
	// The first integration owns one world with one dedicated, scalar-GPU main container per allocation.
	if !IsWorkerComponent(string(component.ComponentType)) || ptr.Deref(component.Replicas, 1) != 1 ||
		component.EngineGroup.InitialSize < 2 || component.PodTemplate == nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups require one worker world, an explicit podTemplate and initialSize greater than one")
	}
	if component.Multinode != nil || len(component.Roles) != 0 || component.ScalingAdapter != nil ||
		component.Experimental != nil || component.ProviderOverride != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups do not support multinode, roles, scalingAdapter, experimental lifecycle features or provider overrides")
	}
	main := GetMainContainer(component)
	if main == nil || main.Image == "" || main.ReadinessProbe != nil || main.StartupProbe != nil || main.LivenessProbe != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM Engine Groups require an explicit main image without caller-owned probes")
	}
	gpus, err := resolveVLLMComponentGPUs(&component.PodTemplate.Spec, main)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	source := VLLMProfileGeometrySource{
		Command: main.Command, Args: main.Args, Env: main.Env, EnvFrom: main.EnvFrom,
		InitialReplicas: component.EngineGroup.InitialSize, MainContainerGPUs: gpus, DedicatedMainGPUAllocation: true,
	}

	// Exclude allocation ordinals, rendezvous and startup DP from the workload fingerprint.
	command, err := resolveVLLMLaunchCommand(source)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}
	spec := component.PodTemplate.Spec.DeepCopy()
	for i := range spec.Containers {
		if spec.Containers[i].Name == consts.MainContainerName {
			spec.Containers[i].Command, spec.Containers[i].Args = command, nil
		}
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("encode vLLM workload declaration: %w", err)
	}
	source.WorkloadRevisionDigest = fmt.Sprintf("pod-spec:%x", sha256.Sum256(encoded))
	profile, err := ResolveVLLMLaunchProfile(source)
	if err != nil {
		return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, err
	}

	// Policy bounds narrow the growth-only integration, never the engine's reserved maximum.
	if policy := component.EngineGroup.Policy; policy != nil {
		if policy.MinSize != nil && *policy.MinSize != profile.InitialReplicas {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM growth-only policy.minSize must equal initialSize")
		}
		if policy.MaxSize != nil && (*policy.MaxSize < profile.InitialReplicas || *policy.MaxSize > profile.MaximumReplicas) {
			return VLLMProfileGeometrySource{}, ResolvedVLLMElasticEPProfile{}, fmt.Errorf("vLLM policy.maxSize must be between initialSize and the reserved DP maximum")
		}
	}
	return source, profile, nil
}

func resolveVLLMComponentGPUs(spec *corev1.PodSpec, main *corev1.Container) (int64, error) {
	// A container restart, shared host port or opaque GPU claim needs a different launch contract.
	if (spec.RestartPolicy != "" && spec.RestartPolicy != corev1.RestartPolicyNever) || spec.HostNetwork ||
		len(spec.ResourceClaims) != 0 || len(main.Resources.Claims) != 0 {
		return 0, fmt.Errorf("vLLM Engine Groups require restartPolicy Never, Pod networking and scalar GPU limits")
	}
	for _, variable := range main.Env {
		if variable.Name == "CUDA_VISIBLE_DEVICES" || variable.Name == "NVIDIA_VISIBLE_DEVICES" {
			return 0, fmt.Errorf("vLLM GPU visibility must come from the Kubernetes allocation")
		}
	}
	containers := append(append([]corev1.Container(nil), spec.InitContainers...), spec.Containers...)
	for _, container := range containers {
		if container.Name == consts.MainContainerName {
			continue
		}
		limit := container.Resources.Limits[consts.KubeResourceGPUNvidia]
		request := container.Resources.Requests[consts.KubeResourceGPUNvidia]
		if !limit.IsZero() || !request.IsZero() || len(container.Resources.Claims) != 0 {
			return 0, fmt.Errorf("vLLM GPUs must belong exclusively to the main container")
		}
	}

	// The complete TP subgroup must fit the exact, indivisible Pod allocation.
	limit := main.Resources.Limits[consts.KubeResourceGPUNvidia]
	gpus, exact := limit.AsInt64()
	request, requested := main.Resources.Requests[consts.KubeResourceGPUNvidia]
	if !exact || gpus < 1 || (requested && request.Cmp(limit) != 0) {
		return 0, fmt.Errorf("vLLM main requires a positive whole-GPU limit and an equal GPU request when present")
	}
	return gpus, nil
}
