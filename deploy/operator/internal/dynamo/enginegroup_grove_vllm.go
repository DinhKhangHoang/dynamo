/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func (r groveEngineGroupRenderer) renderVLLM(
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Resolve native geometry once, before common Pod infrastructure can add defaults.
	_, profile, err := ResolveVLLMComponentEngineGroup(component, dgd.Spec.Env)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	component = component.DeepCopy()
	main := GetMainContainer(component)
	main.Env = MergeEnvs(dgd.Spec.Env, main.Env)
	propagateDGDAnnotations(dgd.Annotations, component)
	podTemplate := ensurePodTemplate(component)
	podTemplate.Labels[consts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dgd, component)
	discovery := controller_common.GetDiscoveryBackend(r.operatorConfig.Discovery.Backend, dgd.Annotations)
	if discovery != "" {
		podTemplate.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(discovery)
	}

	// Reuse standard metadata, secrets, resources and placement, without legacy vLLM command rewriting.
	role := ServiceRole{Name: component.ComponentName, Role: RoleMain, Replicas: profile.InitialReplicas}
	clique, err := buildCliqueForRole(cliqueParams{
		r: role, component: component, backendFramework: BackendFrameworkNoop,
		secretsRetriever: r.secretsRetriever, dynamoDeployment: dgd, numberOfNodes: 1,
		operatorConfig: r.operatorConfig, runtimeConfig: r.runtimeConfig, componentName: component.ComponentName,
		usesPCSG: true, discoveryBackend: discovery,
		discoveryContext: NewDiscoveryContext(r.operatorConfig.Discovery.Backend, dgd.Annotations),
		restartState:     restart, existingRestartAnnotations: restartAnnotations,
		validatedQueueName: queueName, groveClusterTopologyDomains: topologyDomains,
	})
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}

	// Native allocation readiness replaces generated worker probes, never caller-owned probes.
	for i := range clique.Spec.PodSpec.Containers {
		if clique.Spec.PodSpec.Containers[i].Name == consts.MainContainerName {
			main := &clique.Spec.PodSpec.Containers[i]
			main.ReadinessProbe, main.LivenessProbe, main.StartupProbe = nil, nil, nil
		}
	}
	clique.Spec.PodSpec.RestartPolicy = corev1.RestartPolicyNever
	groupName := EngineGroupNameForComponent(dgd.Name, component.ComponentName, 0)
	template, err := RenderVLLMLaunchPodTemplate(&corev1.PodTemplateSpec{Spec: clique.Spec.PodSpec}, groupName, profile.Geometry.Fingerprint)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	clique.Spec.PodSpec = template.Spec
	clique.Spec.MinAvailable = ptr.To(profile.InitialReplicas)
	clique.Labels[consts.KubeLabelDynamoEngineGroup] = groupName
	clique.Labels[consts.KubeLabelDynamoScaleRepresentative] = consts.KubeLabelDynamoScaleRepresentativeYes

	// The PCSG counts worlds; only its member clique carries this world's initial capacity.
	world := buildGroveScalingGroupConfig(component.ComponentName, component, dgd.Spec.TopologyConstraint,
		[]ServiceRole{role}, []string{clique.Name}, nil, false)
	return clique, world, nil
}
