/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

// groveEngineGroupRenderer owns backend selection and launch layout, not PCS reconciliation.
type groveEngineGroupRenderer struct {
	operatorConfig   *config.OperatorConfiguration
	runtimeConfig    *controller_common.RuntimeConfig
	secretsRetriever SecretsRetriever
}

// Render selects one implemented backend and returns its complete world layout.
// dgd, component, component.EngineGroup and the renderer's configurations must not be nil.
// Nil restart state and annotations mean no pending restart. Inputs are not mutated.
func (r groveEngineGroupRenderer) Render(
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Unsupported engines must not fall back to ordinary, static workload rendering.
	backend, err := getBackendFrameworkFromComponent(component, dgd)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	switch backend {
	case BackendFrameworkVLLM:
		return r.renderVLLM(dgd, component, queueName, topologyDomains, restart, restartAnnotations)
	default:
		return nil, grove.PodCliqueScalingGroupConfig{}, fmt.Errorf("Engine Group Grove rendering is not implemented for backend %q", backend)
	}
}
