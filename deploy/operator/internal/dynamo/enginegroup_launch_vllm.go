/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	vllmruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/vllm"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
)

const (
	VLLMElasticEPBootstrapModule = "dynamo.vllm.elastic_ep_bootstrap"
	vllmLaunchPythonExecutable   = "python3"
	vllmLaunchVolume             = "dynamo-vllm-launch"
	vllmLaunchStateVolume        = "dynamo-vllm-launch-state"
	vllmLaunchDirectory          = "/var/run/dynamo/vllm-launch"
	vllmLaunchStateDirectory     = "/var/run/dynamo/vllm-launch-state"
)

// ResolveVLLMLaunchProfile validates native execution as well as declarative geometry before allocating capacity.
// Native servers and the existing Dynamo Python worker retain their own entrypoints.
func ResolveVLLMLaunchProfile(source VLLMProfileGeometrySource) (ResolvedVLLMElasticEPProfile, error) {
	// Native launch constraints are known before any Pod-specific assignment exists.
	profile, err := ResolveVLLMElasticEPProfile(source)
	if err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if _, err := resolveVLLMLaunchCommand(source); err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if _, _, err := ResolveVLLMControlPorts(source); err != nil {
		return ResolvedVLLMElasticEPProfile{}, err
	}
	if IsVLLMWorkerLaunch(source) {
		values, flags, err := collectVLLMProfileOptions(source.Command, source.Args)
		if err != nil {
			return ResolvedVLLMElasticEPProfile{}, err
		}
		for _, option := range []string{"--enable-lora", "--embedding-worker", "--classify-worker", "--realtime", "--enable-rl", "--gms-shadow-mode", "--route-to-encoder"} {
			if flags[option] {
				return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("managed vLLM worker admission does not support %s", option)
			}
		}
		if (values["--disaggregation-mode"] != "" && values["--disaggregation-mode"] != "agg") || values["--kv-transfer-config"] != "" {
			return ResolvedVLLMElasticEPProfile{}, fmt.Errorf("managed vLLM worker admission requires aggregated generation without a KV-transfer lifecycle")
		}
	}
	return profile, nil
}

// ResolveVLLMControlPorts derives the native HTTP and DP rendezvous ports from a literal declaration.
// This transport profile requires Pod-reachable HTTP, not a Unix socket or native TLS listener.
func ResolveVLLMControlPorts(source VLLMProfileGeometrySource) (int32, int32, error) {
	// Use the same option parser as launch validation, including duplicates and abbreviated-option rejection.
	values, _, err := collectVLLMProfileOptions(source.Command, source.Args)
	if err != nil {
		return 0, 0, err
	}
	if host := values["--host"]; (host != "" && host != "0.0.0.0" && host != "::") ||
		values["--uds"] != "" || values["--ssl-keyfile"] != "" || values["--ssl-certfile"] != "" {
		return 0, 0, fmt.Errorf("vLLM Engine Group control requires an HTTP listener on all Pod interfaces")
	}
	ports := []int32{8000, 13445}
	if IsVLLMWorkerLaunch(source) {
		if values["--port"] != "" {
			return 0, 0, fmt.Errorf("Dynamo worker control uses DYN_SYSTEM_PORT, not vLLM --port")
		}
		ports[0] = consts.DynamoSystemPort
		for _, variable := range source.Env {
			if variable.Name != "DYN_SYSTEM_PORT" {
				continue
			}
			port, err := strconv.ParseInt(variable.Value, 10, 32)
			if variable.ValueFrom != nil || err != nil || port < 1 || port > 32767 {
				return 0, 0, fmt.Errorf("Dynamo worker requires a literal positive DYN_SYSTEM_PORT")
			}
			ports[0] = int32(port)
		}
	}
	for i, name := range []string{"--port", "--data-parallel-rpc-port"} {
		if literal, present := values[name]; present {
			port, err := strconv.ParseInt(literal, 10, 32)
			if err != nil || port < 1 || port > 65535 {
				return 0, 0, fmt.Errorf("vLLM %s must be a valid TCP port", name)
			}
			ports[i] = int32(port)
		}
	}
	if ports[0] == ports[1] {
		return 0, 0, fmt.Errorf("vLLM HTTP and DP rendezvous ports must differ")
	}
	return ports[0], ports[1], nil
}

// RenderVLLMLaunchAssignment resolves one Pod's native command without changing caller-owned objects.
// group and pod must not be nil. A nil target means initial formation; otherwise the target must
// explicitly authorize this joining slot. Initial source geometry is retained across later growth.
func RenderVLLMLaunchAssignment(
	group *api.DynamoGraphDeploymentEngineGroup,
	pod *corev1.Pod,
	source VLLMProfileGeometrySource,
	rendezvousAddress string,
	rpcPort int32,
	target *enginegroup.CapacityTarget,
) (vllmruntime.LaunchAssignment, error) {
	// Resolve ownership and immutable geometry before generating any executable command.
	profile, err := ResolveVLLMLaunchProfile(source)
	if err != nil {
		return vllmruntime.LaunchAssignment{}, err
	}
	if group.UID == "" || pod.UID == "" || pod.Namespace != group.Namespace ||
		pod.Labels[consts.KubeLabelDynamoEngineGroup] != group.Name ||
		group.Annotations[consts.KubeAnnotationDynamoEngineGroupProfile] != profile.Geometry.Fingerprint {
		return vllmruntime.LaunchAssignment{}, fmt.Errorf("vLLM launch requires the exact group, Pod, and resolved profile binding")
	}
	rankLiteral := pod.Labels[grovecommon.LabelPodCliquePodIndex]
	rank, err := strconv.ParseInt(rankLiteral, 10, 32)
	if err != nil || rank < 0 || strconv.FormatInt(rank, 10) != rankLiteral {
		return vllmruntime.LaunchAssignment{}, fmt.Errorf("vLLM launch requires a canonical Grove slot index")
	}
	if rendezvousAddress == "" || strings.ContainsAny(rendezvousAddress, " /\t\r\n") || rpcPort < 1 || rpcPort > 65535 {
		return vllmruntime.LaunchAssignment{}, fmt.Errorf("vLLM launch requires a rendezvous host and valid RPC port")
	}
	assignment := vllmruntime.LaunchAssignment{
		Version: vllmruntime.LaunchVersion, GroupName: group.Name, GroupUID: string(group.UID),
		ProfileFingerprint: profile.Geometry.Fingerprint, PodUID: string(pod.UID),
		Rank: int32(rank), DataParallelSize: profile.InitialReplicas, Mode: "initial",
		Environment: map[string]string{
			"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "0",
			vllmMultiprocessingEnvVar:         "1",
			"VLLM_USE_RUST_FRONTEND":          "0",
		},
	}
	if target == nil {
		if assignment.Rank >= profile.InitialReplicas {
			return vllmruntime.LaunchAssignment{}, fmt.Errorf("a non-initial vLLM slot requires an operation-bound joining assignment")
		}
	} else if err := setVLLMJoiningAssignment(&assignment, *target, profile); err != nil {
		return vllmruntime.LaunchAssignment{}, err
	}

	// Preserve native entrypoint/model syntax, replacing only provider-owned DP launch arguments.
	command, err := renderVLLMNativeLaunch(source, assignment, rendezvousAddress, rpcPort)
	if err != nil {
		return vllmruntime.LaunchAssignment{}, err
	}
	assignment.Command = command
	return assignment, assignment.Validate()
}

func setVLLMJoiningAssignment(assignment *vllmruntime.LaunchAssignment, target enginegroup.CapacityTarget, profile ResolvedVLLMElasticEPProfile) error {
	// No count or slot alone authorizes a process: require the persisted capacity bootstrap intent.
	if target.ProfileFingerprint != assignment.ProfileFingerprint || target.ProcessLifecycleOwner != enginegroup.ProcessLifecycleOwnerOrchestrator ||
		len(target.ReleaseFences) != 0 || len(target.Replicas) > int(profile.MaximumReplicas) || len(target.Replicas) <= int(profile.InitialReplicas) {
		return fmt.Errorf("vLLM joining launch requires a growth-only capacity target within the resolved profile")
	}
	for _, replica := range target.Replicas {
		if replica.ReplicaID != enginegroup.ReplicaID(fmt.Sprintf("replica-%d", assignment.Rank)) {
			continue
		}
		bootstrap := replica.Bootstrap
		if replica.Incarnation != nil || bootstrap == nil || replica.SlotID != enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", assignment.Rank)) ||
			bootstrap.Mode != enginegroup.BootstrapModeJoin || len(bootstrap.NativeMembers) != 1 ||
			bootstrap.NativeMembers[0] != enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", assignment.Rank)) ||
			assignment.Rank < profile.InitialReplicas || int(assignment.Rank) >= len(target.Replicas) {
			return fmt.Errorf("vLLM joining launch does not match the authorized slot and native member")
		}
		assignment.Mode = "join"
		assignment.DataParallelSize = int32(len(target.Replicas))
		assignment.ControlRevision = target.ControlRevision
		assignment.TransitionID = target.TransitionID
		assignment.BaseTopologyGeneration = bootstrap.BaseTopologyGeneration
		assignment.Environment["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "1"
		return nil
	}
	return fmt.Errorf("vLLM Pod slot is absent from the joining capacity target")
}

func renderVLLMNativeLaunch(source VLLMProfileGeometrySource, assignment vllmruntime.LaunchAssignment, address string, port int32) ([]string, error) {
	// Every joiner uses the complete target and common rendezvous, including multi-joiner growth.
	command, err := resolveVLLMLaunchCommand(source)
	if err != nil {
		return nil, err
	}
	return append(command,
		"--data-parallel-size", strconv.FormatInt(int64(assignment.DataParallelSize), 10),
		"--data-parallel-rank", strconv.FormatInt(int64(assignment.Rank), 10),
		"--data-parallel-size-local", "1", "--data-parallel-external-lb",
		"--data-parallel-address", address, "--data-parallel-rpc-port", strconv.FormatInt(int64(port), 10),
	), nil
}

func resolveVLLMLaunchCommand(source VLLMProfileGeometrySource) ([]string, error) {
	// Preserve the declared native or Dynamo entrypoint, never a shell supervisor.
	argv := append(append([]string(nil), source.Command...), source.Args...)
	if len(source.Command) == 0 {
		return nil, fmt.Errorf("vLLM Engine Group bootstrap requires an explicit native command")
	}
	prefix := 0
	if filepath.Base(argv[0]) == string(BackendFrameworkVLLM) && len(argv) >= 2 && argv[1] == "serve" {
		prefix = 2
	} else if len(argv) >= 3 && argv[1] == "-m" && (argv[2] == "vllm.entrypoints.openai.api_server" || argv[2] == vllmWorkerModule) {
		prefix = 3
	} else {
		return nil, fmt.Errorf("vLLM Engine Group bootstrap requires a native server or the Dynamo worker module")
	}
	command := append([]string(nil), argv[:prefix]...)
	for index := prefix; index < len(argv); index++ {
		token := argv[index]
		if token == "--" || hasKubernetesArgumentExpansion(token) {
			return nil, fmt.Errorf("vLLM launch arguments must be literal and cannot contain an option terminator")
		}
		name, _, inline := strings.Cut(token, "=")
		name = strings.ReplaceAll(name, "_", "-")
		if strings.HasPrefix(name, "--no-") {
			name = "--" + strings.TrimPrefix(name, "--no-")
		} else if strings.HasPrefix(name, "--disable-") {
			name = "--enable-" + strings.TrimPrefix(name, "--disable-")
		}
		option, matched := matchVLLMProfileOption(name)
		if matched && vllmManagedLaunchOption(option.name) {
			if !option.boolean && !inline {
				index++
			}
			continue
		}
		command = append(command, token)
	}
	return command, nil
}

// IsVLLMWorkerLaunch identifies the existing Python worker, not a native API server or sidecar.
func IsVLLMWorkerLaunch(source VLLMProfileGeometrySource) bool {
	// Command/args may split the module prefix; do not mutate either caller-owned slice.
	argv := append(append([]string(nil), source.Command...), source.Args...)
	return len(argv) >= 3 && isSupportedProfilePythonExecutable(filepath.Base(argv[0])) &&
		argv[1] == "-m" && argv[2] == vllmWorkerModule
}

func vllmManagedLaunchOption(name string) bool {
	switch name {
	case "--data-parallel-size", "--data-parallel-rank", "--data-parallel-size-local",
		"--data-parallel-start-rank", "--data-parallel-external-lb", "--data-parallel-hybrid-lb",
		"--data-parallel-address", "--data-parallel-rpc-port":
		return true
	default:
		return false
	}
}

// RenderVLLMLaunchPodTemplate installs launch-only bootstrap and allocation readiness on a copy.
// template must not be nil. Engine health and serving admission are deliberately not Pod readiness.
// The caller retains the original native command as the immutable profile source for assignments.
func RenderVLLMLaunchPodTemplate(template *corev1.PodTemplateSpec, groupName, fingerprint string) (*corev1.PodTemplateSpec, error) {
	// Never overwrite user mounts, probes, or launch bindings with ambiguous readiness semantics.
	result := template.DeepCopy()
	if groupName == "" || fingerprint == "" || (result.Spec.RestartPolicy != "" && result.Spec.RestartPolicy != corev1.RestartPolicyNever) {
		return nil, fmt.Errorf("vLLM launch template requires its group/profile and restartPolicy Never")
	}
	if _, present := result.Annotations[vllmruntime.LaunchAnnotation]; present {
		return nil, fmt.Errorf("vLLM launch assignments belong to exact Pods, never the shared template")
	}
	for _, volume := range result.Spec.Volumes {
		if volume.Name == vllmLaunchVolume || volume.Name == vllmLaunchStateVolume {
			return nil, fmt.Errorf("vLLM launch volume name %q is reserved", volume.Name)
		}
	}
	mainIndex := -1
	for i := range result.Spec.Containers {
		if result.Spec.Containers[i].Name == consts.MainContainerName {
			mainIndex = i
		}
	}
	if mainIndex < 0 {
		return nil, fmt.Errorf("vLLM launch template requires the main container")
	}
	main := &result.Spec.Containers[mainIndex]
	if main.ReadinessProbe != nil || main.StartupProbe != nil || main.LivenessProbe != nil {
		return nil, fmt.Errorf("vLLM launch template cannot replace caller-owned probes; inference health belongs to the worker admission path")
	}
	for _, variable := range main.Env {
		if strings.HasPrefix(variable.Name, "DYNAMO_VLLM_LAUNCH_") || strings.HasPrefix(variable.Name, "DYNAMO_ENGINE_GROUP_") {
			return nil, fmt.Errorf("vLLM launch environment prefix is reserved")
		}
	}
	for _, mount := range main.VolumeMounts {
		if mount.MountPath == vllmLaunchDirectory || mount.MountPath == vllmLaunchStateDirectory {
			return nil, fmt.Errorf("vLLM launch mount path %q is reserved", mount.MountPath)
		}
	}

	// Project immutable, Pod-specific metadata without granting the entrypoint Kubernetes credentials.
	result.Spec.RestartPolicy = corev1.RestartPolicyNever
	result.Spec.Volumes = append(result.Spec.Volumes,
		corev1.Volume{Name: vllmLaunchVolume, VolumeSource: corev1.VolumeSource{DownwardAPI: &corev1.DownwardAPIVolumeSource{
			Items: []corev1.DownwardAPIVolumeFile{{Path: "assignment", FieldRef: &corev1.ObjectFieldSelector{
				APIVersion: "v1", FieldPath: "metadata.annotations['" + vllmruntime.LaunchAnnotation + "']",
			}}},
		}}},
		corev1.Volume{Name: vllmLaunchStateVolume, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
	)
	main.VolumeMounts = append(main.VolumeMounts,
		corev1.VolumeMount{Name: vllmLaunchVolume, MountPath: vllmLaunchDirectory, ReadOnly: true},
		corev1.VolumeMount{Name: vllmLaunchStateVolume, MountPath: vllmLaunchStateDirectory},
	)
	main.Env = append(main.Env,
		corev1.EnvVar{Name: "DYNAMO_VLLM_LAUNCH_GROUP", Value: groupName},
		corev1.EnvVar{Name: "DYNAMO_VLLM_LAUNCH_PROFILE", Value: fingerprint},
		corev1.EnvVar{Name: "DYNAMO_VLLM_LAUNCH_POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}},
		corev1.EnvVar{Name: "DYNAMO_VLLM_LAUNCH_SLOT", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['" + grovecommon.LabelPodCliquePodIndex + "']"}}},
	)
	main.Command = []string{vllmLaunchPythonExecutable, "-m", VLLMElasticEPBootstrapModule}
	main.Args = nil
	main.ReadinessProbe = &corev1.Probe{
		ProbeHandler:  corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{vllmLaunchPythonExecutable, "-m", VLLMElasticEPBootstrapModule, "--check"}}},
		PeriodSeconds: 2, TimeoutSeconds: 2, FailureThreshold: 3,
	}
	return result, nil
}
