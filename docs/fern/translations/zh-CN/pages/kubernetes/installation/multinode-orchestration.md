---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 多节点编排
---

多节点部署需要 Grove + KAI Scheduler 或 LeaderWorkerSet（LWS）。使用 LWS 时，需要单独配置跨节点工作负载的调度集成。

## 方案一：Grove + KAI Scheduler

Grove 是默认的多节点编排器，推荐与 KAI Scheduler 配合使用。可以由 Dynamo 自动安装这两个组件，适用于开发和测试；也可以分别安装并管理其生命周期，适用于生产环境。

<Tabs>
  <Tab title="托管安装" value="managed">

托管安装适用于开发和测试。Dynamo 通过捆绑的子 chart 管理 Grove 和 KAI Scheduler 的生命周期。运行以下命令安装 Dynamo 及这两个组件：

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.install=true" \
  --set "global.kai-scheduler.install=true" \
  --set-string "kai-scheduler.scheduler.args.default-staleness-grace-period=-1s"
  ```
  </Tab>
  <Tab title="外部安装" value="external">
  外部安装适用于生产环境或已经安装这些组件的集群。分别安装 Grove 和 KAI Scheduler，可以独立管理生命周期，也可以跨命名空间共享组件。

参阅 [Grove 安装指南](https://github.com/NVIDIA/grove/blob/main/docs/installation.md)和 [KAI Scheduler 部署指南](https://github.com/NVIDIA/KAI-Scheduler)。

运行以下命令，让 Dynamo 使用已安装的 Grove 和 KAI Scheduler：

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.enabled=true" \
  --set "global.kai-scheduler.enabled=true"
  ```

> [!NOTE]
> 外部安装的 Grove 和 KAI Scheduler 必须与 Dynamo Platform 版本兼容。下表列出了各组件的最低版本：
>
> | dynamo-platform | kai-scheduler | Grove |
> |-----------------|---------------|-------|
> | 1.0.x | >= v0.13.0 | >= v0.1.0-alpha.6 |
> | 1.1.x | >= v0.13.4 | >= v0.1.0-alpha.8 |
> | 1.5.x | >= v0.17.0 | >= v0.1.0-alpha.13 |
> | 1.6.x | >= v0.17.0 | >= v0.1.0-alpha.15 |

  </Tab>
</Tabs>


## 方案二：LeaderWorkerSet

安装 [LeaderWorkerSet（LWS）](https://lws.sigs.k8s.io/docs/installation/) 作为 Grove 之外的多节点编排方案。Volcano 集成仍受支持。调度集成的配置与 Dynamo 对 LWS 可用性的检测相互独立。

Dynamo 通过 `leaderworkerset.x-k8s.io` API 组自动检测 LWS。已安装的 API 无法表明 LWS 配置了哪种调度器，Dynamo 也不会自动启用 gang scheduling。部署多节点工作负载前，请配置并验证所选调度集成。如需显式禁用 LWS，在 Operator 配置中设置 `orchestrators.lws.enabled: false`。

标准 DynamoComponentDeployment（DCD）路径支持 LWS `v0.7.0` 及以上版本。启用 DisaggregatedSet（DS）时，需要安装提供 `disaggregatedset.x-k8s.io/v1` API 的 LWS 版本；当前 Dynamo Operator 依赖的 LWS 版本为 `v0.10.0`。

### Volcano

安装 [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide)：

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

安装 LWS，并将调度器提供方设置为 Volcano：

```bash
export LWS_VERSION=0.10.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

LWS 使用此调度器提供方期间，需要保留 Volcano。此集成的调度 PodGroup 由 LWS 管理。

## 按角色配置 Pod 模板

**自 Dynamo 1.6.0 起可用。** 默认情况下，多节点组件的 leader 和 worker 使用同一个组件级 `podTemplate`，后端启动参数由 Dynamo 生成。如果不同角色需要不同的镜像、资源、节点放置或命令，请使用 `roles[].podTemplate`。

角色 Pod 模板是完整配置，不是补丁。省略组件级 `podTemplate`，分别为 `leader` 和 `worker` 提供模板。节点数量仍由 `multinode.nodeCount` 决定：准入逻辑默认将 leader 设为一个 Pod，将 worker 设为 `nodeCount - 1` 个 Pod。

以下 vLLM 组件在不同节点池上运行一个 leader 和一个 worker。每个 Pod 使用一个 GPU，共同组成一个 TP-2 引擎：

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: role-specific-multinode
spec:
  backendFramework: vllm
  components:
  - name: decode
    type: decode
    replicas: 1
    multinode:
      nodeCount: 2
    roles:
    - name: leader
      podTemplate:
        spec:
          nodeSelector:
            example.com/engine-role: leader
          containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
            command: [python3, -m, dynamo.vllm]
            args:
            - --model
            - Qwen/Qwen3-8B
            - --tensor-parallel-size
            - "2"
            - --distributed-executor-backend
            - mp
            - --nnodes
            - "2"
            - --node-rank
            - $(DYNAMO_RANK)
            - --master-addr
            - $(DYNAMO_LEADER_ADDRESS)
            resources:
              limits:
                nvidia.com/gpu: "1"
    - name: worker
      podTemplate:
        spec:
          nodeSelector:
            example.com/engine-role: worker
          containers:
          - name: main
            image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
            command: [python3, -m, dynamo.vllm]
            args:
            - --model
            - Qwen/Qwen3-8B
            - --tensor-parallel-size
            - "2"
            - --distributed-executor-backend
            - mp
            - --nnodes
            - "2"
            - --node-rank
            - $(DYNAMO_RANK)
            - --master-addr
            - $(DYNAMO_LEADER_ADDRESS)
            - --headless
            resources:
              limits:
                nvidia.com/gpu: "1"
```

完整的角色模板将角色相关后端命令的配置交给清单。Dynamo 保留清单中编写的 vLLM、SGLang 或 TensorRT-LLM 拓扑参数，不再生成 leader 和 worker 启动命令。Services、标签、卷、可移植拓扑变量、编排器资源、滚动更新、就绪状态、状态同步和扩缩容仍由 Dynamo 管理。使用 vLLM multiprocessing 时，Dynamo 还会添加 `--master-port=29500`，并保留 worker 在该端口等待 leader 的逻辑。使用 TensorRT-LLM 时，清单中的 worker 命令必须在 `2222` 端口启动 `sshd`；Dynamo 会保留对应的就绪探针，并挂载 MPI SSH 密钥。

以下两种模板来源模式互斥：

```text
Valid:   component.podTemplate
Valid:   leader.podTemplate + worker.podTemplate
Invalid: component.podTemplate + any role.podTemplate
Invalid: a podTemplate on only one required role
```

> [!IMPORTANT]
> 包含 Planner 且使用角色 Pod 模板的 DGD，需要 Planner runtime 1.6.0 或更新版本。如果镜像标签无法标识该版本，请设置 Planner 的 `runtimeVersionOverride`。
> 角色 Pod 模板目前不能与 GPU Memory Service（GMS）或故障转移组合使用。

有关校验和生命周期的详细说明，请参阅 [roles API 参考](../../../../../pages/reference/kubernetes-api/dynamo-component-deployment.mdx#shared-component-spec)。

## 可移植拓扑环境变量

**自 Dynamo 1.6.0 起可用。** Operator 会向新建多节点 DynamoGraphDeployment（DGD）组件的 `main` 容器注入两个与编排器无关的环境变量：

| 变量 | 含义 | Grove 来源 | LWS 来源 |
|---|---|---|---|
| `DYNAMO_RANK` | 引擎节点从零开始的 rank。leader 为 `0`，worker 为 `1` 到 `multinode.nodeCount - 1`。 | `GROVE_PCSG_POD_INDEX` | `LWS_WORKER_INDEX` |
| `DYNAMO_LEADER_ADDRESS` | 引擎 leader 的 DNS 主机名，不含端口。 | PodCliqueScalingGroup leader 的 DNS 名称 | `LWS_LEADER_ADDRESS` |

对于跨 Pod 的 GPU Memory Service（GMS）部署，扩缩容组中还包含权重服务器 clique，因此 Operator 根据引擎角色设置 `DYNAMO_RANK`，而不是使用整个 PCSG 的 Pod 索引。

这些变量由 Operator 管理，组件 `podTemplate` 中的同名值会被替换。自定义命令需要组件拓扑信息时，请使用这些别名，而不是编排器专用的 `GROVE_*` 或 `LWS_*` 变量。

Kubernetes 会展开容器 `command` 和 `args` 中的 `$(NAME)` 引用。容器内的 shell 进程通过 `$NAME` 或 `${NAME}` 读取同一环境变量。例如，可以通过独立参数向引擎命令传入拓扑信息：

```yaml
args:
- --master-addr
- $(DYNAMO_LEADER_ADDRESS)
- --master-port
- "29500"
- --node-rank
- $(DYNAMO_RANK)
```

> [!NOTE]
> 不要向自动配置的标准 vLLM 或 SGLang 组件添加这些参数。在 vLLM multiprocessing 模式下，Operator 将这些别名用于 `--master-addr` 和 `--node-rank`；在 SGLang 中用于 `--dist-init-addr` 和 `--node-rank`。自定义启动命令需要相同拓扑信息时，可以引用这些别名。

此行为由不可变的 Operator 来源版本控制。仅升级 Operator 不会向已有 DGD 添加这些变量，也不会修改其命令行，因此不会触发工作负载滚动更新。如需使用别名，请通过 Dynamo Operator 1.6.0 或更新版本创建 DGD。如果 Grove 不由 Dynamo Platform chart 管理，请先安装 Grove `v0.1.0-alpha.15` 或更新版本。
