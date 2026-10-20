<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Engine Group integration

This package implements the Go membership adapter for vLLM's correlated external
Elastic EP API in [vLLM #62082](https://github.com/vllm-project/vllm/pull/62082).
It does not launch engines, manage discovery registration, or authorize Pod
release. No vLLM Engine Group deployment path is enabled yet.

## Implemented boundary

The adapter calls the native API of each existing rank, including retirees,
using the same operation ID and absolute DP target. Native status is authoritative
for the resize phase, effective topology, process identity, and operation epoch.
Dynamo durably records the exact target and its correlation with those native
identities; it does not implement a second engine transaction state machine.

Supported plans are contiguous append, high-rank-suffix retirement, and re-append
of a fully retired suffix. The profile requires one native DP member per
allocation. Packed allocations, survivor masking, arbitrary selected retirement,
and native identity remapping are not supported by this adapter.

## Ownership

| Owner | Responsibility |
|---|---|
| Engine Group controller | Persist targets, order operations, correlate identities, verify serving, and authorize release. |
| Workload manager | Allocate and place capacity, expose stable slots and Pod UIDs, and perform authorized release. |
| Engine entrypoint | Resolve one allocation's launch assignment and start its native engine. |
| vLLM | Execute membership transactions and expose topology, engine lifetime, admission closure, and drain evidence. |
| Existing Dynamo worker/sidecar | Own serving lifecycle, discovery registration, and KV preparation. Apply Engine Group permission as an additional registration condition. |

## Remaining integration

### Launch-only bootstrap

Each allocation uses one template. Its resolved launch assignment supplies stable
group and member identities, rank, rendezvous, initial/join role, and the
operation-bound DP target. All joiners receive the full resize target, not
`rank + 1`. Initial participants form one complete world without a synthetic grow.

The integration slice implements Pod-UID-bound launch assignments through the
`nvidia.com/dynamo-vllm-launch` Pod annotation and a Downward API volume:

```text
Initial world: immutable native declaration + initial size
Growth:       persisted capacity target + exact joining Pod UID
                                |
                      Go launch assignment
                                |
                  Pod annotation / Downward API
                                |
              launch-only Python entrypoint -> exec native vLLM
```

`VLLMGroveCapacity` reuses Grove's member-clique `/scale` and publishes joining
assignments when Grove creates the Pods. Every joiner receives the complete
resize target. Initial formation uses the same template without a synthetic grow.
Assignments are immutable, published with Pod UID and resource-version
preconditions, and replayed without relaunching retained processes. A replacement
does not inherit the old Pod's assignment. Shrink and replacement remain outside
this growth-only capacity adapter.

Creation seeds an empty launch-binding record on the Engine Group. Before any
command is published, capacity reconciliation persists each slot's exact Pod UID
and assignment digest there, then waits for that cached observation. The record
survives Pod deletion; a new occupant or lost binding fails closed instead of
restarting initial formation. It records launch identity, not engine transaction
phases. Existing Group and Pod lifecycle finalizers must protect effects first.

The launcher has no Kubernetes credentials, engine control loop, proxy, or
discovery owner. It replaces itself with native vLLM. A launch receipt makes
allocation readiness observable before the joiner's API is available; it proves
only assignment consumption, never membership, health, verification, or admission.
Container restarts are prohibited. The Pod-bound launch lifetime and vLLM's native
API instance ID remain distinct and require correlation by the runtime resolver.

Do not wait for inference readiness or registration before submitting the resize
that lets a joiner initialize. Conversely, do not register a joiner because its
Pod or launch receipt is ready.

### Admission belongs in the existing worker

The shared worker is the intended owner of registration. An Engine Group adds
permission for an exact member incarnation and topology; it must not create
another worker, serving loop, or discovery owner.

Registration requires all of:

- Current controller permission for this member and topology.
- Authoritative committed membership and the required topology-bound serving proof.
- Engine serving readiness and valid KV state for the current engine lifetime.
- No shutdown or withdrawal in progress.

Every registration path must honor the same permission, including startup,
resume, and health/KV recovery. After an asynchronous registration, recheck its
evidence before acknowledging admission; if it changed, withdraw and retain the
unconverged target. Accepted permission and observed registration are distinct.

Keep target acceptance separate from slow registration and drain effects. Status
reads and target retries must not wait on a lock held for an inference request's
lifetime. Discovery absence alone is not drain completion, and a wrapper-local
request counter cannot prove engine drain once requests bypass that wrapper.

Engine lifetime, worker/sidecar lifetime, discovery identity, and Pod UID are
separate identities. Restarting a sidecar must not relaunch an engine, inherit
permission without revalidation, or be mistaken for a new engine lifetime.

### Sidecar dependencies

[DEP #14657](https://github.com/ai-dynamo/dynamo/issues/14657) separates the engine
from Dynamo. The replacement stack for
[DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897) provides shared-worker
recovery hooks in [#16752](https://github.com/ai-dynamo/dynamo/pull/16752), native
health and lifetime identity in [#16753](https://github.com/ai-dynamo/dynamo/pull/16753),
and KV recovery in [#16754](https://github.com/ai-dynamo/dynamo/pull/16754) and
[#16755](https://github.com/ai-dynamo/dynamo/pull/16755). Engine Group admission
must compose with that lifecycle rather than replace it.

The next Rust slice should extend the shared worker on top of #16752 and be
validated with the complete stack through #16755. Controller permission is
separate from operator pause/resume intent: recovery or resume must not override
an Engine Group withdrawal. Permission/status controls must be reachable before
serving registration and remain available during recovery; waiting for permission
behind the serving gate would deadlock startup. Withholding permission is not an
engine failure and must not trigger engine replacement or KV recovery.

The stack's health and KV checks do not prove Engine Group membership or
topology-bound serving verification. Native identity binding and real admission
integration remain required before enabling the vLLM deployment path.

The inspected #62082 revision exposes correlated Elastic EP through Python and
HTTP, not the native gRPC control service. The sidecar path still needs supported
remote membership/status controls and engine-authoritative serving and drain
evidence. Do not substitute health for membership, depend on in-process
`AsyncLLM` or middleware globals across containers, or advertise sidecar
conformance before those controls exist.

## Validation and delivery

Current tests cover the Go HTTP client, exact all-rank resize delivery, native
status correlation, and restart recovery of the durable membership binding.
They do not validate real vLLM startup, inference, or discovery integration.

The next slice must prove bootstrap without a joiner-readiness cycle, shared-worker
admission at startup and recovery, topology loss during registration, nonblocking
target submission during drain, and independent engine/sidecar restarts. The
bootstrap renderer and Grove capacity/launch collaborator are covered by CPU and
API-server tests. The DGD pathway now resolves the immutable native declaration,
renders one PCSG with one member clique, seeds the child once, and rolls up
engine-authoritative readiness without resetting its live scale target.

Shared graph assembly and DGD child reconciliation select a backend without
interpreting its engine geometry. The vLLM renderer owns native profile validation,
launch preparation and allocation probes; it reuses the standard Grove Pod
construction for metadata, secrets, resources and placement. Runtime composition,
not the capacity adapter, assembles native membership clients.

`ResolveVLLMGroveCapacity` verifies the complete DGD/PCS/PCSG/clique ownership
chain and derives rendezvous from the exact rank-zero Pod. `ResolveVLLMMembership`
composes direct per-Pod clients with the native membership adapter and a journal
scoped to the physical clique UID. Resolution uses the production Pod-cache
projection and does not contact vLLM: an API-unready joiner must not block launch
delivery or membership preparation. Pod-bound launch lifetime remains separate
from the native API instance ID pinned by membership observation.

These pieces are not registered as a complete production RuntimeProvider.
The admission webhook still rejects DGD opt-in and the default runtime remains
unavailable. Worker traffic admission/drain, topology-bound serving verification,
and startup composition that delivers initial launches before native observation
remain required before deployment can be enabled. No synthetic admission or
health-only verification is substituted for those contracts.
The SGLang PoC's logical-only admission is not a replacement for this contract.

This slice changes no API fields or generated schema. Including `engineGroup` in
`UsesPCSG` selects its workload shape only; the existing hub-only creation block
continues to use its unchanged conversion/preservation rules.
