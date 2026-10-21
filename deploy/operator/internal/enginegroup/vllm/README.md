<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM Engine Group integration

This package implements the Go membership adapter for vLLM's correlated external
Elastic EP API in [vLLM #62082](https://github.com/vllm-project/vllm/pull/62082).
It also observes topology-bound serving verification and real discovery registration
through the existing Dynamo Python worker. Grove-backed DGD creation and growth are
enabled for that narrow profile. Physical release and failure recovery remain gated.

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

### Existing Python-worker transport

The Python worker forwards external ownership through
`/engine/control/scale_elastic_ep` and `/engine/control/is_scaling_elastic_ep`.
`NewWorkerClient` uses these routes with the same native request/status codec as
`NewClient`; vLLM still owns operation serialization, replay, and topology
certainty. Dynamo requires the operation and expected-instance IDs and does not
turn a lost response into rejection or take the legacy Ray restart path.
The existing Ray controls retain their behavior.

This transport requires #62082. A managed component declares `python3 -m
dynamo.vllm`; its launch-only bootstrap supplies rank and rendezvous without
replacing the existing worker lifecycle. Native-only launches are still usable by
the low-level adapter but cannot opt into this complete deployment pathway.

The worker starts its inference listener without discovery publication. Initial
formation and every committed growth must pass a one-token inference probe at
every rank. The worker runs this check and registers independently of the
controller. Proof is bound to the native API instance and committed operation/epoch;
the controller correlates observed evidence with its topology generation and Pod UID.
Preparation can keep the old topology serving; the native request gate, not the
duration of a resize call, determines whether inference is paused. Native evidence
is checked again after registration to fence an in-flight topology change.
Registration failure is inconclusive and retryable, not permission
to restart the engine.

## Ownership

| Owner | Responsibility |
|---|---|
| Engine Group controller | Persist targets, order operations, correlate observed serving evidence, and authorize release. |
| Workload manager | Allocate and place capacity, expose stable slots and Pod UIDs, and perform authorized release. |
| Engine entrypoint | Resolve one allocation's launch assignment and start its native engine. |
| vLLM | Execute membership transactions and expose topology, engine lifetime, admission closure, and drain evidence. |
| Existing Dynamo worker/sidecar | Follow native serving readiness, verify serving progress, and manage registration and KV prerequisites. Honor outstanding withdrawal intent; no controller serving grant. |

## Bootstrap and lifecycle

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
              launch-only Python entrypoint -> exec Dynamo vLLM worker
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
discovery owner. It replaces itself with the assigned native server or Dynamo worker. A launch receipt makes
allocation readiness observable before the joiner's API is available; it proves
only assignment consumption, never membership, health, verification, or admission.
Container restarts are prohibited. The Pod-bound launch lifetime and vLLM's native
API instance ID remain distinct and require correlation by the runtime resolver.

Do not wait for inference readiness or registration before submitting the resize
that lets a joiner initialize. Conversely, do not register a joiner because its
Pod or launch receipt is ready.

### Serving readiness belongs in the existing worker

The existing worker owns registration. No additional controller permission or
controller acknowledgment is required, including after initial formation and growth.

Registration requires all of:

- Authoritative committed membership and the required topology-bound serving proof.
- Engine serving readiness and valid KV state for the current engine lifetime.
- No shutdown or withdrawal in progress.

Every registration path must use the same native evidence, including startup,
resume, and health/KV recovery. After an asynchronous registration, recheck its
evidence; if it changed, withdraw. Registration is an observation, not a controller target.

Keep target acceptance separate from slow registration and drain effects. Status
reads and target retries must not wait on a lock held for an inference request's
lifetime. Discovery absence alone is not drain completion, and a wrapper-local
request counter cannot prove engine drain once requests bypass that wrapper.

Engine lifetime, worker/sidecar lifetime, discovery identity, and Pod UID are
separate identities. Restarting a sidecar must not relaunch an engine, inherit
serving proof without revalidation, or be mistaken for a new engine lifetime.

### Sidecar dependencies

[DEP #14657](https://github.com/ai-dynamo/dynamo/issues/14657) separates the engine
from Dynamo. The replacement stack for
[DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897) provides shared-worker
recovery hooks in [#16752](https://github.com/ai-dynamo/dynamo/pull/16752), native
health and lifetime identity in [#16753](https://github.com/ai-dynamo/dynamo/pull/16753),
and KV recovery in [#16754](https://github.com/ai-dynamo/dynamo/pull/16754) and
[#16755](https://github.com/ai-dynamo/dynamo/pull/16755). Engine Group admission
must compose with that lifecycle rather than replace it.

The separate Rust admission draft [#16951](https://github.com/ai-dynamo/dynamo/pull/16951)
extends the shared worker on top of #16752. The generic unregistered-listener
primitive is shared with this Python integration; sidecar lifecycle code is not
pulled into the Python worker. The positive-permission design in that separate
draft needs alignment with readiness-driven registration; it is not a dependency
of this pathway. Status controls remain reachable before registration and during
recovery. Expected inference unavailability is not authority to restart the engine.

The stack's health and KV checks do not prove Engine Group membership or
topology-bound serving verification. Native identity binding and real admission
integration remain required before enabling the sidecar deployment path.

The inspected #62082 revision exposes correlated Elastic EP through Python and
HTTP, not the native gRPC control service. The sidecar path still needs supported
remote membership/status controls and engine-authoritative serving and drain
evidence. Do not substitute health for membership, depend on in-process
`AsyncLLM` or middleware globals across containers, or advertise sidecar
conformance before those controls exist.

## Validation and delivery

CPU tests cover native protocol correlation, runtime-owned registration, initial
verification without a synthetic resize, multi-rank append, ambiguous registration,
topology changes during registration, pause/resume, and coordinator reconstruction
at durable boundaries. API-server tests cover Grove Scale and Pod-UID-bound launch
delivery. These are not real-engine or GPU end-to-end conformance results.

The DGD pathway resolves the immutable native declaration,
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

`VLLMWorkerRuntimeProvider` composes these pieces when Grove is enabled. Initial
launch delivery runs only after lifecycle finalizers exist, and before engine
observation. The runtime establishes serving proof without a controller request;
the checkpoint records observed proof, not permission. Group-global negative traffic
targets retain accepted withdrawal/drain intent across restarts. Native engines
retain transaction authority.

Only aggregated generation with one DP member per allocation is enabled: no
LoRA, RL, pooling, disaggregation, packed allocations, process restart, shrink,
replacement, or group retirement is claimed. In particular, discovery withdrawal
is not reported as drain completion. Durable runtime retirement/drain delivery is
not implemented by this worker transport; its adapter rejects every nonempty
drain target. The growth-only adapters reject unsupported
plans rather than silently weakening their safety contracts.

[`poc.yaml`](poc.yaml) contains one DGD with its frontend and initial two-worker
world. Build its image from this branch with vLLM #62082 first; the example tag
is not a published image. After startup, scale the generated Engine Group (not
the DGD's world count) to three. Capture the Engine Group status, exact Pod UIDs,
native statuses, inference before/after growth, and discovery membership. GPU
validation is still required before calling this path production-ready.

This slice changes no API fields or generated schema. Including `engineGroup` in
`UsesPCSG` selects its workload shape only; the existing hub-only creation block
continues to use its unchanged conversion/preservation rules.
