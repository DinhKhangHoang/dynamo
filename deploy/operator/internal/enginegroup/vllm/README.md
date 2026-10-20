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

### Bootstrap is a launch concern

Each allocation uses one template. Its resolved launch assignment supplies stable
group and member identities, rank, rendezvous, initial/join role, and the
operation-bound DP target. All joiners receive the full resize target, not
`rank + 1`. Initial participants form one complete world without a synthetic grow.

Assignment acceptance must be recoverable and bound to the exact Pod/process
incarnation. It must remain observable independently of engine initialization.
Do not wait for inference readiness, registration, or a callback blocked behind
engine construction before submitting the resize that lets a joiner initialize.
Bootstrap delivery is not implemented here.

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
target submission during drain, and independent engine/sidecar restarts. The DGD
profile, Grove rendering, traffic adapter, and serving verifier remain disabled.
The SGLang PoC's logical-only admission is not a replacement for this contract.
