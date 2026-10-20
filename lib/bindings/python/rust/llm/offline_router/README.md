<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Host-controlled routing

`dynamo.llm.OfflineRouter` exposes Dynamo's existing `LocalScheduler`,
`DefaultWorkerSelector`, `SessionAffinity` and `RadixTree` to an offline host.
It needs no Endpoint, serving process, model, or simulator package. It is part
of the existing `ai-dynamo-runtime` wheel; `ais-forward-pass` is not required.
The host owns workload execution, engine state, simulation time, and reports.
This API only owns routing, native admission, cache observations and affinity.

The configuration is a JSON object with `schema_version: 1`, `role`
(`aggregated`, `prefill`, `decode`), `num_workers`, `dp_size`, `block_size`, and
`num_gpu_blocks`. Optional fields are `max_num_batched_tokens`, `router_config`,
`affinity: {mode: session|sibling_group, ttl_seconds: 3600}`,
`capture_decisions: false`, and `selector_seed`.

Construct and call the handle on one thread, synchronously outside a Tokio runtime.
Every successful command
returns `{result, pending_count, next_wakeup_ms}`. Time is in virtual
milliseconds. Native timers never follow host CPU time. A failed command
poisons the handle, and the host must fail the run rather than resume partial
state. `snapshot` adds routing `evidence` once, avoiding quadratic copies of
captured decisions during execution. Evidence is the statistics object for this
handle's role, including `native_policy`, `dynamo_version`, and native per-worker
`loads` (`active_blocks`, `active_prefill_tokens`). Prefill routing does not charge
decode occupancy; decode routing counts independent allocations and ignores prefill
load and prefix-cache credit.

Commands:

- `status`, `snapshot`: no other fields; result is null.
- `place`: `now_ms`, `request`, `input_length`, optional `hashes`,
  `max_output_tokens_override`, `session_id`. Result is
  `{placement: null|Placement, released: [Placement]}`; null means queued.
- `observe`: `now_ms`, `events: [{stage, worker_id, dp_rank, events: [KvEvent]}]`.
- `advance_clock`, `topology_settled`: `now_ms`.
- `dispatch_committed`, `dispatch_aborted`, `request_terminal`,
  `prefill_completed`: `now_ms`, `request_id`.
- `cancel_pending`: `request_id`; result is boolean.
- `worker_ready`, `worker_draining`, `worker_removed`: `now_ms`,
  `worker: {worker_id, scheduler_ids}`. Worker IDs begin at zero, and
  `scheduler_id = worker_id * dp_size + dp_rank`.

Clock, observation, terminal, prefill and topology operations return arrays of
released placements. Dispatch commit/abort return null. Commit only after the
engine accepts ownership; abort releases tentative bindings and native bookings.
Poll `advance_clock` at the returned wakeup before advancing the host clock.

A request carries `uuid`, authored `tokens` (or complete block hashes),
`max_output_tokens`, optional exact `output_token_ids`, priorities,
`policy_class`, and optional `replay_context`. Compact hashes contain
`local_block_hashes` and `sequence_hashes`, one of each per complete prompt block.
Unrelated correlation fields are ignored. Length-only synthetic execution tokens
are rejected as cache identities. Rank pins and simulator-owned prefill estimators
are not supported.

`replay_context.agentic` may provide `play_id`, `conversation_id`, and versioned
`lineage` (`schema: aisimulate.agentic.conversation-lineage.v1`,
`root_conversation_id`, optional `parent_conversation_id`). The session mode uses
conversation identity within a play; sibling-group mode requires unambiguous
lineage and groups children of the same parent. Session identity without agentic
metadata comes from `session_id` or `replay_context.session_id`.

A placement contains `request_id`, `scheduler_id`, `reported_overlap_tokens`,
optional `cache_sample: {overlap_blocks, best_available_overlap_blocks, isl_blocks}`
and optional `placement_replica_id`.

A KV event contains `event_id`, `dp_rank`, optional `tier` (device only), and
`data: {stored: {parent_hash, start_position?, session_id?, blocks:
[{block_hash, tokens_hash, token_ids?}]}}` or
`data: {removed: {block_hashes}}`. The API preserves parent links and forwards
session attribution; it does not invent cache residency. Decode placement remains
load-only and does not index decode events.

## Source provenance

The native adapter, affinity ownership and paused-clock host are moved and
modified from this repository's working implementation at
[35ea1e7f689198333b1af0a2db8f94f545817792](https://github.com/ai-dynamo/dynamo/tree/35ea1e7f689198333b1af0a2db8f94f545817792/lib/mocker/src/replay/offline/extensions/kv_router/native),
originally under `lib/mocker/src/replay/offline/extensions/kv_router/native/`.
The upstream file headers and Apache-2.0 licensing are retained. The transport
projects only routing fields; it has no AISim Rust dependency or execution loop.

The optional conversation metadata and neutral event field projection correspond
to the Apache-2.0 AISimulate wire schemas at
[8713218112047d330da2d4952f9bec0092a573df](https://github.com/ai-dynamo/aisimulate/tree/8713218112047d330da2d4952f9bec0092a573df/crates/core/src/replay/protocol.rs)
and `crates/core/src/engine/protocol.rs` (NVIDIA Corporation & Affiliates).
These are modified routing-only projections, not copied simulation code.
