// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-controlled routing, without a serving endpoint or simulator dependency.

mod affinity;
mod helpers;
mod router;
mod runtime;
#[cfg(test)]
mod tests;
mod wire;

use affinity::{ReplayAffinityConfig, RoutingEvidence};
use anyhow::{Context, Result, ensure};
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
use helpers::NativeRouterEngineArgs;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use router::KvRouterPlacement;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;
use wire::{Observation, RequestHashes, RoutingRequest, WorkerTopology};

const NATIVE_DYNAMO_VERSION: &str = env!("CARGO_PKG_VERSION");
type ReplayPrefillLoadEstimator = Arc<dyn dynamo_kv_router::PrefillLoadEstimator>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    role: String,
    dp_size: u32,
    num_workers: usize,
    block_size: u32,
    num_gpu_blocks: usize,
    max_num_batched_tokens: Option<usize>,
    router_config: Option<KvRouterConfig>,
    affinity: Option<ReplayAffinityConfig>,
    #[serde(default)]
    capture_decisions: bool,
    selector_seed: Option<u64>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Status,
    Snapshot,
    Place {
        now_ms: f64,
        request: RoutingRequest,
        input_length: usize,
        hashes: Option<RequestHashes>,
        max_output_tokens_override: Option<usize>,
        session_id: Option<String>,
    },
    Observe {
        now_ms: f64,
        events: Vec<Observation>,
    },
    AdvanceClock {
        now_ms: f64,
    },
    DispatchCommitted {
        now_ms: f64,
        request_id: Uuid,
    },
    DispatchAborted {
        now_ms: f64,
        request_id: Uuid,
    },
    RequestTerminal {
        now_ms: f64,
        request_id: Uuid,
    },
    PrefillCompleted {
        now_ms: f64,
        request_id: Uuid,
    },
    CancelPending {
        request_id: Uuid,
    },
    WorkerReady {
        now_ms: f64,
        worker: WorkerTopology,
    },
    WorkerDraining {
        now_ms: f64,
        worker: WorkerTopology,
    },
    WorkerRemoved {
        now_ms: f64,
        worker: WorkerTopology,
    },
    TopologySettled {
        now_ms: f64,
    },
}

/// Routing policy handle. Hosts own requests, cache observations, time and execution.
/// Any command error poisons the handle: partially applied routing cannot resume.
#[pyclass(module = "dynamo._core", unsendable)]
pub(crate) struct OfflineRouter {
    placement: KvRouterPlacement,
    evidence: RoutingEvidence,
    role: &'static str,
    dp_size: u32,
    block_size: u32,
    active: HashSet<Uuid>,
    workers: HashSet<usize>,
    failed: bool,
}

#[pymethods]
impl OfflineRouter {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        Self::from_json(config_json).map_err(|error| PyValueError::new_err(format!("{error:#}")))
    }

    fn call_json(&mut self, command_json: &str) -> PyResult<String> {
        self.execute_json(command_json)
            .map_err(|error| PyValueError::new_err(format!("{error:#}")))
    }
}

impl OfflineRouter {
    fn from_json(config_json: &str) -> Result<Self> {
        let config: Config =
            serde_json::from_str(config_json).context("invalid OfflineRouter configuration")?;
        ensure!(
            config.schema_version == 1,
            "unsupported OfflineRouter schema version"
        );
        let role = match config.role.as_str() {
            "aggregated" => "aggregated",
            "prefill" => "prefill",
            "decode" => "decode",
            _ => anyhow::bail!("role must be aggregated, prefill, or decode"),
        };
        ensure!(
            config.dp_size > 0
                && config.num_workers > 0
                && config.block_size > 0
                && config.num_gpu_blocks > 0,
            "worker, rank, block size and capacity must be positive"
        );
        ensure!(
            config.max_num_batched_tokens != Some(0),
            "batch token capacity must be positive"
        );
        config
            .num_workers
            .checked_mul(config.dp_size as usize)
            .context("worker topology overflow")?;
        validate_replay_router_config(config.router_config.as_ref())?;
        if config.affinity.is_some() {
            validate_affinity_router_config(config.router_config.as_ref())?;
        }
        ensure!(
            config
                .router_config
                .as_ref()
                .is_none_or(|c| c.router_prefill_load_model != RouterPrefillLoadModel::Ais),
            "OfflineRouter does not accept a simulator-owned prefill estimator"
        );
        // Validate caller policy before deriving the serving role's load model.
        // Prefill has no active decode allocation; decode is load-only and must
        // not assume that a prefill worker's KV is already resident locally.
        let mut router_config = config.router_config.unwrap_or_default();
        match role {
            "prefill" => router_config.router_track_active_blocks = false,
            "decode" => {
                router_config.overlap_score_credit = 0.0;
                router_config.router_assume_kv_reuse = false;
                router_config.router_track_prefill_tokens = false;
                router_config.router_prefill_load_model = RouterPrefillLoadModel::None;
            }
            _ => {}
        }
        let evidence = RoutingEvidence::default();
        let placement = KvRouterPlacement::new_with_selector_seed(
            &NativeRouterEngineArgs {
                dp_size: config.dp_size,
                block_size: config.block_size,
                num_gpu_blocks: config.num_gpu_blocks,
                max_num_batched_tokens: config.max_num_batched_tokens,
            },
            Some(router_config),
            None,
            config.num_workers,
            config.selector_seed,
        )?
        .with_affinity(
            config.affinity,
            role,
            config.capture_decisions,
            evidence.clone(),
        )?;
        Ok(Self {
            placement,
            evidence,
            role,
            dp_size: config.dp_size,
            block_size: config.block_size,
            active: HashSet::new(),
            workers: (0..config.num_workers).collect(),
            failed: false,
        })
    }

    fn execute_json(&mut self, command_json: &str) -> Result<String> {
        ensure!(
            !self.failed,
            "OfflineRouter failed previously; create a new handle"
        );
        let result = (|| {
            let command: Command =
                serde_json::from_str(command_json).context("invalid OfflineRouter command")?;
            let capture = matches!(command, Command::Snapshot);
            let result = self.execute(command)?;
            let mut response = json!({"result": result, "pending_count": self.placement.pending_count(),
                "next_wakeup_ms": self.placement.next_wakeup_ms()});
            if capture {
                let snapshot = self.evidence.snapshot()?;
                response["evidence"] = snapshot["roles"].get(self.role).cloned().unwrap_or_else(|| json!({
                    "native_policy": "dynamo.DefaultWorkerSelector", "dynamo_version": NATIVE_DYNAMO_VERSION,
                }));
                response["evidence"]["loads"] = self.placement.loads()?;
            }
            Ok(serde_json::to_string(&response)?)
        })();
        self.failed = result.is_err();
        result
    }

    fn validate_worker(&self, worker: &WorkerTopology) -> Result<()> {
        let base = worker
            .worker_id
            .checked_mul(self.dp_size as usize)
            .context("worker index overflow")?;
        ensure!(
            worker.scheduler_ids.len() == self.dp_size as usize
                && worker
                    .scheduler_ids
                    .iter()
                    .enumerate()
                    .all(|(rank, &id)| base.checked_add(rank) == Some(id)),
            "worker/rank scheduler IDs must be contiguous"
        );
        Ok(())
    }

    fn execute(&mut self, command: Command) -> Result<Value> {
        match command {
            Command::Status | Command::Snapshot => Ok(Value::Null),
            Command::Place {
                now_ms,
                mut request,
                input_length,
                hashes,
                max_output_tokens_override,
                session_id,
            } => {
                let id = request
                    .uuid
                    .context("routing requires a stable request UUID")?;
                ensure!(!self.active.contains(&id), "duplicate routing request {id}");
                ensure!(
                    request.preferred_dp_rank.is_none()
                        && request.preferred_prefill_dp_rank.is_none(),
                    "OfflineRouter does not support authored DP-rank pins"
                );
                request.input_length = input_length;
                if let Some(hashes) = &hashes {
                    let blocks = input_length / self.block_size as usize;
                    ensure!(
                        hashes.local_block_hashes.len() == blocks
                            && hashes.sequence_hashes.len() == blocks,
                        "request hashes must cover exactly the complete prompt blocks"
                    );
                }
                let limit = request
                    .output_token_ids
                    .as_ref()
                    .map_or(request.max_output_tokens, Vec::len);
                request.max_output_tokens =
                    max_output_tokens_override.map_or(limit, |value| value.min(limit));
                let result = self.placement.place(&request, hashes, session_id, now_ms)?;
                self.active.insert(id);
                Ok(result)
            }
            Command::Observe { now_ms, events } => {
                let mut native = Vec::new();
                for event in events {
                    ensure!(
                        event.stage == self.role,
                        "observation stage disagrees with routing role"
                    );
                    event.validate_topology(&self.workers, self.dp_size)?;
                    let events = event.into_events()?;
                    if self.role != "decode" {
                        native.extend(events);
                    }
                }
                Ok(serde_json::to_value(
                    self.placement.observe(native, now_ms)?,
                )?)
            }
            Command::AdvanceClock { now_ms } => {
                Ok(serde_json::to_value(self.placement.advance_clock(now_ms)?)?)
            }
            Command::DispatchCommitted { now_ms, request_id } => {
                ensure!(
                    self.active.contains(&request_id),
                    "cannot commit an unknown routing request"
                );
                self.placement.dispatch_committed(request_id, now_ms)?;
                Ok(Value::Null)
            }
            Command::DispatchAborted { now_ms, request_id } => {
                self.placement.dispatch_aborted(request_id, now_ms)?;
                self.active.remove(&request_id);
                Ok(Value::Null)
            }
            Command::RequestTerminal { now_ms, request_id } => {
                let result = self.placement.request_terminal(request_id, now_ms)?;
                self.active.remove(&request_id);
                Ok(serde_json::to_value(result)?)
            }
            Command::PrefillCompleted { now_ms, request_id } => Ok(serde_json::to_value(
                self.placement.prefill_completed(request_id, now_ms)?,
            )?),
            Command::CancelPending { request_id } => {
                let canceled = self.placement.cancel_pending(request_id);
                if canceled {
                    self.active.remove(&request_id);
                }
                Ok(json!(canceled))
            }
            Command::WorkerReady { now_ms, worker } => {
                self.validate_worker(&worker)?;
                let id = worker.worker_id;
                ensure!(!self.workers.contains(&id), "worker already exists");
                let ready = self.placement.worker_ready(worker, now_ms)?;
                self.workers.insert(id);
                Ok(serde_json::to_value(ready)?)
            }
            Command::WorkerDraining { now_ms, worker } => {
                self.validate_worker(&worker)?;
                ensure!(
                    self.workers.contains(&worker.worker_id),
                    "unknown draining worker"
                );
                Ok(serde_json::to_value(
                    self.placement.worker_draining(worker, now_ms)?,
                )?)
            }
            Command::WorkerRemoved { now_ms, worker } => {
                self.validate_worker(&worker)?;
                let id = worker.worker_id;
                ensure!(self.workers.contains(&id), "unknown removed worker");
                let ready = self.placement.worker_removed(worker, now_ms)?;
                self.workers.remove(&id);
                Ok(serde_json::to_value(ready)?)
            }
            Command::TopologySettled { now_ms } => Ok(serde_json::to_value(
                self.placement.topology_settled(now_ms)?,
            )?),
        }
    }
}

fn validate_affinity_router_config(config: Option<&KvRouterConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    let defaults = KvRouterConfig::default();
    ensure!(
        serde_json::to_value(config)? == serde_json::to_value(&defaults)?
            && config.router_approximate_cache_policy == defaults.router_approximate_cache_policy
            && config.router_prefill_policy.is_none()
            && config.router_decode_policy.is_none()
            && config.policy_model_name.is_none()
            && config.policy_config_cache.get().is_none(),
        "conversation affinity supports only the default native KV router configuration; queue, custom-policy, cache-tier, and prefill-load overrides require ordinary non-affinity replay"
    );
    Ok(())
}

fn validate_replay_router_config(config: Option<&KvRouterConfig>) -> Result<()> {
    let Some(config) = config else {
        return Ok(());
    };
    let defaults = KvRouterConfig::default();
    let configured = serde_json::to_value(config)?;
    let baseline = serde_json::to_value(&defaults)?;
    // These are live-service facilities. Replay supplies synchronous local KV
    // events and a fixed topology, so accepting overrides would silently discard
    // the requested behavior. Native scoring, tracking and queue controls remain
    // available to ordinary routing.
    for field in [
        "use_kv_events",
        "router_replica_sync",
        "router_track_output_blocks",
        "router_ttl_secs",
        "router_event_threads",
        "skip_initial_worker_wait",
        "use_remote_indexer",
        "serve_indexer",
        "enable_session_prefix_index",
        "shared_cache_multiplier",
        "shared_cache_type",
        "router_predicted_ttl_secs",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
        "conditional_disagg_enabled",
        "conditional_disagg_policy",
        "conditional_disagg_eff_isl_threshold",
        "conditional_disagg_eff_isl_ratio_threshold",
        "conditional_disagg_prefill_busy_threshold",
        "conditional_disagg_decode_busy_threshold",
    ] {
        ensure!(
            configured.get(field) == baseline.get(field),
            "native replay does not support overriding router_config.{field}"
        );
    }
    ensure!(
        config.router_approximate_cache_policy == defaults.router_approximate_cache_policy,
        "native replay does not support approximate-cache policies"
    );
    Ok(())
}
