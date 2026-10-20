// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous bindings to native scheduler and affinity primitives. Hosts own
//! request interpretation, indexing, dispatch, diagnostics and clock advancement.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use dynamo_kv_router::config::{KvRouterConfig, RouterPrefillLoadModel};
use dynamo_kv_router::indexer::{MatchDetails, TieredMatchDetails};
use dynamo_kv_router::protocols::{
    BlockHashOptions, LocalBlockHash, RoutingConstraints, WorkerAffinityTarget, WorkerConfigLike,
    WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::queue::{DEFAULT_MAX_BATCHED_TOKENS, SchedulerBookingDescriptor};
use dynamo_kv_router::scheduling::{
    AdmissionAttempt, AdmittedSchedulingResponse, LocalScheduler, OverlapAnalysis, ScheduleMode,
    ScheduleRequest,
};
use dynamo_kv_router::sequences::topology::WorkerDpRange;
use dynamo_kv_router::services::selection::affinity::{
    self, AcquireStep, Hold, SessionAffinityConfig,
};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, RoutingPartitionRef, SessionContext,
    TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tokio::runtime::{Handle, Runtime};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::kv::OverlapScores;

fn pyerr(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn elapsed(now_ms: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(now_ms / 1000.0).map_err(pyerr)
}

#[derive(Clone, PartialEq)]
struct WorkerConfig {
    dp_size: u32,
    blocks: u64,
    tokens: u64,
}
impl WorkerConfigLike for WorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        self.dp_size
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        Some(self.tokens)
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(self.blocks)
    }
}

type Scheduler = LocalScheduler<NoopSequencePublisher, WorkerConfig, DefaultWorkerSelector>;
type AdmissionFuture =
    Pin<Box<dyn Future<Output = anyhow::Result<AdmittedSchedulingResponse>> + Send>>;
type Admission = (String, u64, u32, u32, u32);

#[derive(Default)]
struct Notified(AtomicBool);
impl Wake for Notified {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::Release);
    }
}

/// A native LocalScheduler with explicit time and no request expiry or endpoint.
#[pyclass(module = "dynamo._core", unsendable)]
pub(crate) struct OfflineRouter {
    scheduler: Arc<Scheduler>,
    slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    workers: watch::Sender<HashMap<u64, WorkerConfig>>,
    template: WorkerConfig,
    config: KvRouterConfig,
    tracking: TrackingHashContext,
    block_size: u32,
    pending: Vec<(String, AdmissionFuture)>,
    bookings: HashMap<String, SchedulerBookingDescriptor>,
    runtime: Runtime,
    epoch: Instant,
    elapsed: Duration,
}

#[pymethods]
impl OfflineRouter {
    #[new]
    #[pyo3(signature = (block_size, num_workers, dp_size, num_gpu_blocks, max_num_batched_tokens=None, router_config_json="{}", selector_seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        block_size: u32,
        num_workers: usize,
        dp_size: u32,
        num_gpu_blocks: u64,
        max_num_batched_tokens: Option<u64>,
        router_config_json: &str,
        selector_seed: Option<u64>,
    ) -> PyResult<Self> {
        if Handle::try_current().is_ok() {
            return Err(pyerr(
                "OfflineRouter must run outside an active Tokio runtime",
            ));
        }
        if block_size == 0
            || num_workers == 0
            || dp_size == 0
            || num_gpu_blocks == 0
            || max_num_batched_tokens == Some(0)
        {
            return Err(pyerr("worker counts and capacities must be positive"));
        }
        let config: KvRouterConfig = serde_json::from_str(router_config_json).map_err(pyerr)?;
        if config.request_classifier_config().map_err(pyerr)?.is_some()
            || config
                .selected_worker_selection_policy_instance()
                .map_err(pyerr)?
                .is_some()
            || config.router_prefill_load_model == RouterPrefillLoadModel::Ais
        {
            return Err(pyerr(
                "OfflineRouter requires the builtin selector and a native prefill load model",
            ));
        }
        let tracking = TrackingHashContext::from_config(&config).map_err(pyerr)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .map_err(pyerr)?;
        let entered = runtime.enter();
        let epoch = Instant::now();
        let template = WorkerConfig {
            dp_size,
            blocks: num_gpu_blocks,
            tokens: max_num_batched_tokens.unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
        };
        let catalog = (0..num_workers as u64)
            .map(|id| (id, template.clone()))
            .collect::<HashMap<_, _>>();
        let slots = Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
            NoopSequencePublisher,
            block_size as usize,
            catalog.keys().map(|&id| (id, (0, dp_size))).collect(),
            false,
            0,
            "offline",
        ));
        let (workers, receiver) = watch::channel(catalog);
        let selector = match selector_seed {
            Some(seed) => DefaultWorkerSelector::new_seeded(Some(config.clone()), "offline", seed),
            None => DefaultWorkerSelector::new(Some(config.clone()), "offline"),
        };
        let scheduler = Arc::new(LocalScheduler::new(
            Arc::clone(&slots),
            receiver,
            config.configured_policy_profile().map_err(pyerr)?,
            block_size,
            selector,
            None,
            None,
            None,
            None,
            Duration::from_secs(60),
            config.router_track_prefill_tokens,
            CancellationToken::new(),
            "offline",
            false,
        ));
        drop(entered);
        Ok(Self {
            scheduler,
            slots,
            workers,
            template,
            config,
            tracking,
            block_size,
            pending: Vec::new(),
            bookings: HashMap::new(),
            runtime,
            epoch,
            elapsed: Duration::ZERO,
        })
    }

    fn advance_clock(&mut self, now_ms: f64) -> PyResult<()> {
        let next = elapsed(now_ms)?;
        if next < self.elapsed || self.epoch.checked_add(next).is_none() {
            return Err(pyerr("invalid or backwards router clock"));
        }
        if Handle::try_current().is_ok() {
            return Err(pyerr(
                "OfflineRouter must run outside an active Tokio runtime",
            ));
        }
        self.runtime
            .block_on(tokio::time::advance(next - self.elapsed));
        self.elapsed = next;
        Ok(())
    }

    #[pyo3(signature = (request_id, input_length, max_output_tokens, overlaps, local_block_hashes, sequence_hashes=None, tokens=None, affinity_target=None, priority=0.0, strict_priority=0, policy_class=None, session_id=None))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &mut self,
        request_id: String,
        input_length: usize,
        max_output_tokens: u32,
        overlaps: &OverlapScores,
        local_block_hashes: Vec<u64>,
        sequence_hashes: Option<Vec<u64>>,
        tokens: Option<Vec<u32>>,
        affinity_target: Option<(u64, u32)>,
        priority: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        session_id: Option<String>,
    ) -> PyResult<()> {
        if self.bookings.contains_key(&request_id)
            || self.pending.iter().any(|(id, _)| id == &request_id)
        {
            return Err(pyerr("duplicate routing request ID"));
        }
        let blocks = input_length / self.block_size as usize;
        if request_id.is_empty()
            || !priority.is_finite()
            || priority < 0.0
            || local_block_hashes.len() != blocks
            || sequence_hashes.as_ref().is_some_and(|h| h.len() != blocks)
            || tokens.as_ref().is_some_and(|t| t.len() != input_length)
        {
            return Err(pyerr(
                "invalid request identity, priority, token or hash lengths",
            ));
        }
        let token_seq = if !self.config.router_track_active_blocks {
            None
        } else if !self.config.router_assume_kv_reuse {
            self.config.random_seq_hashes_for_tracking(blocks)
        } else if self.tracking.algorithm() == TrackingHashAlgorithm::PublicXxh3V1
            && sequence_hashes.is_some()
        {
            sequence_hashes
        } else {
            let tokens = tokens.as_deref().ok_or_else(|| {
                pyerr("tracking requires prompt token IDs or compatible sequence hashes")
            })?;
            self.config.compute_seq_hashes_for_tracking_with_context(
                &self.tracking,
                TrackingHashScope {
                    partition: RoutingPartitionRef::new("offline", "default"),
                    block_size: self.block_size,
                },
                tokens,
                None,
                BlockHashOptions::default(),
                None,
            )
        };
        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: overlaps.inner.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        let target = affinity_target.map(|(worker_id, dp_rank)| WorkerAffinityTarget {
            worker_id,
            dp_rank: Some(dp_rank),
        });
        let request = ScheduleRequest {
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: request_id.clone(),
            },
            token_seq,
            block_hashes: Some(local_block_hashes.into_iter().map(LocalBlockHash).collect()),
            isl_tokens: input_length,
            overlap: OverlapAnalysis::new(&self.config, self.block_size, &tiered).signals(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            router_config_override: None,
            lora_name: None,
            priority_jump: priority,
            strict_priority,
            policy_class,
            session_context: session_id.map(|id| SessionContext::new(id, None, None, None)),
            expected_output_tokens: Some(max_output_tokens),
            affinity_target: target,
            pinned_worker: affinity_target.map(|(id, rank)| WorkerWithDpRank::new(id, rank)),
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            shared_cache_hits: None,
        };
        let scheduler = Arc::clone(&self.scheduler);
        self.pending.push((
            request_id,
            Box::pin(async move { Ok(scheduler.schedule_request_admitted(request).await?) }),
        ));
        Ok(())
    }

    /// Drive native admission until quiescent; no host-side selection or queue policy.
    fn poll(&mut self) -> PyResult<Vec<Admission>> {
        let wake = Arc::new(Notified::default());
        let waker = Waker::from(Arc::clone(&wake));
        let mut context = Context::from_waker(&waker);
        let mut ready = Vec::new();
        loop {
            wake.0.store(false, Ordering::Release);
            let entered = self.runtime.enter();
            let mut i = 0;
            while i < self.pending.len() {
                let Poll::Ready(result) = self.pending[i].1.as_mut().poll(&mut context) else {
                    i += 1;
                    continue;
                };
                let (id, _) = self.pending.remove(i);
                let admitted = result.map_err(pyerr)?;
                let AdmissionAttempt::Tracked(attempt_id) = admitted.attempt else {
                    return Err(pyerr("native admission did not return a booking"));
                };
                let response = admitted.response;
                self.bookings.insert(
                    id.clone(),
                    SchedulerBookingDescriptor {
                        request_id: id.clone(),
                        worker: response.best_worker,
                        attempt_id,
                    },
                );
                ready.push((
                    id,
                    response.best_worker.worker_id,
                    response.best_worker.dp_rank,
                    response.effective_overlap_blocks.floor() as u32,
                    (response.max_raw_cached_tokens.unwrap_or_default() / self.block_size as usize)
                        as u32,
                ));
            }
            drop(entered);
            self.run(self.scheduler.admission_barrier())?
                .map_err(pyerr)?;
            if !wake.0.swap(false, Ordering::AcqRel) {
                break;
            }
        }
        Ok(ready)
    }

    fn cancel(&mut self, request_id: &str) -> bool {
        let before = self.pending.len();
        self.pending.retain(|(id, _)| id != request_id);
        before != self.pending.len()
    }

    fn prefill_complete(&self, request_id: &str) -> PyResult<()> {
        if let Some(booking) = self.bookings.get(request_id) {
            self.run(self.scheduler.mark_prefill_completed_if_booking(booking))?
                .map_err(pyerr)?;
        }
        Ok(())
    }

    fn free(&mut self, request_id: &str) -> PyResult<()> {
        if let Some(booking) = self.bookings.remove(request_id) {
            self.run(self.scheduler.free_if_booking(&booking))?
                .map_err(pyerr)?;
        }
        Ok(())
    }

    #[getter]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn add_worker(&mut self, worker_id: u64) -> PyResult<()> {
        let mut workers = self.workers.borrow().clone();
        if workers.contains_key(&worker_id) {
            return Err(pyerr("worker already exists"));
        }
        self.slots
            .upsert_worker(WorkerDpRange::new(worker_id, 0, self.template.dp_size))
            .map_err(pyerr)?;
        workers.insert(worker_id, self.template.clone());
        self.workers.send_replace(workers);
        Ok(())
    }

    fn remove_worker(&mut self, worker_id: u64) {
        self.workers.send_modify(|workers| {
            workers.remove(&worker_id);
        });
    }

    fn unregister_worker(&self, worker_id: u64) -> PyResult<()> {
        self.slots.unregister_worker(worker_id).map_err(pyerr)
    }
}

impl OfflineRouter {
    fn run<F: Future>(&self, future: F) -> PyResult<F::Output> {
        if Handle::try_current().is_ok() {
            return Err(pyerr(
                "OfflineRouter must run outside an active Tokio runtime",
            ));
        }
        let (value, now) = self
            .runtime
            .block_on(async { (future.await, Instant::now()) });
        if now.duration_since(self.epoch) != self.elapsed {
            return Err(pyerr(
                "native operation advanced time outside the host clock",
            ));
        }
        Ok(value)
    }
}

#[pyclass(module = "dynamo._core")]
pub(crate) struct SessionAffinity {
    inner: affinity::SessionAffinity,
    epoch: Instant,
}

#[pymethods]
impl SessionAffinity {
    #[new]
    fn new(ttl_seconds: f64) -> PyResult<Self> {
        let epoch = Instant::now();
        let config =
            SessionAffinityConfig::new(Duration::try_from_secs_f64(ttl_seconds).map_err(pyerr)?);
        Ok(Self {
            inner: affinity::SessionAffinity::with_manual_clock(config, epoch).map_err(pyerr)?,
            epoch,
        })
    }
    fn advance_clock(&self, now_ms: f64) -> PyResult<()> {
        let now = self
            .epoch
            .checked_add(elapsed(now_ms)?)
            .ok_or_else(|| pyerr("affinity clock overflow"))?;
        self.inner.advance_clock(now).map_err(pyerr)
    }
    fn try_acquire(&self, key: &str) -> PyResult<Option<AffinityHold>> {
        match self.inner.try_acquire(key, None).map_err(pyerr)? {
            AcquireStep::Held(hold) => Ok(Some(AffinityHold { inner: Some(hold) })),
            AcquireStep::Wait(_) => Ok(None),
        }
    }
    fn commit(
        &self,
        hold: &mut AffinityHold,
        worker_id: u64,
        dp_rank: u32,
    ) -> PyResult<AffinityLease> {
        let hold = hold
            .inner
            .take()
            .ok_or_else(|| pyerr("affinity hold was already released or committed"))?;
        Ok(AffinityLease {
            inner: Some(
                self.inner
                    .commit(
                        hold,
                        WorkerAffinityTarget {
                            worker_id,
                            dp_rank: Some(dp_rank),
                        },
                    )
                    .map_err(pyerr)?,
            ),
        })
    }
}

#[pyclass(module = "dynamo._core")]
pub(crate) struct AffinityHold {
    inner: Option<Hold>,
}
#[pymethods]
impl AffinityHold {
    #[getter]
    fn target(&self) -> Option<(u64, u32)> {
        self.inner
            .as_ref()?
            .target()
            .and_then(|t| t.dp_rank.map(|rank| (t.worker_id, rank)))
    }
    fn release(&mut self) {
        self.inner.take();
    }
}

#[pyclass(module = "dynamo._core")]
pub(crate) struct AffinityLease {
    inner: Option<affinity::AffinityLease>,
}
#[pymethods]
impl AffinityLease {
    fn release(&mut self) {
        self.inner.take();
    }
}

#[pyfunction]
pub(crate) fn subagent_group_affinity_id(parent_session_id: &str) -> String {
    affinity::subagent_group_affinity_id(parent_session_id)
}
