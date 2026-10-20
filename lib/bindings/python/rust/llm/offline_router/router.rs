// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Modified for the standalone, host-controlled Dynamo routing API.
// Derived from https://github.com/ai-dynamo/dynamo/blob/6d6d36d7bfa63d80a0302df637c08f85523aa5b6/lib/mocker/src/replay/offline/extensions/kv_router/mod.rs

use dynamo_custom_policy_builtin::DefaultWorkerSelector;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dynamo_kv_router::LocalBlockHash;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::indexer::{MatchDetails, TieredMatchDetails};
use dynamo_kv_router::protocols::WorkerAffinityTarget;
use dynamo_kv_router::protocols::{
    BlockHashOptions, OverlapScores, RouterEvent, RoutingConstraints, WorkerId, WorkerWithDpRank,
    compute_block_hash_for_seq,
};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::scheduling::{
    AdmissionAttempt, AdmittedSchedulingResponse, LocalScheduler, OverlapAnalysis, OverlapSignals,
    ScheduleMode, ScheduleRequest,
};
use dynamo_kv_router::sequences::topology::WorkerDpRange;
use dynamo_kv_router::services::selection::affinity::Hold;
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, RadixTree, RoutingPartitionRef,
    SessionContext, TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
};
use tokio::sync::watch;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::runtime::{ReplayRuntime, WakeFlag};
use uuid::Uuid;

use super::ReplayPrefillLoadEstimator;
use super::affinity::{ReplayAffinity, ReplayAffinityConfig, RoutingEvidence};
use super::helpers::NativeRouterEngineArgs;
use super::helpers::{
    ReplayWorkerConfig, replay_router_config, replay_selector_with_seed, replay_slots,
    replay_worker_config, replay_workers_with_configs,
};
use super::wire::{
    Placement, PlacementCacheSample, RequestHashes as ReplayRequestHashes,
    RoutingRequest as DirectRequest, WorkerTopology,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerAdmission {
    uuid: Uuid,
    worker_idx: usize,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    isl_blocks: u32,
}

#[derive(Debug, Default)]
pub(crate) struct RouterEffects {
    admissions: Vec<WorkerAdmission>,
}

struct SyncReplayIndexer {
    block_size: u32,
    tree: RadixTree,
    /// Disaggregated decode placement never observes KV events, so its tree
    /// stays empty and hashing a prompt to query it would be wasted work.
    has_indexed_events: bool,
}

impl SyncReplayIndexer {
    fn new(block_size: u32) -> Self {
        Self {
            block_size,
            tree: RadixTree::new(),
            has_indexed_events: false,
        }
    }

    fn find_matches_for_request(&self, tokens: &[u32], lora_name: Option<&str>) -> OverlapScores {
        if !self.has_indexed_events {
            return OverlapScores::default();
        }
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        self.tree.find_matches(sequence, false)
    }

    fn find_matches_for_hashes(&self, local_block_hashes: Vec<LocalBlockHash>) -> OverlapScores {
        self.tree.find_matches(local_block_hashes, false)
    }

    fn apply_event(&mut self, event: RouterEvent) -> Result<()> {
        // TODO: support lower tier events in replay indexer
        if !event.storage_tier.is_gpu() {
            return Ok(());
        }
        self.has_indexed_events = true;
        self.tree.apply_event(event).map_err(Into::into)
    }
}

pub(super) struct PendingRequest {
    pub(super) uuid: Uuid,
    pub(super) token_seq: Option<Vec<u64>>,
    pub(super) isl_tokens: usize,
    pub(super) overlaps: OverlapSignals,
    pub(super) expected_output_tokens: Option<u32>,
    pub(super) priority_jump: f64,
    pub(super) strict_priority: u32,
    pub(super) policy_class: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) group_key: Option<String>,
}

impl PendingRequest {
    fn request_id(&self) -> String {
        self.uuid.to_string()
    }

    fn scheduling_request(mut self, target: Option<WorkerAffinityTarget>) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::TrackedWithLifecycle {
                request_id: self.request_id(),
            },
            token_seq: self.token_seq.take(),
            block_hashes: None,
            isl_tokens: self.isl_tokens,
            overlap: self.overlaps,
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            router_config_override: None,
            lora_name: None,
            priority_jump: self.priority_jump,
            strict_priority: self.strict_priority,
            policy_class: self.policy_class,
            session_context: self
                .session_id
                .map(|id| SessionContext::new(id, None, None, None)),
            expected_output_tokens: self.expected_output_tokens,
            affinity_target: target,
            pinned_worker: target.and_then(|target| {
                target
                    .dp_rank
                    .map(|rank| WorkerWithDpRank::new(target.worker_id, rank))
            }),
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            shared_cache_hits: None,
        }
    }
}

type NativeScheduler =
    LocalScheduler<NoopSequencePublisher, ReplayWorkerConfig, DefaultWorkerSelector>;

struct PendingAdmission {
    uuid: Uuid,
    future:
        Pin<Box<dyn Future<Output = Result<(AdmittedSchedulingResponse, Option<Hold>)>> + Send>>,
    group_key: Option<String>,
    isl_blocks: u32,
}

struct OfflineReplayRouter {
    config: KvRouterConfig,
    block_size: u32,
    dp_size: u32,
    worker_config_template: ReplayWorkerConfig,
    workers_with_configs: HashMap<WorkerId, ReplayWorkerConfig>,
    slots: Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    scheduler: Arc<NativeScheduler>,
    workers_tx: watch::Sender<HashMap<WorkerId, ReplayWorkerConfig>>,
    pending: Vec<PendingAdmission>,
    bookings: HashMap<Uuid, SchedulerBookingDescriptor>,
    runtime: ReplayRuntime,
    indexer: SyncReplayIndexer,
    decay_time_epoch: Instant,
    tracking_hash: TrackingHashContext,
    affinity: Option<ReplayAffinity>,
}

pub struct KvRouterPlacement {
    router: OfflineReplayRouter,
}

impl KvRouterPlacement {
    /// Native scheduler state for diagnostics; no host load model is inferred.
    pub(super) fn loads(&self) -> Result<serde_json::Value> {
        let now = self.router.runtime.block_on(async { Instant::now() })?;
        let blocks = self.router.slots.active_blocks();
        let tokens = self.router.slots.active_tokens(now);
        let mut workers = self
            .router
            .workers_with_configs
            .keys()
            .copied()
            .collect::<Vec<_>>();
        workers.sort_unstable();
        Ok(serde_json::Value::Array(
            workers
                .into_iter()
                .flat_map(|worker_id| {
                    (0..self.router.dp_size).map(move |rank| WorkerWithDpRank::new(worker_id, rank))
                })
                .map(|worker| {
                    serde_json::json!({
                        "worker_id": worker.worker_id, "dp_rank": worker.dp_rank,
                        "active_blocks": blocks.get(&worker).copied().unwrap_or_default(),
                        "active_prefill_tokens": tokens.get(&worker).copied().unwrap_or_default(),
                    })
                })
                .collect(),
        ))
    }

    pub(super) fn new_with_selector_seed(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        let router = match selector_seed {
            Some(seed) => OfflineReplayRouter::new_with_selector_seed(
                args,
                router_config,
                prefill_load_estimator,
                num_workers,
                Some(seed),
            )?,
            None => {
                OfflineReplayRouter::new(args, router_config, prefill_load_estimator, num_workers)?
            }
        };
        Ok(Self { router })
    }

    pub(super) fn with_affinity(
        mut self,
        config: Option<ReplayAffinityConfig>,
        role: &'static str,
        capture: bool,
        evidence: RoutingEvidence,
    ) -> Result<Self> {
        if let Some(config) = config {
            self.router.affinity = Some(ReplayAffinity::new(
                config,
                self.router.decay_time_epoch,
                role,
                capture,
                evidence,
            )?);
        }
        Ok(self)
    }

    fn placement(&self, admission: WorkerAdmission) -> Placement {
        Placement {
            request_id: admission.uuid,
            scheduler_id: admission.worker_idx,
            placement_replica_id: None,
            reported_overlap_tokens: admission.overlap_blocks as usize
                * self.router.block_size as usize,
            cache_sample: Some(PlacementCacheSample {
                overlap_blocks: admission.overlap_blocks,
                best_available_overlap_blocks: admission.best_available_overlap_blocks,
                isl_blocks: admission.isl_blocks,
            }),
        }
    }

    fn placements(&self, admissions: Vec<WorkerAdmission>) -> Vec<Placement> {
        admissions
            .into_iter()
            .map(|admission| self.placement(admission))
            .collect()
    }
}

impl KvRouterPlacement {
    pub(super) fn place(
        &mut self,
        request: &DirectRequest,
        hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<serde_json::Value> {
        let request_id = request
            .uuid
            .context("KV placement requires a request UUID")?;
        let admissions = self
            .router
            .on_request_arrival_for_session(
                request,
                request.max_output_tokens,
                hashes,
                session_id,
                now_ms,
            )?
            .admissions;
        let mut placement = None;
        let mut released = Vec::new();
        for admission in admissions {
            let ready = self.placement(admission);
            if ready.request_id == request_id {
                placement = Some(ready);
            } else {
                released.push(ready);
            }
        }
        Ok(serde_json::json!({"placement": placement, "released": released}))
    }

    pub(super) fn observe(
        &mut self,
        observation: Vec<RouterEvent>,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        let effects = self.router.on_kv_events(observation)?;
        Ok(self.placements(effects.admissions))
    }

    pub(super) fn cancel_pending(&mut self, request_id: Uuid) -> bool {
        self.router.cancel_pending(request_id)
    }

    pub(super) fn dispatch_committed(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.router.advance_time(now_ms)?;
        if let Some(affinity) = &mut self.router.affinity {
            affinity.advance(now_ms)?;
            affinity.commit(request_id)?;
        }
        Ok(())
    }

    pub(super) fn dispatch_aborted(&mut self, request_id: Uuid, now_ms: f64) -> Result<()> {
        self.router.advance_time(now_ms)?;
        if let Some(affinity) = &mut self.router.affinity {
            affinity.release(request_id, true)?;
        }
        self.router.free_booking(request_id)?;
        Ok(())
    }

    pub(super) fn advance_clock(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        let admissions = self.router.collect_admissions()?;
        Ok(self.placements(admissions))
    }

    pub(super) fn next_wakeup_ms(&self) -> Option<f64> {
        self.router
            .affinity
            .as_ref()
            .filter(|affinity| affinity.ready && self.router.pending_count() > 0)
            .map(|affinity| affinity.now_ms)
    }

    pub(super) fn request_terminal(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        let effects = self.router.on_request_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    pub(super) fn prefill_completed(
        &mut self,
        request_id: Uuid,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        let effects = self.router.on_prefill_completed(request_id, now_ms)?;
        Ok(self.placements(effects.admissions))
    }

    pub(super) fn pending_count(&self) -> usize {
        self.router.pending_count()
    }

    pub(super) fn worker_ready(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        self.router.add_worker(worker.worker_id)?;
        Ok(Vec::new())
    }

    pub(super) fn worker_draining(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        self.router.remove_worker(worker.worker_id)?;
        Ok(Vec::new())
    }

    pub(super) fn worker_removed(
        &mut self,
        worker: WorkerTopology,
        now_ms: f64,
    ) -> Result<Vec<Placement>> {
        self.router.advance_time(now_ms)?;
        self.router.finalize_worker_removal(worker.worker_id)?;
        Ok(Vec::new())
    }

    pub(super) fn topology_settled(&mut self, now_ms: f64) -> Result<Vec<Placement>> {
        let effects = self.router.on_topology_changed(now_ms)?;
        Ok(self.placements(effects.admissions))
    }
}

impl OfflineReplayRouter {
    fn new(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
    ) -> Result<Self> {
        Self::new_with_selector_seed(
            args,
            router_config,
            prefill_load_estimator,
            num_workers,
            None,
        )
    }

    fn new_with_selector_seed(
        args: &NativeRouterEngineArgs,
        router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
        num_workers: usize,
        selector_seed: Option<u64>,
    ) -> Result<Self> {
        let runtime = ReplayRuntime::new()?;
        let _entered = runtime.enter();
        let config = replay_router_config(router_config);
        let tracking_hash = TrackingHashContext::from_config(&config)?;
        let worker_config_template = replay_worker_config(args);
        let workers_with_configs = replay_workers_with_configs(args, num_workers);
        let slots = replay_slots(args, &workers_with_configs);
        let selector = replay_selector_with_seed(&config, selector_seed)?;
        let profile = config
            .configured_policy_profile()
            .map_err(anyhow::Error::from)?;

        let (workers_tx, workers_rx) = watch::channel(workers_with_configs.clone());
        let scheduler = Arc::new(LocalScheduler::new(
            Arc::clone(&slots),
            workers_rx,
            profile,
            args.block_size,
            selector,
            prefill_load_estimator,
            None,
            None,
            None,
            Duration::from_secs(60),
            config.router_track_prefill_tokens,
            CancellationToken::new(),
            "replay",
            false,
        ));
        let decay_time_epoch = runtime.epoch();
        drop(_entered);
        Ok(Self {
            config,
            block_size: args.block_size,
            dp_size: args.dp_size.max(1),
            worker_config_template,
            workers_with_configs,
            workers_tx,
            slots,
            scheduler,
            pending: Vec::new(),
            bookings: HashMap::new(),
            indexer: SyncReplayIndexer::new(args.block_size),
            decay_time_epoch,
            runtime,
            tracking_hash,
            affinity: None,
        })
    }

    fn on_request_arrival_for_session(
        &mut self,
        request: &DirectRequest,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
        now_ms: f64,
    ) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        let pending =
            self.build_pending_request(request, max_output_tokens, replay_hashes, session_id)?;
        let uuid = pending.uuid;
        let group_key = pending.group_key.clone();
        let table = self
            .affinity
            .as_ref()
            .map(|affinity| affinity.table.clone());
        let scheduler = Arc::clone(&self.scheduler);
        let block_size = self.block_size;
        let isl_blocks =
            u32::try_from(pending.isl_tokens.div_ceil(block_size as usize)).unwrap_or(u32::MAX);
        let key = group_key.clone();
        let future = Box::pin(async move {
            let hold = match (table, key) {
                (Some(table), Some(key)) => Some(table.acquire(&key, None).await?),
                _ => None,
            };
            let target = hold.as_ref().and_then(Hold::target);
            let admitted = scheduler
                .schedule_request_admitted(pending.scheduling_request(target))
                .await?;
            Ok((admitted, hold))
        });
        self.pending.push(PendingAdmission {
            uuid,
            future,
            group_key,
            isl_blocks,
        });
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    /// Poll native admission futures to quiescence, with the scheduler's own
    /// update acknowledgement as the actor barrier. No simulator queue policy.
    fn collect_admissions(&mut self) -> Result<Vec<WorkerAdmission>> {
        let wake = Arc::new(WakeFlag::default());
        let waker = wake.waker();
        let mut context = TaskContext::from_waker(&waker);
        let mut admissions = Vec::new();
        loop {
            wake.take();
            let _entered = self.runtime.enter();
            let mut index = 0;
            while index < self.pending.len() {
                let Poll::Ready(result) = self.pending[index].future.as_mut().poll(&mut context)
                else {
                    index += 1;
                    continue;
                };
                let pending = self.pending.remove(index);
                let (admitted, hold) = result?;
                let selection = admitted.response;
                let AdmissionAttempt::Tracked(attempt_id) = admitted.attempt else {
                    anyhow::bail!("Dynamo returned an untracked simulator admission");
                };
                self.bookings.insert(
                    pending.uuid,
                    SchedulerBookingDescriptor {
                        request_id: pending.uuid.to_string(),
                        worker: selection.best_worker,
                        attempt_id,
                    },
                );
                let overlap_blocks = selection.effective_overlap_blocks.floor() as u32;
                let best_available_overlap_blocks = (selection.max_raw_cached_tokens.unwrap_or(0)
                    / self.block_size as usize)
                    as u32;
                if let Some(affinity) = &mut self.affinity {
                    affinity.stage(
                        pending.uuid,
                        hold.context("native affinity admission has no hold")?,
                        WorkerAffinityTarget {
                            worker_id: selection.best_worker.worker_id,
                            dp_rank: Some(selection.best_worker.dp_rank),
                        },
                        pending
                            .group_key
                            .as_deref()
                            .context("native affinity admission has no group key")?,
                        overlap_blocks,
                        best_available_overlap_blocks,
                    )?;
                }
                let worker_idx = usize::try_from(selection.best_worker.worker_id)?
                    .checked_mul(self.dp_size as usize)
                    .and_then(|base| base.checked_add(selection.best_worker.dp_rank as usize))
                    .context("selected worker/rank index overflow")?;
                admissions.push(WorkerAdmission {
                    uuid: pending.uuid,
                    worker_idx,
                    overlap_blocks,
                    best_available_overlap_blocks,
                    isl_blocks: pending.isl_blocks,
                });
            }
            drop(_entered);
            self.runtime
                .block_on(self.scheduler.admission_barrier())??;
            if !wake.take() {
                break;
            }
        }
        if let Some(affinity) = &mut self.affinity {
            affinity.ready = false;
        }
        Ok(admissions)
    }

    fn advance_time(&mut self, now_ms: f64) -> Result<()> {
        self.runtime.advance_to_ms(now_ms)?;
        if let Some(affinity) = &mut self.affinity {
            affinity.advance(now_ms)?;
        }
        Ok(())
    }

    fn free_booking(&mut self, uuid: Uuid) -> Result<()> {
        if let Some(booking) = self.bookings.remove(&uuid) {
            self.runtime
                .block_on(self.scheduler.free_if_booking(&booking))??;
        }
        Ok(())
    }

    fn on_kv_events(&mut self, events: Vec<RouterEvent>) -> Result<RouterEffects> {
        if let Some(affinity) = &self.affinity {
            anyhow::ensure!(
                events.iter().all(|event| event.storage_tier.is_gpu()),
                "conversation affinity supports device KV events only"
            );
            affinity.record_events(events.len())?;
        }
        for event in events {
            let worker_id = event.worker_id;
            let event_id = event.event.event_id;
            let dp_rank = event.event.dp_rank;
            self.indexer.apply_event(event).with_context(|| {
                format!(
                    "failed to apply replay KV event worker={worker_id} dp_rank={dp_rank} event_id={event_id}"
                )
            })?;
        }
        Ok(RouterEffects::default())
    }

    fn on_prefill_completed(&mut self, uuid: Uuid, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        if let Some(booking) = self.bookings.get(&uuid) {
            self.runtime
                .block_on(self.scheduler.mark_prefill_completed_if_booking(booking))??;
        }
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    fn on_request_completed(&mut self, uuid: Uuid, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        if let Some(affinity) = &mut self.affinity {
            affinity.release(uuid, false)?;
        }
        self.free_booking(uuid)?;
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    fn cancel_pending(&mut self, uuid: Uuid) -> bool {
        let before = self.pending.len();
        // Dropping the native lifecycle future cancels its admission lease.
        self.pending.retain(|pending| pending.uuid != uuid);
        let removed = self.pending.len() != before;
        if let Some(affinity) = &mut self.affinity {
            affinity.ready |= removed;
        }
        removed
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Register a new worker with the router without disturbing existing slot state.
    fn add_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        if self
            .workers_with_configs
            .insert(wid, self.worker_config_template.clone())
            .is_some()
        {
            return Err(anyhow!("router worker {worker_id} already exists"));
        }
        if let Err(error) = self
            .slots
            .upsert_worker(WorkerDpRange::new(wid, 0, self.dp_size))
        {
            self.workers_with_configs.remove(&wid);
            return Err(error.into());
        }

        self.workers_tx
            .send_replace(self.workers_with_configs.clone());
        Ok(())
    }

    /// Remove a worker from routing eligibility.
    ///
    /// Only removes the worker from the config map so the selector won't
    /// pick it for new requests.  The radix tree and active-sequence slots
    /// are left intact so that in-flight requests on this worker can still
    /// complete (free / mark_prefill_completed) and KV events can still
    /// reference existing blocks without "parent block not found" errors.
    /// Stale slot and indexer state is harmless — the selector and
    /// `all_workers_busy` both skip workers absent from `workers_with_configs`.
    fn remove_worker(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.workers_with_configs.remove(&wid);
        self.workers_tx
            .send_replace(self.workers_with_configs.clone());
        Ok(())
    }

    /// Drop the retained topology/cache state after the engine confirms that
    /// the draining worker no longer owns any request lifecycle state.
    fn finalize_worker_removal(&mut self, worker_id: usize) -> Result<()> {
        let wid = worker_id as WorkerId;
        self.slots
            .unregister_worker(wid)
            .map_err(anyhow::Error::from)?;
        self.indexer.tree.remove_worker(wid);
        Ok(())
    }

    fn on_topology_changed(&mut self, now_ms: f64) -> Result<RouterEffects> {
        self.advance_time(now_ms)?;
        Ok(RouterEffects {
            admissions: self.collect_admissions()?,
        })
    }

    fn build_pending_request(
        &self,
        request: &DirectRequest,
        max_output_tokens: usize,
        replay_hashes: Option<ReplayRequestHashes>,
        session_id: Option<String>,
    ) -> Result<PendingRequest> {
        if self.affinity.is_some() {
            anyhow::ensure!(
                request.preferred_dp_rank.is_none() && request.preferred_prefill_dp_rank.is_none(),
                "conversation affinity does not support authored DP-rank pins"
            );
            anyhow::ensure!(
                request.policy_class.is_none(),
                "conversation affinity does not support authored policy classes"
            );
        }
        let input_length = request.input_length;
        let uuid = request
            .uuid
            .ok_or_else(|| anyhow!("offline replay requires requests to have stable UUIDs"))?;
        let (priority_jump, strict_priority) = request.router_priorities();
        let (overlaps, token_seq) = match replay_hashes {
            Some(replay_hashes) => {
                let overlaps = self.indexer.find_matches_for_hashes(
                    replay_hashes
                        .local_block_hashes
                        .into_iter()
                        .map(LocalBlockHash)
                        .collect(),
                );
                let token_seq = if !self.config.router_track_active_blocks {
                    None
                } else if self.config.router_assume_kv_reuse
                    && self.tracking_hash.algorithm() == TrackingHashAlgorithm::PublicXxh3V1
                {
                    Some(replay_hashes.sequence_hashes)
                } else if !self.config.router_assume_kv_reuse {
                    self.config
                        .random_seq_hashes_for_tracking(input_length / self.block_size as usize)
                } else {
                    let tokens = request.prompt_tokens_for_placement()?;
                    self.config.compute_seq_hashes_for_tracking_with_context(
                        &self.tracking_hash,
                        self.tracking_hash_scope(),
                        &tokens,
                        None,
                        BlockHashOptions::default(),
                        None,
                    )
                };
                (overlaps, token_seq)
            }
            None => {
                let tokens = request.prompt_tokens_for_placement()?;
                let overlaps = self.indexer.find_matches_for_request(&tokens, None);
                let token_seq = self.config.compute_seq_hashes_for_tracking_with_context(
                    &self.tracking_hash,
                    self.tracking_hash_scope(),
                    &tokens,
                    None,
                    BlockHashOptions::default(),
                    None,
                );
                (overlaps, token_seq)
            }
        };

        let tiered = TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: overlaps,
                ..Default::default()
            },
            ..Default::default()
        };
        let overlaps = OverlapAnalysis::new(&self.config, self.block_size, &tiered).signals();
        Ok(PendingRequest {
            uuid,
            token_seq,
            isl_tokens: input_length,
            overlaps,
            expected_output_tokens: Some(
                u32::try_from(max_output_tokens)
                    .context("max_output_tokens does not fit into u32")?,
            ),
            priority_jump,
            strict_priority,
            policy_class: request.policy_class.clone(),
            group_key: self
                .affinity
                .as_ref()
                .map(|affinity| affinity.config.group_key(request, session_id.as_deref()))
                .transpose()?,
            session_id,
        })
    }

    fn tracking_hash_scope(&self) -> TrackingHashScope<'_> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new("replay", "default"),
            block_size: self.block_size,
        }
    }
}
