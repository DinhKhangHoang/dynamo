// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON values accepted by the host-controlled router, independent of a simulator crate.

use anyhow::{Result, ensure};
use dynamo_kv_router::protocols::{
    KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData, RouterEvent, StorageTier,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use uuid::Uuid;

pub(super) const CONVERSATION_LINEAGE_SCHEMA_V1: &str =
    "aisimulate.agentic.conversation-lineage.v1";

#[derive(Deserialize)]
pub(super) struct ConversationLineage {
    pub schema: String,
    pub root_conversation_id: String,
    pub parent_conversation_id: Option<String>,
}
#[derive(Deserialize)]
pub(super) struct ConversationContext {
    pub play_id: String,
    pub conversation_id: String,
    pub lineage: Option<ConversationLineage>,
}
#[derive(Deserialize)]
pub(super) struct RequestContext {
    pub session_id: Option<String>,
    #[serde(default)]
    pub prompt_token_source: Option<String>,
    pub agentic: Option<ConversationContext>,
}

/// Routing-only projection; unrelated host request metadata is not interpreted.
#[derive(Deserialize)]
pub(super) struct RoutingRequest {
    pub uuid: Option<Uuid>,
    #[serde(default)]
    pub tokens: Vec<u32>,
    pub max_output_tokens: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub preferred_dp_rank: Option<u32>,
    pub preferred_prefill_dp_rank: Option<u32>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub strict_priority: u32,
    pub policy_class: Option<String>,
    pub replay_context: Option<RequestContext>,
    #[serde(skip)]
    pub input_length: usize,
}
impl RoutingRequest {
    pub fn router_priorities(&self) -> (f64, u32) {
        (f64::from(self.priority.max(0)), self.strict_priority)
    }
    pub fn prompt_tokens_for_placement(&self) -> Result<Cow<'_, [u32]>> {
        ensure!(
            self.replay_context.as_ref().is_none_or(|context| {
                context
                    .prompt_token_source
                    .as_deref()
                    .is_none_or(|source| source == "materialized")
            }),
            "Dynamo KV Router placement requires authored prompt token IDs or replay hashes; length-only execution tokens are not valid KV identities"
        );
        ensure!(
            self.tokens.len() == self.input_length,
            "prompt tokens are unavailable: provide block hashes for a compact prompt"
        );
        Ok(Cow::Borrowed(&self.tokens))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestHashes {
    pub local_block_hashes: Vec<u64>,
    pub sequence_hashes: Vec<u64>,
}
#[derive(Serialize)]
pub(super) struct PlacementCacheSample {
    pub overlap_blocks: u32,
    pub best_available_overlap_blocks: u32,
    pub isl_blocks: u32,
}
#[derive(Serialize)]
pub(super) struct Placement {
    pub request_id: Uuid,
    pub scheduler_id: usize,
    pub reported_overlap_tokens: usize,
    pub cache_sample: Option<PlacementCacheSample>,
    pub placement_replica_id: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkerTopology {
    pub worker_id: usize,
    pub scheduler_ids: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum EventData {
    Stored {
        #[serde(flatten)]
        data: KvCacheStoreData,
        session_id: Option<String>,
    },
    Removed(KvCacheRemoveData),
}
#[derive(Deserialize)]
pub(super) struct CacheEvent {
    event_id: u64,
    dp_rank: u32,
    #[serde(default)]
    tier: StorageTier,
    data: EventData,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Observation {
    pub stage: String,
    worker_id: u64,
    dp_rank: u32,
    events: Vec<CacheEvent>,
}
impl Observation {
    pub fn validate_topology(
        &self,
        workers: &std::collections::HashSet<usize>,
        dp_size: u32,
    ) -> Result<()> {
        ensure!(
            usize::try_from(self.worker_id)
                .ok()
                .is_some_and(|worker| workers.contains(&worker))
                && self.dp_rank < dp_size,
            "observation references an unknown worker or rank"
        );
        Ok(())
    }
    pub fn into_events(self) -> Result<Vec<RouterEvent>> {
        ensure!(
            matches!(self.stage.as_str(), "aggregated" | "prefill" | "decode"),
            "invalid observation stage"
        );
        self.events
            .into_iter()
            .map(|event| {
                ensure!(
                    event.dp_rank == self.dp_rank,
                    "observation rank disagrees with event rank"
                );
                ensure!(
                    event.tier == StorageTier::Device,
                    "offline routing supports device KV events only"
                );
                let (data, session_id) = match event.data {
                    EventData::Stored { data, session_id } => {
                        (KvCacheEventData::Stored(data), session_id)
                    }
                    EventData::Removed(data) => (KvCacheEventData::Removed(data), None),
                };
                let mut event = RouterEvent::with_storage_tier(
                    self.worker_id,
                    KvCacheEvent {
                        event_id: event.event_id,
                        dp_rank: event.dp_rank,
                        data,
                    },
                    event.tier,
                );
                event.session_id = session_id;
                Ok(event)
            })
            .collect()
    }
}
