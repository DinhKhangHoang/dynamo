// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn config(affinity: Value) -> Value {
    json!({"schema_version":1,"role":"aggregated","dp_size":1,"num_workers":2,
        "block_size":4,"num_gpu_blocks":128,"max_num_batched_tokens":32,
        "affinity":affinity,"selector_seed":7,"capture_decisions":true})
}
fn handle() -> OfflineRouter {
    OfflineRouter::from_json(&config(json!({"mode":"session","ttl_seconds":1})).to_string())
        .unwrap()
}
fn call(router: &mut OfflineRouter, command: Value) -> Value {
    serde_json::from_str(&router.execute_json(&command.to_string()).unwrap()).unwrap()
}
fn place(id: u128, session: &str, now_ms: f64) -> Value {
    json!({"op":"place","now_ms":now_ms,"input_length":4,"session_id":session,
        "request":{"uuid":Uuid::from_u128(id),"tokens":[1,2,3,4],"max_output_tokens":2}})
}
fn lifecycle(op: &str, id: u128, now_ms: f64) -> Value {
    json!({"op":op,"request_id":Uuid::from_u128(id),"now_ms":now_ms})
}

#[test]
fn same_session_waits_for_dispatch_then_reuses_binding() {
    let mut router = handle();
    let first = call(&mut router, place(1, "s", 0.0));
    let first_worker = first["result"]["placement"]["scheduler_id"].clone();
    assert!(first_worker.is_number());
    assert!(first.get("evidence").is_none());
    let pending = call(&mut router, place(2, "s", 0.0));
    assert!(pending["result"]["placement"].is_null());
    assert_eq!(pending["pending_count"], 1);
    let committed = call(&mut router, lifecycle("dispatch_committed", 1, 0.0));
    assert_eq!(committed["next_wakeup_ms"], 0.0);
    let ready = call(&mut router, json!({"op":"advance_clock","now_ms":0.0}));
    assert_eq!(ready["result"][0]["scheduler_id"], first_worker);
    assert_eq!(ready["pending_count"], 0);
    call(&mut router, lifecycle("dispatch_committed", 2, 0.0));
    let snapshot = call(&mut router, json!({"op":"snapshot"}));
    let evidence = &snapshot["evidence"];
    assert_eq!(evidence["decision_count"], 2);
    assert_eq!(evidence["post_dispatch_checks"], 2);
    assert_eq!(evidence["decisions"][1]["binding_reused"], true);
}

#[test]
fn ttl_uses_host_clock_and_retains_active_leases() {
    let mut router = handle();
    call(&mut router, place(1, "s", 0.0));
    call(&mut router, lifecycle("dispatch_committed", 1, 0.0));
    call(&mut router, place(2, "s", 2000.0));
    call(&mut router, lifecycle("dispatch_committed", 2, 2000.0));
    call(&mut router, lifecycle("request_terminal", 1, 2000.0));
    call(&mut router, lifecycle("request_terminal", 2, 2000.0));
    call(&mut router, place(3, "s", 3001.0));
    let snapshot = call(&mut router, json!({"op":"snapshot"}));
    let decisions = &snapshot["evidence"]["decisions"];
    assert_eq!(decisions[1]["binding_reused"], true);
    assert_eq!(decisions[2]["binding_reused"], false);
}

#[test]
fn abort_releases_waiter_without_committing_affinity() {
    let mut router = handle();
    call(&mut router, place(1, "s", 0.0));
    call(&mut router, place(2, "s", 0.0));
    call(&mut router, lifecycle("dispatch_aborted", 1, 0.0));
    let ready = call(&mut router, json!({"op":"advance_clock","now_ms":0.0}));
    assert_eq!(
        ready["result"][0]["request_id"],
        Uuid::from_u128(2).to_string()
    );
    let snapshot = call(&mut router, json!({"op":"snapshot"}));
    let evidence = &snapshot["evidence"];
    assert_eq!(evidence["dispatch_aborts"], 1);
    assert_eq!(evidence["decisions"][1]["binding_reused"], false);
}

#[test]
fn cancel_drops_native_admission_future() {
    let mut router = handle();
    call(&mut router, place(1, "s", 0.0));
    call(&mut router, place(2, "s", 0.0));
    let canceled = call(
        &mut router,
        json!({"op":"cancel_pending","request_id":Uuid::from_u128(2)}),
    );
    assert_eq!(canceled["result"], true);
    assert_eq!(canceled["pending_count"], 0);
    call(&mut router, lifecycle("dispatch_committed", 1, 0.0));
    assert_eq!(
        call(&mut router, json!({"op":"advance_clock","now_ms":0.0}))["result"],
        json!([])
    );
}

#[test]
fn malformed_or_backwards_commands_poison_the_handle() {
    for invalid in [
        json!({"op":"unknown"}),
        json!({"op":"advance_clock","now_ms":-1}),
        json!({"op":"advance_clock","now_ms":1,"ignored":true}),
    ] {
        let mut router = handle();
        assert!(router.execute_json(&invalid.to_string()).is_err());
        assert!(
            router
                .execute_json(r#"{"op":"status"}"#)
                .unwrap_err()
                .to_string()
                .contains("failed previously")
        );
    }
    let mut router = handle();
    call(&mut router, json!({"op":"advance_clock","now_ms":1}));
    assert!(
        router
            .execute_json(r#"{"op":"advance_clock","now_ms":0}"#)
            .is_err()
    );
}

#[test]
fn compact_prompt_needs_complete_hashes_and_duplicates_are_rejected() {
    let mut router = handle();
    let mut request = place(1, "s", 0.0);
    request["request"]["tokens"] = json!([]);
    assert!(router.execute_json(&request.to_string()).is_err());
    let mut router = handle();
    call(&mut router, place(1, "s", 0.0));
    assert!(
        router
            .execute_json(&place(1, "s", 0.0).to_string())
            .is_err()
    );
}

#[test]
fn invalid_config_and_topology_are_rejected() {
    for (key, value) in [
        ("schema_version", json!(2)),
        ("dp_size", json!(0)),
        ("num_workers", json!(0)),
        ("block_size", json!(0)),
        ("role", json!("other")),
    ] {
        let mut args = config(Value::Null);
        args[key] = value;
        assert!(OfflineRouter::from_json(&args.to_string()).is_err());
    }
    let mut router = handle();
    assert!(router.execute_json(&json!({"op":"worker_ready","now_ms":0,"worker":{"worker_id":2,"scheduler_ids":[3]}}).to_string()).is_err());
}

#[test]
fn native_indexer_observes_physical_cache_events() {
    use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    let mut router = handle();
    let first = call(&mut router, place(1, "s", 0.0));
    let worker = first["result"]["placement"]["scheduler_id"]
        .as_u64()
        .unwrap();
    call(&mut router, lifecycle("dispatch_committed", 1, 0.0));
    let hash = compute_block_hash_for_seq(&[1, 2, 3, 4], 4, BlockHashOptions::default())[0].0;
    call(
        &mut router,
        json!({"op":"observe","now_ms":1,"events":[{
        "stage":"aggregated","worker_id":worker,"dp_rank":0,"events":[{
            "event_id":1,"dp_rank":0,"data":{"stored":{"parent_hash":null,
                "session_id":"s","blocks":[{"block_hash":100,"tokens_hash":hash}]}}
        }]}]}),
    );
    call(&mut router, lifecycle("request_terminal", 1, 1.0));
    let reused = call(&mut router, place(2, "s", 2.0));
    assert_eq!(reused["result"]["placement"]["reported_overlap_tokens"], 4);
    assert_eq!(
        call(&mut router, json!({"op":"snapshot"}))["evidence"]["physical_kv_events"],
        1
    );
}

#[test]
fn sibling_group_keys_use_conversation_parent_within_play() {
    let mut args = config(json!({"mode":"sibling_group","ttl_seconds":1}));
    args["capture_decisions"] = json!(true);
    let mut router = OfflineRouter::from_json(&args.to_string()).unwrap();
    for (id, conversation, parent) in [
        (1, "child-a", "parent"),
        (2, "child-b", "parent"),
        (3, "other-child", "other-parent"),
    ] {
        let mut command = place(id, conversation, 0.0);
        command["request"]["replay_context"] = json!({"agentic":{
            "play_id":"play","conversation_id":conversation,"lineage":{
                "schema":"aisimulate.agentic.conversation-lineage.v1",
                "root_conversation_id":"root","parent_conversation_id":parent}}});
        call(&mut router, command);
        call(&mut router, lifecycle("dispatch_committed", id, 0.0));
    }
    let snapshot = call(&mut router, json!({"op":"snapshot"}));
    let decisions = &snapshot["evidence"]["decisions"];
    assert_eq!(decisions[0]["group_key"], decisions[1]["group_key"]);
    assert_ne!(decisions[1]["group_key"], decisions[2]["group_key"]);
    assert_eq!(decisions[1]["binding_reused"], true);
    assert_eq!(decisions[2]["binding_reused"], false);
}
