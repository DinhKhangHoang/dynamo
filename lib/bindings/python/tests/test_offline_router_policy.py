# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The standalone policy API is usable without a simulation executor or service."""

import json
import subprocess
import sys
import textwrap
import time
from uuid import UUID

import pytest

from dynamo.llm import OfflineRouter

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """Override the bindings suite's service fixture: this API is fully local."""
    yield


def router(mode="session"):
    return OfflineRouter(
        json.dumps(
            {
                "schema_version": 1,
                "role": "aggregated",
                "dp_size": 1,
                "num_workers": 2,
                "block_size": 4,
                "num_gpu_blocks": 128,
                "max_num_batched_tokens": 32,
                "affinity": {"mode": mode, "ttl_seconds": 1},
                "capture_decisions": True,
                "selector_seed": 7,
            }
        )
    )


def call(policy, op, **fields):
    return json.loads(policy.call_json(json.dumps({"op": op, **fields})))


def place(policy, request_id, now_ms=0):
    return call(
        policy,
        "place",
        now_ms=now_ms,
        input_length=4,
        session_id="conversation",
        request={
            "uuid": str(UUID(int=request_id)),
            "tokens": [1, 2, 3, 4],
            "max_output_tokens": 2,
        },
    )


def test_dispatch_commit_releases_same_session_waiter():
    policy = router()
    first = place(policy, 1)
    assert first["result"]["placement"] is not None
    assert "evidence" not in first
    queued = place(policy, 2)
    assert queued["result"]["placement"] is None
    assert queued["pending_count"] == 1
    call(policy, "dispatch_committed", request_id=str(UUID(int=1)), now_ms=0)
    ready = call(policy, "advance_clock", now_ms=0)
    assert (
        ready["result"][0]["scheduler_id"]
        == first["result"]["placement"]["scheduler_id"]
    )
    assert ready["pending_count"] == 0
    decisions = call(policy, "snapshot")["evidence"]["decisions"]
    assert decisions[1]["binding_reused"] is True


def test_invalid_time_poisoning_crosses_python_boundary():
    policy = router()
    call(policy, "advance_clock", now_ms=1)
    with pytest.raises(ValueError, match="backwards"):
        call(policy, "advance_clock", now_ms=0)
    with pytest.raises(ValueError, match="failed previously"):
        call(policy, "status")


def test_configuration_has_no_simulator_callback_or_endpoint():
    policy = router()
    assert call(policy, "status") == {
        "result": None,
        "pending_count": 0,
        "next_wakeup_ms": None,
    }


def test_idle_ttl_follows_virtual_time_and_keeps_active_sessions():
    policy = router()
    place(policy, 1)
    call(policy, "dispatch_committed", request_id=str(UUID(int=1)), now_ms=0)
    place(policy, 2, now_ms=2000)
    call(policy, "dispatch_committed", request_id=str(UUID(int=2)), now_ms=2000)
    for request_id in (1, 2):
        call(
            policy,
            "request_terminal",
            request_id=str(UUID(int=request_id)),
            now_ms=2000,
        )
    place(policy, 3, now_ms=3001)
    decisions = call(policy, "snapshot")["evidence"]["decisions"]
    assert decisions[1]["binding_reused"] is True
    assert decisions[2]["binding_reused"] is False


def test_abort_releases_uncommitted_session_to_waiter():
    policy = router()
    place(policy, 1)
    place(policy, 2)
    call(policy, "dispatch_aborted", request_id=str(UUID(int=1)), now_ms=0)
    ready = call(policy, "advance_clock", now_ms=0)
    assert ready["result"][0]["request_id"] == str(UUID(int=2))
    evidence = call(policy, "snapshot")["evidence"]
    assert evidence["dispatch_aborts"] == 1
    assert evidence["decisions"][1]["binding_reused"] is False


def test_cancel_removes_native_waiter():
    policy = router()
    place(policy, 1)
    place(policy, 2)
    canceled = call(policy, "cancel_pending", request_id=str(UUID(int=2)))
    assert canceled["result"] is True
    assert canceled["pending_count"] == 0
    call(policy, "dispatch_committed", request_id=str(UUID(int=1)), now_ms=0)
    assert call(policy, "advance_clock", now_ms=0)["result"] == []


def test_physical_cache_events_feed_native_radix_tree():
    from dynamo.llm import compute_block_hash_for_seq

    policy = router()
    first = place(policy, 1)
    worker = first["result"]["placement"]["scheduler_id"]
    call(policy, "dispatch_committed", request_id=str(UUID(int=1)), now_ms=0)
    call(
        policy,
        "observe",
        now_ms=1,
        events=[
            {
                "stage": "aggregated",
                "worker_id": worker,
                "dp_rank": 0,
                "events": [
                    {
                        "event_id": 1,
                        "dp_rank": 0,
                        "data": {
                            "stored": {
                                "parent_hash": None,
                                "session_id": "conversation",
                                "blocks": [
                                    {
                                        "block_hash": 100,
                                        "tokens_hash": compute_block_hash_for_seq(
                                            [1, 2, 3, 4], 4
                                        )[0],
                                    }
                                ],
                            }
                        },
                    }
                ],
            }
        ],
    )
    call(policy, "request_terminal", request_id=str(UUID(int=1)), now_ms=1)
    reused = place(policy, 2, now_ms=2)
    assert reused["result"]["placement"]["reported_overlap_tokens"] == 4
    assert call(policy, "snapshot")["evidence"]["physical_kv_events"] == 1


@pytest.mark.parametrize(
    "command",
    [
        {"op": "typo"},
        {"op": "advance_clock", "now_ms": 0, "ignored": True},
        {
            "op": "worker_ready",
            "now_ms": 0,
            "worker": {"worker_id": 2, "scheduler_ids": [3]},
        },
    ],
)
def test_invalid_commands_fail_closed(command):
    policy = router()
    with pytest.raises(ValueError):
        policy.call_json(json.dumps(command))
    with pytest.raises(ValueError, match="failed previously"):
        call(policy, "snapshot")


def test_sibling_group_requires_conversation_lineage_and_scopes_parent():
    policy = router(mode="sibling_group")
    for request_id, conversation, parent in (
        (1, "child-a", "parent"),
        (2, "child-b", "parent"),
        (3, "other", "other-parent"),
    ):
        call(
            policy,
            "place",
            now_ms=0,
            input_length=4,
            session_id=conversation,
            request={
                "uuid": str(UUID(int=request_id)),
                "tokens": [1, 2, 3, 4],
                "max_output_tokens": 2,
                "replay_context": {
                    "agentic": {
                        "play_id": "play",
                        "conversation_id": conversation,
                        "lineage": {
                            "schema": "aisimulate.agentic.conversation-lineage.v1",
                            "root_conversation_id": "root",
                            "parent_conversation_id": parent,
                        },
                    }
                },
            },
        )
        call(
            policy, "dispatch_committed", request_id=str(UUID(int=request_id)), now_ms=0
        )
    decisions = call(policy, "snapshot")["evidence"]["decisions"]
    assert decisions[0]["group_key"] == decisions[1]["group_key"]
    assert decisions[1]["group_key"] != decisions[2]["group_key"]
    assert decisions[1]["binding_reused"] is True
    assert decisions[2]["binding_reused"] is False

    policy = router(mode="sibling_group")
    with pytest.raises(ValueError, match="unambiguous conversation lineage"):
        place(policy, 1)


def test_idle_affinity_does_not_expire_in_host_wall_time():
    policy = router()
    place(policy, 1)
    call(policy, "dispatch_committed", request_id=str(UUID(int=1)), now_ms=0)
    call(policy, "request_terminal", request_id=str(UUID(int=1)), now_ms=0)
    # Deliberately exceed the configured one-second TTL in host time while the
    # caller's virtual clock stays at zero. This tests clock isolation itself.
    time.sleep(1.05)
    place(policy, 2, now_ms=0)
    assert (
        call(policy, "snapshot")["evidence"]["decisions"][1]["binding_reused"] is True
    )


def test_policy_runs_when_simulator_python_imports_are_blocked():
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys
        class NoSimulator(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"aisimulate", "aiconfigurator", "aiconfigurator_core"}:
                    raise AssertionError("policy attempted simulator import: " + fullname)
        sys.meta_path.insert(0, NoSimulator())
        from dynamo._core import OfflineRouter
        policy = OfflineRouter(json.dumps({
            "schema_version": 1, "role": "aggregated", "dp_size": 1,
            "num_workers": 1, "block_size": 4, "num_gpu_blocks": 128,
        }))
        result = json.loads(policy.call_json(json.dumps({
            "op": "place", "now_ms": 0, "input_length": 4,
            "request": {"uuid": "00000000-0000-0000-0000-000000000001",
                        "tokens": [1, 2, 3, 4], "max_output_tokens": 2},
        })))
        assert result["result"]["placement"]["scheduler_id"] == 0
    """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


def role_router(role, seed):
    return OfflineRouter(
        json.dumps(
            {
                "schema_version": 1,
                "role": role,
                "dp_size": 1,
                "num_workers": 2,
                "block_size": 4,
                "num_gpu_blocks": 128,
                "selector_seed": seed,
            }
        )
    )


def role_place(policy, request_id):
    return call(
        policy,
        "place",
        now_ms=0,
        input_length=64,
        request={
            "uuid": str(UUID(int=request_id)),
            "tokens": [1] * 64,
            "max_output_tokens": 1,
        },
    )["result"]["placement"]["scheduler_id"]


def test_prefill_routing_does_not_reserve_decode_blocks():
    policy = role_router("prefill", 7)
    role_place(policy, 1)
    loads = call(policy, "snapshot")["evidence"]["loads"]
    assert sum(worker["active_blocks"] for worker in loads) == 0
    assert sum(worker["active_prefill_tokens"] for worker in loads) == 64
    call(policy, "prefill_completed", request_id=str(UUID(int=1)), now_ms=0)
    assert (
        sum(
            worker["active_prefill_tokens"]
            for worker in call(policy, "snapshot")["evidence"]["loads"]
        )
        == 0
    )


def test_decode_load_tracks_independent_allocations_for_identical_prompts():
    # Route two equal prompts to one worker. Their decode allocations are
    # independent even though their prefill token identities match.
    policy = OfflineRouter(
        json.dumps(
            {
                "schema_version": 1,
                "role": "decode",
                "dp_size": 1,
                "num_workers": 1,
                "block_size": 4,
                "num_gpu_blocks": 128,
                "selector_seed": 7,
            }
        )
    )
    role_place(policy, 1)
    role_place(policy, 2)
    loads = call(policy, "snapshot")["evidence"]["loads"]
    assert loads[0]["active_blocks"] == 32
    assert loads[0]["active_prefill_tokens"] == 0


def test_native_admission_queue_releases_after_prefill_completion():
    policy = OfflineRouter(
        json.dumps(
            {
                "schema_version": 1,
                "role": "aggregated",
                "dp_size": 1,
                "num_workers": 1,
                "block_size": 4,
                "num_gpu_blocks": 128,
                "max_num_batched_tokens": 64,
                "router_config": {"router_queue_threshold": 0.5},
            }
        )
    )
    role_place(policy, 1)
    queued = call(
        policy,
        "place",
        now_ms=0,
        input_length=64,
        request={
            "uuid": str(UUID(int=2)),
            "tokens": [2] * 64,
            "max_output_tokens": 1,
        },
    )
    assert queued["result"]["placement"] is None
    assert queued["pending_count"] == 1
    released = call(policy, "prefill_completed", request_id=str(UUID(int=1)), now_ms=1)
    assert released["result"][0]["request_id"] == str(UUID(int=2))
    assert released["pending_count"] == 0
