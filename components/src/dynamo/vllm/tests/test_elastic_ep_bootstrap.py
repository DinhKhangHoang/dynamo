# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from unittest.mock import Mock

import pytest

from dynamo.vllm import elastic_ep_bootstrap as bootstrap

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def assignment():
    return {
        "version": "vllm-launch/v1",
        "groupName": "world",
        "groupUID": "group-uid",
        "profileFingerprint": "profile",
        "podUID": "pod-uid",
        "rank": 2,
        "dataParallelSize": 3,
        "mode": "join",
        "controlRevision": 4,
        "transitionID": "grow-three",
        "baseTopologyGeneration": 1,
        "command": ["vllm", "serve", "model", "--data-parallel-size", "3"],
        "environment": {
            "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_USE_RUST_FRONTEND": "0",
        },
    }


@pytest.fixture
def launch_environment(monkeypatch):
    environment = {
        "DYNAMO_VLLM_LAUNCH_GROUP": "world",
        "DYNAMO_VLLM_LAUNCH_PROFILE": "profile",
        "DYNAMO_VLLM_LAUNCH_POD_UID": "pod-uid",
        "DYNAMO_VLLM_LAUNCH_SLOT": "2",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return environment


def test_launch_executes_native_engine_once(
    assignment, launch_environment, monkeypatch, tmp_path
):
    assignment_path = tmp_path / "assignment"
    assignment_path.write_text(json.dumps(assignment))
    receipt_path = tmp_path / "state" / "receipt"
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execvpe", execute)
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0")
    before = copy.deepcopy(assignment)

    bootstrap.launch(assignment_path, receipt_path, 1)

    execute.assert_called_once()
    executable, command, environment = execute.call_args.args
    assert executable == "vllm"
    assert command == tuple(assignment["command"])
    assert environment["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] == "1"
    assert assignment == before
    bootstrap.check_launch(assignment_path, receipt_path)
    with pytest.raises(RuntimeError, match="already launched"):
        bootstrap.launch(assignment_path, receipt_path, 1)
    execute.assert_called_once()


def test_launch_binds_the_existing_dynamo_worker_to_its_assigned_group(
    assignment, launch_environment, monkeypatch, tmp_path
):
    assignment["command"] = ["python3", "-m", "dynamo.vllm", "--model=model"]
    assignment_path = tmp_path / "assignment"
    assignment_path.write_text(json.dumps(assignment))
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execvpe", execute)

    bootstrap.launch(assignment_path, tmp_path / "receipt", 1)

    executable, command, environment = execute.call_args.args
    assert executable == "python3"
    assert command == tuple(assignment["command"])
    assert environment["DYNAMO_ENGINE_GROUP_ID"] == assignment["groupUID"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "future-version"),
        ("podUID", "old-pod-uid"),
        ("groupName", "another-group"),
        ("groupUID", ""),
        ("profileFingerprint", "another-profile"),
        ("rank", 1),
        ("rank", True),
        ("dataParallelSize", 2),
        ("controlRevision", 0),
        ("baseTopologyGeneration", 0),
        ("transitionID", ""),
        ("mode", "recover"),
        ("command", ["sh", "-c", "vllm serve model"]),
        ("command", ["python3", "-m", "dynamo.frontend"]),
        ("command", ["vllm", "serve", "model\0"]),
        ("command", "vllm serve model"),
        ("environment", {"VLLM_USE_RUST_FRONTEND": "1"}),
    ],
)
def test_launch_rejects_stale_or_malformed_assignments(
    assignment, launch_environment, field, value
):
    assignment[field] = value
    with pytest.raises(ValueError):
        bootstrap.validate_assignment(assignment, launch_environment)


def test_launch_initial_world_has_no_synthetic_operation(
    assignment, launch_environment
):
    assignment.update(
        mode="initial", controlRevision=0, transitionID="", baseTopologyGeneration=0
    )
    assignment["environment"]["VLLM_ELASTIC_EP_SCALE_UP_LAUNCH"] = "0"

    bootstrap.validate_assignment(assignment, launch_environment)

    assignment["transitionID"] = "synthetic-grow"
    with pytest.raises(ValueError, match="initial formation"):
        bootstrap.validate_assignment(assignment, launch_environment)


def test_changed_assignment_invalidates_launch_readiness(
    assignment, launch_environment, monkeypatch, tmp_path
):
    assignment_path = tmp_path / "assignment"
    assignment_path.write_text(json.dumps(assignment))
    receipt_path = tmp_path / "receipt"
    monkeypatch.setattr(bootstrap.os, "execvpe", Mock())
    bootstrap.launch(assignment_path, receipt_path, 1)

    assignment["transitionID"] = "another-operation"
    assignment_path.write_text(json.dumps(assignment))

    with pytest.raises(RuntimeError, match="does not match"):
        bootstrap.check_launch(assignment_path, receipt_path)


@pytest.mark.timeout(5)
def test_waits_only_for_unpublished_assignment(
    assignment, launch_environment, monkeypatch, tmp_path
):
    native = bootstrap.NativeLaunch(
        tuple(assignment["command"]),
        dict(assignment["environment"]),
        assignment["groupUID"],
    )
    read = Mock(side_effect=[FileNotFoundError(), (native, "digest")])
    sleep = Mock()
    monkeypatch.setattr(bootstrap, "read_assignment", read)
    monkeypatch.setattr(bootstrap.time, "sleep", sleep)

    observed, digest = bootstrap.wait_for_assignment(
        tmp_path / "assignment", launch_environment, 1
    )

    assert observed == native
    assert digest == "digest"
    assert read.call_count == 2
    sleep.assert_called_once()


@pytest.mark.timeout(5)
def test_missing_assignment_times_out_without_engine_launch(
    launch_environment, monkeypatch, tmp_path
):
    monkeypatch.setattr(bootstrap.time, "monotonic", Mock(side_effect=[0, 2]))
    execute = Mock()
    monkeypatch.setattr(bootstrap.os, "execvpe", execute)

    with pytest.raises(TimeoutError, match="launch assignment"):
        bootstrap.launch(tmp_path / "missing", tmp_path / "receipt", 1)

    execute.assert_not_called()


def test_missing_or_unconsumed_intent_is_not_ready(
    assignment, launch_environment, tmp_path
):
    assignment_path = tmp_path / "assignment"
    receipt_path = tmp_path / "receipt"
    with pytest.raises(FileNotFoundError):
        bootstrap.check_launch(assignment_path, receipt_path)
    assignment_path.write_text(json.dumps(assignment))
    with pytest.raises(FileNotFoundError):
        bootstrap.check_launch(assignment_path, receipt_path)
