# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume one Pod-bound launch assignment, then exec the assigned vLLM worker.

This is a launch-only entrypoint, not a worker, proxy, supervisor, or membership
controller. Its readiness check proves assignment consumption only. A worker
must separately enforce engine health, committed membership, serving verification,
and Engine Group admission before exposing a serving endpoint.
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

_VERSION = "vllm-launch/v1"
_ASSIGNMENT_PATH = Path("/var/run/dynamo/vllm-launch/assignment")
_RECEIPT_PATH = Path("/var/run/dynamo/vllm-launch-state/receipt")
_FIELDS = frozenset(
    {
        "version",
        "groupName",
        "groupUID",
        "profileFingerprint",
        "podUID",
        "rank",
        "dataParallelSize",
        "mode",
        "controlRevision",
        "transitionID",
        "baseTopologyGeneration",
        "command",
        "environment",
    }
)


@dataclass(frozen=True)
class NativeLaunch:
    """Executable fields retained after validating the wire document's identities."""

    command: tuple[str, ...]
    environment: dict[str, str]
    group_uid: str


def validate_assignment(document: dict, environment: dict[str, str]) -> None:
    """Reject stale identities or malformed private launch documents before exec."""
    if not isinstance(document, dict) or document.keys() != _FIELDS:
        raise ValueError("invalid vLLM launch assignment fields")
    if document["version"] != _VERSION:
        raise ValueError("unsupported vLLM launch assignment version")
    for field, variable in (
        ("groupName", "DYNAMO_VLLM_LAUNCH_GROUP"),
        ("profileFingerprint", "DYNAMO_VLLM_LAUNCH_PROFILE"),
        ("podUID", "DYNAMO_VLLM_LAUNCH_POD_UID"),
    ):
        expected = environment[variable]
        if not expected or document[field] != expected:
            raise ValueError(f"vLLM launch {field} does not match this Pod")
    if not isinstance(document["groupUID"], str) or not document["groupUID"]:
        raise ValueError("vLLM launch requires its Engine Group UID")
    for field in (
        "rank",
        "dataParallelSize",
        "controlRevision",
        "baseTopologyGeneration",
    ):
        if type(document[field]) is not int:
            raise ValueError(f"vLLM launch {field} must be an integer")
    rank = document["rank"]
    if (
        rank < 0
        or document["dataParallelSize"] <= 1
        or rank >= document["dataParallelSize"]
        or str(rank) != environment["DYNAMO_VLLM_LAUNCH_SLOT"]
    ):
        raise ValueError("vLLM launch rank/size does not match this Pod slot")
    if not isinstance(document["transitionID"], str):
        raise TypeError("vLLM launch transitionID must be a string")
    if document["mode"] == "initial":
        if (
            document["controlRevision"] != 0
            or document["baseTopologyGeneration"] != 0
            or document["transitionID"]
        ):
            raise ValueError("initial formation must not carry a resize operation")
    elif document["mode"] == "join":
        if (
            document["controlRevision"] <= 0
            or document["baseTopologyGeneration"] <= 0
            or not document["transitionID"]
        ):
            raise ValueError("joining launch requires exact resize correlation")
    else:
        raise ValueError("unsupported vLLM launch mode")

    command = document["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(
            not isinstance(token, str) or not token or "\0" in token
            for token in command
        )
    ):
        raise ValueError("vLLM launch requires a native exec-form command")
    executable = Path(command[0]).name
    if not (
        (executable == "vllm" and command[1:2] == ["serve"])
        or (
            executable in {"python", "python3"}
            and command[1:3]
            in (
                ["-m", "vllm.entrypoints.openai.api_server"],
                ["-m", "dynamo.vllm"],
            )
        )
    ):
        raise ValueError("vLLM launch requires the native server or Dynamo worker")
    expected_overrides = {
        "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1" if document["mode"] == "join" else "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_USE_RUST_FRONTEND": "0",
    }
    if document["environment"] != expected_overrides:
        raise ValueError(
            "vLLM launch environment does not match its native launch mode"
        )


def read_assignment(
    path: Path, environment: dict[str, str]
) -> tuple[NativeLaunch, str]:
    """Read one atomic Downward API projection; empty means not published yet."""
    payload = path.read_text()
    if not payload:
        raise FileNotFoundError("vLLM launch assignment has not been published")
    document = json.loads(payload)
    validate_assignment(document, environment)
    native = NativeLaunch(
        tuple(document["command"]), dict(document["environment"]), document["groupUID"]
    )
    return native, hashlib.sha256(payload.encode()).hexdigest()


def wait_for_assignment(
    path: Path, environment: dict[str, str], timeout: float
) -> tuple[NativeLaunch, str]:
    """Wait only for unpublished metadata, never for inference or registration."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("vLLM bootstrap timeout must be positive")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return read_assignment(path, environment)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the vLLM launch assignment")
            time.sleep(0.25)


def launch(assignment_path: Path, receipt_path: Path, timeout: float) -> None:
    """Replace the launcher with native vLLM exactly once in this Pod lifetime."""
    if receipt_path.exists():
        raise RuntimeError(
            "vLLM was already launched in this Pod; recovery is required"
        )
    environment = dict(os.environ)
    native, digest = wait_for_assignment(assignment_path, environment, timeout)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as receipt:
        receipt.write(digest)
    environment.update(native.environment)
    if native.command[1:3] == ("-m", "dynamo.vllm"):
        environment["DYNAMO_ENGINE_GROUP_ID"] = native.group_uid
    os.execvpe(native.command[0], native.command, environment)


def check_launch(assignment_path: Path, receipt_path: Path) -> None:
    """Check consumed launch intent, not native health, membership, or admission."""
    _, digest = read_assignment(assignment_path, dict(os.environ))
    if receipt_path.read_text() != digest:
        raise RuntimeError("vLLM launch receipt does not match the current assignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--assignment", type=Path, default=_ASSIGNMENT_PATH)
    parser.add_argument("--receipt", type=Path, default=_RECEIPT_PATH)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.check:
        check_launch(args.assignment, args.receipt)
    else:
        launch(args.assignment, args.receipt, args.timeout)


if __name__ == "__main__":
    main()
