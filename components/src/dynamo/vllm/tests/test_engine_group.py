# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm.engine_group import EngineGroupServing, RegistrationError

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@dataclass
class NativeStatus:
    instance_id: str = "engine-0"
    operation_id: str | None = None
    epoch: str | None = None
    phase: str = "idle"
    requested_data_parallel_size: int | None = None
    effective_data_parallel_size: int = 2
    topology_known: bool = True
    error: str | None = None


@pytest.fixture
def worker():
    engine = AsyncMock()
    engine.get_external_elastic_ep_status.return_value = NativeStatus()
    engine.is_paused.return_value = False
    endpoint = Mock(
        register_endpoint_instance=AsyncMock(),
        unregister_endpoint_instance=AsyncMock(),
        connection_id=Mock(return_value=7),
    )
    serving = EngineGroupServing(
        engine,
        endpoint,
        "group",
        "pod-0",
        0,
        Mock(instance_ids=Mock(return_value=[])),
        Mock(return_value=True),
        AsyncMock(return_value=True),
    )
    serving.is_ready = True
    return serving


@pytest.mark.asyncio
async def test_registration_and_growth_need_no_controller_permission(worker):
    await worker.refresh()
    assert worker.is_admitted
    assert worker.status()["verified"]
    worker.verify.assert_awaited_once()
    worker.ensure_serving()

    worker.engine.get_external_elastic_ep_status.return_value = NativeStatus(
        phase="completed",
        operation_id="grow",
        epoch="epoch-2",
        effective_data_parallel_size=3,
    )
    await worker.refresh()
    assert worker.is_admitted
    assert worker.binding.data_parallel_size == 3
    assert worker.verify.await_count == 2
    assert worker.endpoint.register_endpoint_instance.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_source", ["worker", "engine"])
async def test_preparation_keeps_base_serving_but_actual_pause_withdraws(
    worker, pause_source
):
    await worker.refresh()
    worker.discovery.instance_ids.return_value = [7]
    base = worker.verified
    worker.engine.get_external_elastic_ep_status.return_value = NativeStatus(
        phase="preparing",
        operation_id="grow",
        epoch="next-epoch",
        requested_data_parallel_size=3,
    )
    await worker.refresh()
    assert worker.binding == base
    assert worker.is_admitted
    worker.verify.assert_awaited_once()
    worker.endpoint.unregister_endpoint_instance.assert_not_awaited()

    if pause_source == "worker":
        worker.can_serve.return_value = False
    else:
        worker.engine.is_paused.return_value = True
    await worker.refresh()
    assert not worker.is_admitted
    assert not worker.status()["verified"]
    worker.endpoint.unregister_endpoint_instance.assert_awaited_once()
    with pytest.raises(ConnectionError, match="not inference-ready"):
        worker.ensure_serving()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unknown", "committing", "retired"])
async def test_unusable_membership_cannot_register(worker, change):
    status = worker.engine.get_external_elastic_ep_status.return_value
    if change == "unknown":
        status.topology_known = False
    elif change == "committing":
        status.phase = "committing"
    else:
        worker.rank = 2
    await worker.refresh()
    assert not worker.is_admitted
    worker.verify.assert_not_awaited()
    worker.endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_restart_requires_fresh_proof_not_controller_ack(worker):
    await worker.refresh()
    restarted = EngineGroupServing(
        worker.engine,
        worker.endpoint,
        "group",
        "pod-0",
        0,
        worker.discovery,
        worker.can_serve,
        AsyncMock(return_value=True),
    )
    restarted.is_ready = True
    assert restarted.verified is None
    await restarted.refresh()
    assert restarted.is_admitted
    restarted.verify.assert_awaited_once()


@pytest.mark.asyncio
async def test_topology_change_during_verification_discards_the_result(worker):
    worker.engine.get_external_elastic_ep_status.side_effect = [
        NativeStatus(),
        NativeStatus(topology_known=False),
    ]
    await worker.refresh()
    assert worker.verified is None
    assert not worker.is_admitted
    worker.endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_failure_and_drift_are_retried_without_a_target(worker):
    worker.endpoint.register_endpoint_instance.side_effect = TimeoutError(
        "discovery timeout"
    )
    with pytest.raises(RegistrationError):
        await worker.refresh()
    assert not worker.is_admitted
    worker.endpoint.register_endpoint_instance.side_effect = None
    await worker.refresh()
    assert worker.is_admitted

    worker.discovery.instance_ids.return_value = [7]
    calls = worker.endpoint.register_endpoint_instance.await_count
    await worker.refresh()
    assert worker.endpoint.register_endpoint_instance.await_count == calls
    worker.discovery.instance_ids.return_value = []
    await worker.refresh()
    assert worker.endpoint.register_endpoint_instance.await_count == calls + 1


@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_late_registration_cannot_override_native_pause(worker):
    entered, release = asyncio.Event(), asyncio.Event()

    async def register():
        entered.set()
        await release.wait()

    worker.endpoint.register_endpoint_instance.side_effect = register
    registering = asyncio.create_task(worker.refresh())
    try:
        await entered.wait()
        worker.can_serve.return_value = False
        status = await asyncio.wait_for(worker.observe({}), timeout=1)
        assert not status["admitted"]
        release.set()
        await registering
        assert not worker.is_admitted
        worker.endpoint.unregister_endpoint_instance.assert_awaited_once()
    finally:
        registering.cancel()
        await asyncio.gather(registering, return_exceptions=True)
