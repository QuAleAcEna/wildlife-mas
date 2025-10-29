from __future__ import annotations

import asyncio
import base64
import datetime as dt
import secrets
from typing import Any, Callable, Dict, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from core.messages import (
    INFORM,
    TELEMETRY,
    json_dumps,
    json_loads,
    make_inform_alert,
)

_DEFAULT_SAMPLE_SIZE = 512


def _default_sampler(_: Dict[str, Any], size: int = _DEFAULT_SAMPLE_SIZE) -> bytes:
    return secrets.token_bytes(size)


class DroneAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        ranger_jid: str,
        cruise_speed_mps: float = 22.0,
        photo_sampler: Optional[Callable[[Dict[str, Any]], bytes]] = None,
        ir_sampler: Optional[Callable[[Dict[str, Any]], bytes]] = None,
    ):
        super().__init__(jid, password)
        self.ranger_jid = ranger_jid
        self.cruise_speed_mps = cruise_speed_mps
        self.photo_sampler = photo_sampler or _default_sampler
        self.ir_sampler = ir_sampler or _default_sampler

    async def setup(self) -> None:
        self.add_behaviour(self.AlertRelayBehaviour(self))

    async def handle_sensor_alert(self, msg: Message) -> None:
        sensor = str(msg.sender)
        payload = self._safe_load(msg.body)
        ack_payload = self._build_ack_payload(sensor, payload)
        attachments = await self._collect_attachments(payload)
        await self._reply_to_sensor(sensor, ack_payload)
        await self._notify_ranger(sensor, payload, ack_payload, attachments)

    def _safe_load(self, body: Optional[str]) -> Dict[str, Any]:
        if not body:
            return {}
        try:
            return json_loads(body)
        except Exception:
            return {"raw_body": body}

    def _build_ack_payload(self, sensor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        distance = float(payload.get("distance_m", 0.0))
        eta = self._estimate_eta(distance)
        return {
            "sensor": sensor,
            "drone": str(self.jid),
            "alert_id": payload.get("id"),
            "received_at": dt.datetime.utcnow().isoformat() + "Z",
            "estimated_arrival_s": eta,
        }

    def _estimate_eta(self, distance_m: float) -> float:
        if self.cruise_speed_mps <= 0:
            return 0.0
        return max(0.0, distance_m / self.cruise_speed_mps)

    async def _collect_attachments(self, payload: Dict[str, Any]) -> Dict[str, str]:
        photo_task = asyncio.create_task(
            asyncio.to_thread(self.photo_sampler, payload)
        )
        ir_task = asyncio.create_task(asyncio.to_thread(self.ir_sampler, payload))
        photo_bytes, ir_bytes = await asyncio.gather(photo_task, ir_task)
        return {
            "photo_base64": base64.b64encode(photo_bytes).decode("ascii"),
            "ir_base64": base64.b64encode(ir_bytes).decode("ascii"),
        }

    async def _reply_to_sensor(self, sensor: str, ack_payload: Dict[str, Any]) -> None:
        reply = Message(to=sensor)
        reply.set_metadata("performative", INFORM)
        reply.set_metadata("type", TELEMETRY)
        reply.body = json_dumps(ack_payload)
        await self.send(reply)

    async def _notify_ranger(
        self,
        sensor: str,
        original_payload: Dict[str, Any],
        ack_payload: Dict[str, Any],
        attachments: Dict[str, str],
    ) -> None:
        ranger_payload = {
            "sensor": sensor,
            "drone": str(self.jid),
            "ack": ack_payload,
            "alert": original_payload,
            "attachments": attachments,
        }
        msg = make_inform_alert(self.ranger_jid, ranger_payload)
        await self.send(msg)

    class AlertRelayBehaviour(CyclicBehaviour):
        def __init__(self, drone: "DroneAgent") -> None:
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            msg = await self.receive(timeout=0.1)
            if not msg:
                return
            await self.drone.handle_sensor_alert(msg)
