from __future__ import annotations

import asyncio
import base64
import datetime as dt
import secrets
from typing import Any, Callable, Dict, List, Optional, Tuple

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from spade.message import Message

from core.messages import (
    ALERT_ANOMALY,
    INFORM,
    TELEMETRY,
    json_dumps,
    json_loads,
    make_inform_alert,
)
from core.env import Reserve

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
        reserve: Optional[Reserve] = None,
        patrol_period_s: float = 5.0,
    ):
        super().__init__(jid, password)
        self.ranger_jid = ranger_jid
        self.cruise_speed_mps = cruise_speed_mps
        self.photo_sampler = photo_sampler or _default_sampler
        self.ir_sampler = ir_sampler or _default_sampler
        self.reserve = reserve or Reserve()
        self.patrol_period_s = patrol_period_s
        self._patrol_route: List[Tuple[int, int]] = self._build_patrol_route()
        self._route_index: int = -1
        self.position: Tuple[int, int] = self._next_waypoint()

    async def setup(self) -> None:
        self.add_behaviour(self.AlertRelayBehaviour(self))
        self.add_behaviour(
            self.PatrolBehaviour(self, period=self.patrol_period_s)
        )

    async def handle_sensor_alert(
        self, behaviour: "DroneAgent.AlertRelayBehaviour", msg: Message
    ) -> None:
        if msg.get_metadata("type") != ALERT_ANOMALY:
            return
        sensor = str(msg.sender)
        payload = self._safe_load(msg.body)
        ack_payload = self._build_ack_payload(sensor, payload)
        attachments = await self._collect_attachments(payload)
        await self._reply_to_sensor(behaviour, sensor, ack_payload)
        await self._notify_ranger(
            behaviour, sensor, payload, ack_payload, attachments
        )

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

    def _build_patrol_route(self) -> List[Tuple[int, int]]:
        width = max(1, self.reserve.width)
        height = max(1, self.reserve.height)
        route: List[Tuple[int, int]] = []
        for y in range(height):
            xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
            for x in xs:
                cell = (x, y)
                if self.reserve.is_no_fly(cell):
                    continue
                route.append(cell)
        if not route:
            route = [(0, 0)]
        return route

    def _next_waypoint(self) -> Tuple[int, int]:
        if not self._patrol_route:
            return (0, 0)
        self._route_index = (self._route_index + 1) % len(self._patrol_route)
        return self._patrol_route[self._route_index]

    async def _broadcast_patrol_status(
        self, behaviour: "DroneAgent.PatrolBehaviour"
    ) -> None:
        payload = {
            "drone": str(self.jid),
            "position": {"x": self.position[0], "y": self.position[1]},
            "route_index": self._route_index,
            "route_length": len(self._patrol_route),
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        }
        msg = Message(to=self.ranger_jid)
        msg.set_metadata("performative", INFORM)
        msg.set_metadata("type", TELEMETRY)
        msg.body = json_dumps(payload)
        await behaviour.send(msg)

    def log(self, *args: Any) -> None:
        print("[DRONE]", *args)

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

    async def _reply_to_sensor(
        self,
        behaviour: "DroneAgent.AlertRelayBehaviour",
        sensor: str,
        ack_payload: Dict[str, Any],
    ) -> None:
        reply = Message(to=sensor)
        reply.set_metadata("performative", INFORM)
        reply.set_metadata("type", TELEMETRY)
        reply.body = json_dumps(ack_payload)
        await behaviour.send(reply)

    async def _notify_ranger(
        self,
        behaviour: "DroneAgent.AlertRelayBehaviour",
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
        await behaviour.send(msg)

    class AlertRelayBehaviour(CyclicBehaviour):
        def __init__(self, drone: "DroneAgent") -> None:
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            msg = await self.receive(timeout=0.1)
            if not msg:
                return
            if msg.get_metadata("type") != ALERT_ANOMALY:
                self.drone.log(
                    "Ignoring non-alert message from", str(msg.sender)
                )
                return
            await self.drone.handle_sensor_alert(self, msg)
    class PatrolBehaviour(PeriodicBehaviour):
        def __init__(self, drone: "DroneAgent", period: float) -> None:
            super().__init__(period=period)
            self.drone = drone

        async def run(self) -> None:
            self.drone.position = self.drone._next_waypoint()
            self.drone.log(
                "Patrolling waypoint",
                self.drone.position,
                f"[{self.drone._route_index + 1}/{len(self.drone._patrol_route)}]",
            )
            await self.drone._broadcast_patrol_status(self)
