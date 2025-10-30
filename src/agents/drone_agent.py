from __future__ import annotations

import asyncio
import base64
import datetime as dt
import secrets
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
        free_cells: Set[Tuple[int, int]] = {
            (x, y)
            for y in range(height)
            for x in range(width)
            if not self.reserve.is_no_fly((x, y))
        }
        if not free_cells:
            return [(0, 0)]

        ordered_targets = sorted(free_cells, key=lambda cell: (cell[1], cell[0]))
        start = ordered_targets[0]
        reachable = self._reachable_cells(start, free_cells)
        ordered_targets = [cell for cell in ordered_targets if cell in reachable]
        route: List[Tuple[int, int]] = [start]
        visited: Set[Tuple[int, int]] = {start}
        current = start

        for target in ordered_targets[1:]:
            if target in visited:
                continue
            path = self._shortest_path(current, target, reachable)
            if not path:
                continue
            for step in path[1:]:
                route.append(step)
                visited.add(step)
            current = path[-1]

        return route

    def _shortest_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return [start]

        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            cell, path = queue.popleft()
            for neighbor in self._neighbors(cell):
                if neighbor in seen or neighbor not in walkable:
                    continue
                if neighbor == goal:
                    return path + [neighbor]
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return None

    def _reachable_cells(
        self,
        start: Tuple[int, int],
        free_cells: Set[Tuple[int, int]],
    ) -> Set[Tuple[int, int]]:
        queue = deque([start])
        reachable = {start}
        while queue:
            cell = queue.popleft()
            for neighbor in self._neighbors(cell):
                if neighbor in reachable or neighbor not in free_cells:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)
        return reachable

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = cell
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        neighbors: List[Tuple[int, int]] = []
        for nx, ny in candidates:
            if 0 <= nx < max(1, self.reserve.width) and 0 <= ny < max(
                1, self.reserve.height
            ):
                neighbors.append((nx, ny))
        return neighbors

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
