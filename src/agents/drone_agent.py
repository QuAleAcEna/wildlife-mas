from __future__ import annotations

import asyncio
import base64
import datetime as dt
import random
import secrets
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

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
        battery_capacity: float = 100.0,
        battery_consumption_per_step: float = 1.0,
        charge_rate_per_tick: float = 20.0,
        base_position: Tuple[int, int] = (0, 0),
        patrol_waypoint_count: int = 10,
    ):
        super().__init__(jid, password)
        self.ranger_jid = ranger_jid
        self.cruise_speed_mps = cruise_speed_mps
        self.photo_sampler = photo_sampler or _default_sampler
        self.ir_sampler = ir_sampler or _default_sampler
        self.reserve = reserve or Reserve()
        self.patrol_period_s = patrol_period_s
        self.base_position = base_position
        self.patrol_waypoint_count = max(0, patrol_waypoint_count)
        self.max_battery = max(0.0, battery_capacity)
        self.battery_consumption_per_step = max(0.0, battery_consumption_per_step)
        self.charge_rate_per_tick = max(0.0, charge_rate_per_tick)
        self.battery_level = self.max_battery
        self.is_returning_to_base: bool = False
        self.is_charging: bool = False
        self._return_path: Deque[Tuple[int, int]] = deque()
        self._walkable_cells: Set[Tuple[int, int]] = set()
        self._next_battery_log_pct: Optional[float] = (
            90.0 if self.max_battery > 0 else None
        )
        self._planned_patrol_targets: List[Tuple[int, int]] = []
        self._patrol_route: List[Tuple[int, int]] = self._build_patrol_route()
        self._route_index: int = -1
        self.position: Tuple[int, int] = (
            self._next_waypoint() if self._patrol_route else self.base_position
        )

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
            self._walkable_cells = {self.base_position}
            return [self.base_position]

        if self.base_position not in free_cells and not self.reserve.is_no_fly(
            self.base_position
        ):
            free_cells.add(self.base_position)

        ordered_targets = sorted(free_cells, key=lambda cell: (cell[1], cell[0]))
        start_candidate = (
            self.base_position if self.base_position in free_cells else ordered_targets[0]
        )
        reachable = self._reachable_cells(start_candidate, free_cells)
        if not reachable:
            reachable = {start_candidate}
        if self.base_position in reachable:
            start = self.base_position
        else:
            start = next((cell for cell in ordered_targets if cell in reachable), start_candidate)

        self._walkable_cells = set(reachable)

        route: List[Tuple[int, int]] = [start]
        current = start
        planned_targets: List[Tuple[int, int]] = []

        available_targets = [cell for cell in self._walkable_cells if cell != start]
        if available_targets and self.patrol_waypoint_count > 0:
            waypoint_count = min(len(available_targets), self.patrol_waypoint_count)
            random_targets = random.sample(available_targets, waypoint_count)
            for target in random_targets:
                path = self._shortest_path(current, target, self._walkable_cells)
                if not path:
                    continue
                for step in path[1:]:
                    route.append(step)
                current = target
                planned_targets.append(target)

        if current != start:
            back_path = self._shortest_path(current, start, self._walkable_cells)
            if back_path:
                for step in back_path[1:]:
                    route.append(step)

        self._planned_patrol_targets = planned_targets
        return route or [start]

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

    def _current_status(self) -> str:
        if self.is_charging:
            return "charging"
        if self.is_returning_to_base:
            return "returning"
        return "patrolling"

    async def _broadcast_patrol_status(
        self, behaviour: "DroneAgent.PatrolBehaviour"
    ) -> None:
        payload = {
            "drone": str(self.jid),
            "position": {"x": self.position[0], "y": self.position[1]},
            "route_index": self._route_index,
            "route_length": len(self._patrol_route),
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "battery": {
                "level": self.battery_level,
                "capacity": self.max_battery,
                "status": self._current_status(),
            },
        }
        msg = Message(to=self.ranger_jid)
        msg.set_metadata("performative", INFORM)
        msg.set_metadata("type", TELEMETRY)
        msg.body = json_dumps(payload)
        await behaviour.send(msg)

    def log(self, *args: Any) -> None:
        print("[DRONE]", *args)

    def _estimate_energy_cost(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        if start == goal:
            return 0.0
        walkable = self._walkable_cells or {start}
        path = self._shortest_path(start, goal, walkable)
        if not path:
            return float("inf")
        steps = max(0, len(path) - 1)
        return steps * self.battery_consumption_per_step

    def _consume_battery_for_move(
        self, previous: Tuple[int, int], current: Tuple[int, int]
    ) -> None:
        if previous == current or self.battery_consumption_per_step <= 0:
            return
        before_level = self.battery_level
        self.battery_level = max(
            0.0, self.battery_level - self.battery_consumption_per_step
        )
        self._log_battery_drop_if_needed(before_level)

    def _log_battery_drop_if_needed(self, previous_level: float) -> None:
        if (
            self._next_battery_log_pct is None
            or self.max_battery <= 0
            or previous_level <= self.battery_level
        ):
            return
        previous_pct = (previous_level / self.max_battery) * 100.0
        current_pct = (self.battery_level / self.max_battery) * 100.0
        while (
            self._next_battery_log_pct is not None
            and previous_pct > self._next_battery_log_pct >= current_pct
        ):
            threshold = self._next_battery_log_pct
            self.log(
                f"Battery at {threshold:.0f}% "
                f"({self.battery_level:.1f}/{self.max_battery:.1f})."
            )
            self._next_battery_log_pct -= 10.0
            if self._next_battery_log_pct < 0:
                self._next_battery_log_pct = None
                break

    def _should_return_to_base(self) -> bool:
        if self.is_returning_to_base or self.is_charging:
            return False
        energy_to_base = self._estimate_energy_cost(self.position, self.base_position)
        if energy_to_base == float("inf"):
            return False
        threshold = energy_to_base + self.battery_consumption_per_step
        return self.battery_level <= threshold

    def _begin_return_to_base(self) -> None:
        walkable = self._walkable_cells or {self.position}
        path = self._shortest_path(self.position, self.base_position, walkable)
        if not path:
            self.log(
                "Unable to compute a path back to base from", self.position
            )
            return
        if len(path) <= 1:
            self.is_returning_to_base = False
            self.is_charging = True
            self.log("Already at base, starting recharge.")
            return
        self._return_path = deque(path[1:])
        self.is_returning_to_base = True
        self.log(
            "Battery low. Returning to base via",
            len(self._return_path),
            "steps.",
        )

    def _advance_return_path(self, previous_position: Tuple[int, int]) -> None:
        if not self._return_path:
            self.is_returning_to_base = False
            self.is_charging = True
            self.position = self.base_position
            self.log("Arrived at base, starting recharge.")
            return

        next_position = self._return_path.popleft()
        self.position = next_position
        self._consume_battery_for_move(previous_position, next_position)
        self.log(
            "Returning to base",
            self.position,
            f"{len(self._return_path)} steps remaining.",
        )
        if not self._return_path:
            self.is_returning_to_base = False
            self.is_charging = True
            self.position = self.base_position
            self.log("Arrived at base, starting recharge.")

    def _perform_charging(self) -> None:
        if self.battery_level >= self.max_battery:
            self._finish_charging()
            return
        if self.charge_rate_per_tick <= 0:
            self.battery_level = self.max_battery
        else:
            self.battery_level = min(
                self.max_battery, self.battery_level + self.charge_rate_per_tick
            )
        if self.battery_level >= self.max_battery:
            self._finish_charging()

    def _finish_charging(self) -> None:
        self.battery_level = self.max_battery
        self.is_charging = False
        self._route_index = -1
        self._patrol_route = self._build_patrol_route()
        self.position = self.base_position
        self.log("Battery full. Resuming patrol route.")
        self._next_battery_log_pct = 90.0 if self.max_battery > 0 else None

    async def _tick_patrol(self, behaviour: "DroneAgent.PatrolBehaviour") -> None:
        previous_position = self.position

        if not self.is_charging and self._should_return_to_base():
            self._begin_return_to_base()

        if self.is_charging:
            self._perform_charging()
        elif self.is_returning_to_base:
            self._advance_return_path(previous_position)
        else:
            next_position = self._next_waypoint()
            if (
                previous_position == self.base_position
                and next_position != self.base_position
                and self._planned_patrol_targets
            ):
                targets_str = ", ".join(
                    f"({x},{y})" for x, y in self._planned_patrol_targets
                )
                self.log("Departing base to cover:", targets_str)
            self.position = next_position
            self._consume_battery_for_move(previous_position, next_position)
            self.log(
                "Patrolling waypoint",
                self.position,
                f"[{self._route_index + 1}/{len(self._patrol_route)}]",
            )

        await self._broadcast_patrol_status(behaviour)

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
            await self.drone._tick_patrol(self)
