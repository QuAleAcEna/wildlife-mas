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
    """Fallback sampler that emits pseudo-random bytes for attachments."""
    return secrets.token_bytes(size)


class DroneAgent(Agent):
    """Autonomous drone that patrols the reserve and assists ranger dispatches."""
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
        patrol_sector: Optional[Tuple[int, int, int, int]] = None,
        patrol_seed: Optional[int] = None,
        callsign: Optional[str] = None,
    ):
        """Configure the drone's patrol parameters, sensors, and battery model."""
        super().__init__(jid, password)
        self.ranger_jid = ranger_jid
        self.cruise_speed_mps = cruise_speed_mps
        self.photo_sampler = photo_sampler or _default_sampler
        self.ir_sampler = ir_sampler or _default_sampler
        self.reserve = reserve or Reserve()
        self.patrol_period_s = patrol_period_s
        self.base_position = base_position
        self.patrol_waypoint_count = max(0, patrol_waypoint_count)
        self._sector_bounds = self._normalize_sector(patrol_sector)
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
        seed = patrol_seed if patrol_seed is not None else secrets.randbits(64)
        self._patrol_rng = random.Random(seed)
        self.patrol_seed = seed
        self.callsign = (callsign or str(self.jid)).upper()
        self._planned_patrol_targets: List[Tuple[int, int]] = []
        self._patrol_route: List[Tuple[int, int]] = self._build_patrol_route()
        self._route_index: int = -1
        self.position: Tuple[int, int] = (
            self._next_waypoint() if self._patrol_route else self.base_position
        )

    async def setup(self) -> None:
        """Start behaviours for relaying alerts and patrolling the reserve."""
        self.add_behaviour(self.AlertRelayBehaviour(self))
        self.add_behaviour(
            self.PatrolBehaviour(self, period=self.patrol_period_s)
        )

    async def handle_sensor_alert(
        self, behaviour: "DroneAgent.AlertRelayBehaviour", msg: Message
    ) -> None:
        """Translate sensor anomaly alerts into ranger notifications with evidence."""
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
        """Protect against malformed JSON messages by falling back to raw data."""
        if not body:
            return {}
        try:
            return json_loads(body)
        except Exception:
            return {"raw_body": body}

    def _build_ack_payload(self, sensor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compose a telemetry acknowledgment back to the originating sensor."""
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
        """Convert a distance in meters into seconds at the current cruise speed."""
        if self.cruise_speed_mps <= 0:
            return 0.0
        return max(0.0, distance_m / self.cruise_speed_mps)

    def _normalize_sector(
        self, sector: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Clamp a patrol sector to reserve bounds and validate coordinates."""
        if sector is None:
            return None
        x_min, y_min, x_max, y_max = sector
        x_min = max(0, min(self.reserve.width - 1, x_min))
        y_min = max(0, min(self.reserve.height - 1, y_min))
        x_max = max(0, min(self.reserve.width - 1, x_max))
        y_max = max(0, min(self.reserve.height - 1, y_max))
        if x_min > x_max or y_min > y_max:
            raise ValueError(
                f"Invalid patrol sector bounds after clamping: {(x_min, y_min, x_max, y_max)}"
            )
        return (x_min, y_min, x_max, y_max)

    def _within_sector(self, cell: Tuple[int, int]) -> bool:
        """Return True if the coordinate falls inside the configured patrol sector."""
        if not self._sector_bounds:
            return True
        x_min, y_min, x_max, y_max = self._sector_bounds
        return x_min <= cell[0] <= x_max and y_min <= cell[1] <= y_max

    def _build_patrol_route(self) -> List[Tuple[int, int]]:
        """Generate a patrol loop that avoids no-fly zones and respects patrol sectors."""
        if self._sector_bounds and not self._within_sector(self.base_position):
            self.log(
                "Base position",
                self.base_position,
                "is outside configured patrol sector",
                self._sector_bounds,
            )
        width = max(1, self.reserve.width)
        height = max(1, self.reserve.height)
        free_cells: Set[Tuple[int, int]] = {
            (x, y)
            for y in range(height)
            for x in range(width)
            if not self.reserve.is_no_fly((x, y))
            and self._within_sector((x, y))
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
            # Choose the first reachable cell to seed the patrol if base is isolated.
            start = next((cell for cell in ordered_targets if cell in reachable), start_candidate)

        self._walkable_cells = set(reachable)

        route: List[Tuple[int, int]] = [start]
        current = start
        planned_targets: List[Tuple[int, int]] = []

        available_targets = [cell for cell in self._walkable_cells if cell != start]
        if available_targets and self.patrol_waypoint_count > 0:
            waypoint_count = min(len(available_targets), self.patrol_waypoint_count)
            random_targets = self._patrol_rng.sample(available_targets, waypoint_count)
            for target in random_targets:
                path = self._shortest_path(current, target, self._walkable_cells)
                if not path:
                    # If the BFS fails the tile is effectively isolated; ignore it.
                    continue
                for step in path[1:]:
                    route.append(step)
                current = target
                planned_targets.append(target)

        if current != start:
            back_path = self._shortest_path(current, start, self._walkable_cells)
            if back_path:
                for step in back_path[1:]:
                    # Always close the loop so the drone eventually returns home.
                    route.append(step)

        self._planned_patrol_targets = planned_targets
        return route or [start]

    def _shortest_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Set[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """Breadth-first search to find the cheapest walkable path between cells."""
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
        """Convenience flood fill used to limit patrol targets to connected tiles."""
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
        """Return valid orthogonal neighbors constrained to the reserve bounds."""
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
        """Advance along the patrol route, looping when necessary."""
        if not self._patrol_route:
            return (0, 0)
        self._route_index = (self._route_index + 1) % len(self._patrol_route)
        return self._patrol_route[self._route_index]

    def _current_status(self) -> str:
        """Expose a human readable status for telemetry broadcasts."""
        if self.is_charging:
            return "charging"
        if self.is_returning_to_base:
            return "returning"
        return "patrolling"

    async def _broadcast_patrol_status(
        self, behaviour: "DroneAgent.PatrolBehaviour"
    ) -> None:
        """Send periodic telemetry updates to the ranger command agent."""
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

    async def _maybe_emit_patrol_alert(
        self, behaviour: "DroneAgent.PatrolBehaviour"
    ) -> None:
        """Allow the drone to raise opportunistic alerts while patrolling."""
        # TODO: replace random trigger with detections tied to poachers, movement, etc.
        if self._patrol_rng.random() >= 0.1:  # 10% chance to emit a patrol alert
            return
        alert_id = f"{self.jid}-patrol-{secrets.token_hex(4)}"
        alert_payload = {
            "sensor": str(self.jid),
            "id": alert_id,
            "pos": (self.position[0], self.position[1]),
            "confidence": round(self._patrol_rng.uniform(0.55, 0.9), 2),
            "ts": dt.datetime.utcnow().isoformat() + "Z",
        }
        ranger_payload = {
            "sensor": str(self.jid),
            "drone": str(self.jid),
            "alert": alert_payload,
            "ack": {"alert_id": alert_id, "source": "drone_patrol"},
        }
        msg = make_inform_alert(self.ranger_jid, ranger_payload)
        await behaviour.send(msg)
        self.log("Self-reported patrol alert", alert_id, "at", alert_payload["pos"])

    def log(self, *args: Any) -> None:
        """Prefix drone log messages for easier tracing in shared consoles."""
        print(f"[DRONE-{self.callsign}]", *args)

    def _estimate_energy_cost(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> float:
        """Approximate battery requirement to travel between two points."""
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
        """Spend battery when physically moving to a new grid cell."""
        if previous == current or self.battery_consumption_per_step <= 0:
            return
        before_level = self.battery_level
        self.battery_level = max(
            0.0, self.battery_level - self.battery_consumption_per_step
        )
        self._log_battery_drop_if_needed(before_level)

    def _log_battery_drop_if_needed(self, previous_level: float) -> None:
        """Emit log lines only when crossing 10% battery thresholds."""
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
        """Decide when to abort patrol so the drone has enough energy to get home."""
        if self.is_returning_to_base or self.is_charging:
            return False
        energy_to_base = self._estimate_energy_cost(self.position, self.base_position)
        if energy_to_base == float("inf"):
            return False
        threshold = energy_to_base + self.battery_consumption_per_step
        # Leave a one-step buffer so we do not strand the drone on the way back.
        return self.battery_level <= threshold

    def _begin_return_to_base(self) -> None:
        """Schedule a path back to base and flag the state machine accordingly."""
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
        """Step along the buffered return route until the drone reaches base."""
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
        """Incrementally recharge the battery, then flip state when full."""
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
        """Reset patrol state once the battery hits full capacity."""
        self.battery_level = self.max_battery
        self.is_charging = False
        self._route_index = -1
        self._patrol_route = self._build_patrol_route()
        self.position = self.base_position
        self.log("Battery full. Resuming patrol route.")
        self._next_battery_log_pct = 90.0 if self.max_battery > 0 else None

    async def _tick_patrol(self, behaviour: "DroneAgent.PatrolBehaviour") -> None:
        """Main patrol finite-state loop invoked each period."""
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
                # Announce the patrol queue on the first leg to aid debugging.
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
            await self._maybe_emit_patrol_alert(behaviour)

        await self._broadcast_patrol_status(behaviour)

    async def _collect_attachments(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Capture photo/IR samples concurrently and return base64 payloads."""
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
        """Send the sensor a quick ACK so it knows the alert was received."""
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
        """Forward the full alert package to the ranger command agent."""
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
        """Cyclic behaviour that receives sensor alerts addressed to the drone."""
        def __init__(self, drone: "DroneAgent") -> None:
            """Store a reference to the owning drone agent."""
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            """Listen for sensor alerts and let the drone process them."""
            msg = await self.receive(timeout=0.1)
            if not msg:
                return
            if msg.get_metadata("type") != ALERT_ANOMALY:
                # Telemetry and other categories fall through to other behaviours.
                self.drone.log(
                    "Ignoring non-alert message from", str(msg.sender)
                )
                return
            await self.drone.handle_sensor_alert(self, msg)
    class PatrolBehaviour(PeriodicBehaviour):
        """Periodic behaviour that ticks the drone patrol control loop."""
        def __init__(self, drone: "DroneAgent", period: float) -> None:
            """Tick the drone FSM at a fixed cadence."""
            super().__init__(period=period)
            self.drone = drone

        async def run(self) -> None:
            """Delegate the actual patrol logic to the drone."""
            await self.drone._tick_patrol(self)
