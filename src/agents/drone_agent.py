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
# NOVO #
from spade.template import Template
# NOVO #

from core.messages import (
    ALERT_ANOMALY,
    INFORM,
    TELEMETRY,
    # NOVO #
    CFP,
    PROPOSE,
    ACCEPT,
    REJECT,
    CNP_ALERT,
    # NOVO #
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

        # NOVO #
        # Fila de incidentes
        self._incident_queue: Deque[Dict[str, Any]] = deque()
        # Caminho até o incidente ativo
        self._incident_path: Deque[Tuple[int, int]] = deque()
        # Incidente atualmente em resolução
        self._active_incident: Optional[Dict[str, Any]] = None
        # NOVO #

    async def setup(self) -> None:
        """Start behaviours for relaying alerts and patrolling the reserve."""

        # NOVO #
        # ALERTS — Agora com Template para NÃO engolir mensagens do CNP
        alert_behaviour = self.AlertRelayBehaviour(self)
        alert_template = Template()
        alert_template.set_metadata("performative", INFORM)
        alert_template.set_metadata("type", ALERT_ANOMALY)
        self.add_behaviour(alert_behaviour, alert_template)

        # CNP — Receber CFP → enviar PROPOSE
        cnp_participation = self.CNPParticipationBehaviour(self)
        cfp_template = Template()
        cfp_template.set_metadata("performative", CFP)
        cfp_template.set_metadata("type", CNP_ALERT)
        self.add_behaviour(cnp_participation, cfp_template)

        # CNP — Receber ACCEPT / REJECT
        cnp_decision = self.CNPDecisionBehaviour(self)
        decision_template = Template()
        decision_template.set_metadata("type", CNP_ALERT)
        self.add_behaviour(cnp_decision, decision_template)
        # NOVO #

        # Patrulha periódica
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

        # NOVO #
        # Antes: enfileirávamos aqui. Agora, só CNP decide → drones NÃO enfileiram aqui.
        # NOVO #

        ack_payload = self._build_ack_payload(sensor, payload)
        attachments = await self._collect_attachments(payload)
        await self._reply_to_sensor(behaviour, sensor, ack_payload)
        await self._notify_ranger(
            behaviour, sensor, payload, ack_payload, attachments
        )

    # ---------------------------------------------------------------
    #   BASIC HELPERS
    # ---------------------------------------------------------------

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
        if self.cruise_speed_mps <= 0:
            return 0.0
        return max(0.0, distance_m / self.cruise_speed_mps)

    # ---------------------------------------------------------------
    #   PATROL ROUTE GENERATION
    # ---------------------------------------------------------------

    def _normalize_sector(
        self, sector: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
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
        if not self._sector_bounds:
            return True
        x_min, y_min, x_max, y_max = self._sector_bounds
        return x_min <= cell[0] <= x_max and y_min <= cell[1] <= y_max

    def _build_patrol_route(self) -> List[Tuple[int, int]]:
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

        if (
            self.base_position not in free_cells
            and not self.reserve.is_no_fly(self.base_position)
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
            start = next(
                (cell for cell in ordered_targets if cell in reachable),
                start_candidate,
            )

        self._walkable_cells = set(reachable)

        route: List[Tuple[int, int]] = [start]
        current = start
        planned_targets: List[Tuple[int, int]] = []

        available_targets = [cell for cell in self._walkable_cells if cell != start]
        if available_targets and self.patrol_waypoint_count > 0:
            waypoint_count = min(len(available_targets), self.patrol_waypoint_count)
            random_targets = self._patrol_rng.sample(
                available_targets, waypoint_count
            )
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

    # ---------------------------------------------------------------
    #   GRAPH HELPERS
    # ---------------------------------------------------------------

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
        cand = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        out = []
        for nx, ny in cand:
            if 0 <= nx < self.reserve.width and 0 <= ny < self.reserve.height:
                out.append((nx, ny))
        return out

    def _next_waypoint(self) -> Tuple[int, int]:
        if not self._patrol_route:
            return (0, 0)
        self._route_index = (self._route_index + 1) % len(self._patrol_route)
        return self._patrol_route[self._route_index]

    # ---------------------------------------------------------------
    #   BATTERY / ENERGY
    # ---------------------------------------------------------------

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

    # NOVO #
    def _battery_required_for_round_trip(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> float:
        """Energia para ir e voltar ao ponto base."""
        to_target = self._estimate_energy_cost(start, target)
        back_home = self._estimate_energy_cost(target, self.base_position)
        if to_target == float("inf") or back_home == float("inf"):
            return float("inf")
        return to_target + back_home
    # NOVO #

    def _consume_battery_for_move(
        self, previous: Tuple[int, int], current: Tuple[int, int]
    ) -> None:
        if previous == current:
            return
        before = self.battery_level
        self.battery_level = max(
            0.0, self.battery_level - self.battery_consumption_per_step
        )

        # Logging thresholds
        if (
            self._next_battery_log_pct is not None
            and self.max_battery > 0
            and before > self.battery_level
        ):
            prev_pct = (before / self.max_battery) * 100
            cur_pct = (self.battery_level / self.max_battery) * 100
            while (
                self._next_battery_log_pct is not None
                and prev_pct > self._next_battery_log_pct >= cur_pct
            ):
                self.log(
                    f"Battery at {self._next_battery_log_pct:.0f}% "
                    f"({self.battery_level:.1f}/{self.max_battery:.1f})."
                )
                self._next_battery_log_pct -= 10
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
            self.log("Unable to compute return to base.")
            return
        if len(path) <= 1:
            self.is_charging = True
            self.is_returning_to_base = False
            self.position = self.base_position
            self.log("Already at base.")
            return

        self._return_path = deque(path[1:])
        self.is_returning_to_base = True
        self.log(
            "Battery low. Returning via",
            len(self._return_path),
            "steps.",
        )

    def _advance_return_path(self, previous_position: Tuple[int, int]) -> None:
        if not self._return_path:
            self.is_returning_to_base = False
            self.is_charging = True
            self.position = self.base_position
            self.log("Arrived at base.")
            return

        nxt = self._return_path.popleft()
        self.position = nxt
        self._consume_battery_for_move(previous_position, nxt)
        self.log(
            "Returning",
            self.position,
            f"{len(self._return_path)} steps left.",
        )
        if not self._return_path:
            self.is_returning_to_base = False
            self.is_charging = True
            self.position = self.base_position
            self.log("Arrived at base.")

    def _perform_charging(self) -> None:
        if self.battery_level >= self.max_battery:
            self._finish_charging()
            return

        self.battery_level = min(
            self.max_battery,
            self.battery_level + self.charge_rate_per_tick,
        )
        if self.battery_level >= self.max_battery:
            self._finish_charging()

    def _finish_charging(self) -> None:
        self.battery_level = self.max_battery
        self.is_charging = False
        self._route_index = -1
        self._patrol_route = self._build_patrol_route()
        self.position = self.base_position
        self._next_battery_log_pct = 90.0 if self.max_battery > 0 else None
        self.log("Battery full. Resuming patrol.")

    # ---------------------------------------------------------------
    #   INCIDENT MANAGEMENT
    # ---------------------------------------------------------------

    # NOVO #
    def _enqueue_incident(self, payload: Dict[str, Any]) -> None:
        """Enfileira incidentes decididos via ACCEPT no CNP."""
        if "pos" in payload and isinstance(payload["pos"], (list, tuple)):
            raw_pos = payload["pos"]
            incident_id = payload.get("alert_id") or payload.get("id")
            category = payload.get("category", "unknown")
        else:
            raw_alert = payload.get("alert", payload)
            raw_pos = raw_alert.get("pos")
            incident_id = raw_alert.get("id") or payload.get("id")
            category = raw_alert.get("category") or payload.get("category") or "unknown"

        if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 2:
            return

        try:
            target = (int(raw_pos[0]), int(raw_pos[1]))
        except (TypeError, ValueError):
            return

        incident = {"id": incident_id, "pos": target, "category": category}

        # Prioridade: poacher vai para o início
        if category == "poacher":
            self._incident_queue.appendleft(incident)
        else:
            self._incident_queue.append(incident)

        self.log(
            "Queued incident",
            incident_id,
            "category",
            category,
            "at",
            target,
            "queue size:",
            len(self._incident_queue),
        )

    # NOVO #
    def _plan_incident_path(self) -> None:
        """Seleciona incidente viável e cria caminho para lá."""
        while self._incident_queue:
            incident = self._incident_queue.popleft()
            target = incident["pos"]
            required = self._battery_required_for_round_trip(self.position, target)

            if required == float("inf"):
                self.log("Skipping unreachable incident", incident["id"])
                continue
            if required > self.battery_level:
                self.log("Skipping low battery incident", incident["id"])
                continue

            walkable = self._walkable_cells or {self.position}
            path = self._shortest_path(self.position, target, walkable)
            if not path or len(path) <= 1:
                self.log("No path to incident", incident["id"])
                continue

            self._incident_path = deque(path[1:])
            self._active_incident = incident
            self.log(
                "Responding to incident",
                incident["id"],
                "via",
                len(self._incident_path),
                "steps.",
            )
            return

        # Nada viável
        self._active_incident = None
        self._incident_path.clear()
    # NOVO #

    # ---------------------------------------------------------------
    #   PATROL LOOP
    # ---------------------------------------------------------------

    async def _tick_patrol(self, behaviour: "DroneAgent.PatrolBehaviour") -> None:
        previous = self.position

        # Regra base: bateria
        if not self.is_charging and self._should_return_to_base():
            self._begin_return_to_base()

        # 1) Modo carregamento
        if self.is_charging:
            self._perform_charging()

        # 2) Modo regressar
        elif self.is_returning_to_base:
            self._advance_return_path(previous)

        # 3) Modo incidente ativo
        elif self._incident_path:
            nxt = self._incident_path.popleft()
            self.position = nxt
            self._consume_battery_for_move(previous, nxt)
            remaining = len(self._incident_path)
            self.log(
                "Incident",
                (self._active_incident or {}).get("id"),
                "→",
                self.position,
                f"{remaining} steps left",
            )
            if remaining == 0 and self._active_incident is not None:
                self.log(
                    "Arrived at incident",
                    self._active_incident.get("id"),
                    "category",
                    self._active_incident.get("category"),
                    "at",
                    self.position,
                )
                self._active_incident = None

        # 4) Tentar apanhar incidente da fila
        elif self._incident_queue:
            self._plan_incident_path()

        # 5) Patrulha normal
        else:
            nxt = self._next_waypoint()
            if previous == self.base_position and nxt != self.base_position:
                if self._planned_patrol_targets:
                    targets_str = ", ".join(
                        f"({x},{y})" for x, y in self._planned_patrol_targets
                    )
                    self.log("Departing base to:", targets_str)
            self.position = nxt
            self._consume_battery_for_move(previous, nxt)
            self.log(
                "Patrol",
                self.position,
                f"[{self._route_index + 1}/{len(self._patrol_route)}]",
            )
            await self._maybe_emit_patrol_alert(behaviour)

        # TELEMETRIA
        await self._broadcast_patrol_status(behaviour)

    # ---------------------------------------------------------------
    #   TELEMETRY + ATTACHMENTS
    # ---------------------------------------------------------------

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
                "status": (
                    "charging"
                    if self.is_charging
                    else "returning"
                    if self.is_returning_to_base
                    else "patrolling"
                ),
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
        """Alerts espontâneos de patrulha (não são poachers reais)."""
        if self._patrol_rng.random() > 0.1:
            return

        alert_id = f"{self.jid}-patrol-{secrets.token_hex(4)}"
        payload = {
            "sensor": str(self.jid),
            "id": alert_id,
            "pos": (self.position[0], self.position[1]),
            "confidence": round(self._patrol_rng.uniform(0.55, 0.9), 2),
            "ts": dt.datetime.utcnow().isoformat() + "Z",
        }
        package = {
            "sensor": str(self.jid),
            "drone": str(self.jid),
            "alert": payload,
            "ack": {"alert_id": alert_id, "source": "drone_patrol"},
        }

        msg = make_inform_alert(self.ranger_jid, package)
        await behaviour.send(msg)
        self.log("Self alert:", alert_id, "at", payload["pos"])

    async def _collect_attachments(self, payload: Dict[str, Any]) -> Dict[str, str]:
        photo_task = asyncio.to_thread(self.photo_sampler, payload)
        ir_task = asyncio.to_thread(self.ir_sampler, payload)
        photo_bytes, ir_bytes = await asyncio.gather(photo_task, ir_task)

        return {
            "photo_base64": base64.b64encode(photo_bytes).decode(),
            "ir_base64": base64.b64encode(ir_bytes).decode(),
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

    # ---------------------------------------------------------------
    #   BEHAVIOURS
    # ---------------------------------------------------------------

    class AlertRelayBehaviour(CyclicBehaviour):
        """Recebe ALERT_ANOMALY dos sensores (graças ao Template)."""

        def __init__(self, drone: "DroneAgent"):
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            msg = await self.receive(timeout=0.2)
            if msg:
                await self.drone.handle_sensor_alert(self, msg)

    class PatrolBehaviour(PeriodicBehaviour):
        """Tick principal da patrulha."""

        def __init__(self, drone: "DroneAgent", period: float) -> None:
            super().__init__(period=period)
            self.drone = drone

        async def run(self) -> None:
            await self.drone._tick_patrol(self)

    # ---------------------------------------------------------------
    #   CNP BEHAVIOURS
    # ---------------------------------------------------------------

    # NOVO #
    class CNPParticipationBehaviour(CyclicBehaviour):
        """Recebe CFP → envia PROPOSE."""

        def __init__(self, drone: "DroneAgent"):
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            payload = self.drone._safe_load(msg.body)
            alert_id = payload.get("alert_id")
            pos = payload.get("pos")
            category = payload.get("category", "unknown")

            if not alert_id or not isinstance(pos, (list, tuple)) or len(pos) != 2:
                self.drone.log("Invalid CFP:", payload)
                return

            try:
                target = (int(pos[0]), int(pos[1]))
            except (TypeError, ValueError):
                self.drone.log("Bad CFP pos:", pos)
                return

            required = self.drone._battery_required_for_round_trip(
                self.drone.position, target
            )
            can_serve = (
                required < float("inf") and required <= self.drone.battery_level
            )

            proposal = {
                "alert_id": alert_id,
                "drone": str(self.drone.jid),
                "category": category,
                "pos": target,
                "cost": required if can_serve else float("inf"),
                "battery": self.drone.battery_level,
                "can_serve": can_serve,
            }

            reply = Message(to=self.drone.ranger_jid)
            reply.set_metadata("performative", PROPOSE)
            reply.set_metadata("type", CNP_ALERT)
            reply.body = json_dumps(proposal)
            await self.send(reply)

            self.drone.log(
                "PROPOSE:", alert_id, "cost", proposal["cost"], "serve?", can_serve
            )

    class CNPDecisionBehaviour(CyclicBehaviour):
        """Recebe ACCEPT / REJECT da fase final do CNP."""

        def __init__(self, drone: "DroneAgent"):
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return

            perf = msg.get_metadata("performative")
            payload = self.drone._safe_load(msg.body)

            if perf == ACCEPT:
                alert_id = payload.get("alert_id")
                pos = payload.get("pos")
                category = payload.get("category", "unknown")
                if not isinstance(pos, (list, tuple)):
                    return

                self.drone.log(
                    "ACCEPT:", alert_id, "pos", pos, "cat", category
                )
                self.drone._enqueue_incident(payload)

            elif perf == REJECT:
                self.drone.log("REJECT:", payload)
    # NOVO #

    # ---------------------------------------------------------------
    #   LOGGING
    # ---------------------------------------------------------------

    def log(self, *args: Any) -> None:
        print(f"[DRONE-{self.callsign}]", *args)
