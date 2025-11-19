"""Autonomous drone agent responsible for patrols and alert triage."""

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
    DIRECT_ALERT,
    # NOVO #
    json_dumps,
    json_loads,
    make_inform_alert,
)
from core.env import Reserve

__all__ = ["DroneAgent", "_default_sampler"]

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
        # Caminho para regressar ao setor após incidente
        self._rejoin_path: Deque[Tuple[int, int]] = deque()
        # NOVO #
        self._global_walkable: Set[Tuple[int, int]] = self._compute_global_walkable()
        self._observed_events: Set[str] = set()

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

        direct_behaviour = self.DirectAssignmentBehaviour(self)
        direct_template = Template()
        direct_template.set_metadata("type", DIRECT_ALERT)
        self.add_behaviour(direct_behaviour, direct_template)

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
        await self._reply_to_sensor(behaviour, sensor, ack_payload)
        if not self._confirm_sensor_alert(payload):
            self.log(
                "Alert dismissed (no threat confirmed)",
                payload.get("id"),
                payload.get("pos"),
            )
            return

        attachments = await self._collect_attachments(payload)
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
        """Convert a distance into an estimated arrival time.

        Args:
            distance_m (float): Distance from the current location in meters.

        Returns:
            float: ETA in seconds at cruise speed.
        """
        if self.cruise_speed_mps <= 0:
            return 0.0
        return max(0.0, distance_m / self.cruise_speed_mps)

    # ---------------------------------------------------------------
    #   PATROL ROUTE GENERATION
    # ---------------------------------------------------------------

    def _normalize_sector(
        self, sector: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Clamp patrol sectors to the reserve bounds and validate ordering."""
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
        """Return True when a cell falls inside the configured patrol sector."""
        if not self._sector_bounds:
            return True
        x_min, y_min, x_max, y_max = self._sector_bounds
        return x_min <= cell[0] <= x_max and y_min <= cell[1] <= y_max

    def _build_patrol_route(self) -> List[Tuple[int, int]]:
        """Generate a deterministic serpentine patrol route over walkable cells."""
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
        """Compute a BFS path between two cells restricted to walkable coordinates."""
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
        """Return the subset of free cells reachable from a starting cell."""
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
        """Enumerate orthogonally adjacent cells that stay inside the reserve."""
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

    def _compute_global_walkable(self) -> Set[Tuple[int, int]]:
        """Cache every traversable cell, ignoring patrol sector constraints."""
        cells: Set[Tuple[int, int]] = {
            (x, y)
            for y in range(self.reserve.height)
            for x in range(self.reserve.width)
            if not self.reserve.is_no_fly((x, y))
        }
        if not cells:
            cells.add(self.base_position)
        return cells

    def _next_waypoint(self) -> Tuple[int, int]:
        """Advance the patrol index and return the next waypoint."""
        if not self._patrol_route:
            return (0, 0)
        self._route_index = (self._route_index + 1) % len(self._patrol_route)
        return self._patrol_route[self._route_index]

    # ---------------------------------------------------------------
    #   BATTERY / ENERGY
    # ---------------------------------------------------------------

    def _estimate_energy_cost(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walkable: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Estimate energy required to travel between two points.

        Args:
            start (Tuple[int, int]): Current drone coordinates.
            goal (Tuple[int, int]): Target coordinates.
            walkable (set[Tuple[int, int]] | None): Graph to use for routing.

        Returns:
            float: Required energy units or inf when no path is available.
        """
        if start == goal:
            return 0.0

        base_walkable = walkable or (self._walkable_cells and set(self._walkable_cells)) or {start}
        if (
            (start not in base_walkable or goal not in base_walkable)
            and self._global_walkable
        ):
            base_walkable = self._global_walkable

        path = self._shortest_path(start, goal, base_walkable)
        if not path:
            return float("inf")

        steps = max(0, len(path) - 1)
        return steps * self.battery_consumption_per_step

    def _battery_required_for_round_trip(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        walkable: Optional[Set[Tuple[int, int]]] = None,
    ) -> float:
        """Estimate the energy required to reach a target and return to base."""
        to_target = self._estimate_energy_cost(start, target, walkable)
        back_home = self._estimate_energy_cost(target, self.base_position, walkable)
        if to_target == float("inf") or back_home == float("inf"):
            return float("inf")
        return to_target + back_home

    def _consume_battery_for_move(
        self, previous: Tuple[int, int], current: Tuple[int, int]
    ) -> None:
        """Deduct energy for a single movement step and emit threshold logs."""
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
        """Determine when the drone must leave patrol to recharge at base."""
        if self.is_returning_to_base or self.is_charging:
            return False

        energy_to_base = self._estimate_energy_cost(self.position, self.base_position)
        if energy_to_base == float("inf"):
            return False

        threshold = energy_to_base + self.battery_consumption_per_step
        return self.battery_level <= threshold

    def _begin_return_to_base(self) -> None:
        """Plan and start following a route back to base for charging."""
        walkable: Optional[Set[Tuple[int, int]]] = self._walkable_cells
        if (
            not walkable
            or self.position not in walkable
            or self.base_position not in walkable
        ):
            walkable = self._global_walkable or walkable or {self.position}
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
        """Move one step along the return path and manage arrival bookkeeping."""
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
        """Incrementally recharge the drone until the battery is full."""
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
        """Reset patrol state after a full recharge."""
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

    def _enqueue_incident(self, payload: Dict[str, Any]) -> None:
        """Queue incidents confirmed through either CNP or direct orders."""
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

        incident = {
            "id": incident_id,
            "pos": target,
            "category": category,
            "follow": bool(payload.get("follow")),
        }

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

    def _plan_incident_path(self) -> None:
        """Pick the next feasible incident and prepare the travel path."""
        while self._incident_queue:
            incident = self._incident_queue.popleft()
            target = incident["pos"]
            required = self._battery_required_for_round_trip(
                self.position,
                target,
                walkable=self._global_walkable,
            )

            if required == float("inf"):
                self.log("Skipping unreachable incident", incident["id"])
                continue
            if required > self.battery_level:
                self.log("Skipping low battery incident", incident["id"])
                continue

            walkable = self._global_walkable or self._walkable_cells or {self.position}
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

        # Nothing viable
        self._active_incident = None
        self._incident_path.clear()

    def _schedule_rejoin_patrol(self) -> None:
        """Plan how to return to the patrol sector or base after an incident."""
        walkable = self._global_walkable or self._walkable_cells or {self.position}
        path = self._shortest_path(self.position, self.base_position, walkable)
        if not path or len(path) <= 1:
            self._rejoin_path.clear()
            if self.position != self.base_position:
                self.position = self.base_position
            self._route_index = -1
            return
        self._rejoin_path = deque(path[1:])
        self.log(
            "Returning to patrol sector via",
            len(self._rejoin_path),
            "steps.",
        )

    def _cancel_incident(self, alert_id: Optional[str]) -> None:
        """Remove queued or active incidents, typically via stand-down orders."""
        if not alert_id:
            return
        self._incident_queue = deque(
            incident for incident in self._incident_queue if incident.get("id") != alert_id
        )
        if self._active_incident and self._active_incident.get("id") == alert_id:
            self.log("Cancelling active incident", alert_id)
            self._active_incident = None
            self._incident_path.clear()
            self._schedule_rejoin_patrol()

    def _refresh_follow_target(self) -> None:
        """Continuously retarget follow incidents towards the latest coordinates."""
        incident = self._active_incident
        if not incident or not incident.get("follow"):
            return
        engine = getattr(self.reserve, "events", None)
        if engine is None:
            return
        desired_category = incident.get("category", "poacher")
        current_target = incident.get("pos") or self.position
        nearby = engine.nearby_entities(
            tuple(current_target),
            radius=3,
            kinds=(desired_category,),
        )
        candidates = nearby.get(desired_category, [])
        if not candidates:
            return
        pos, _, entity = min(candidates, key=lambda item: item[1])
        new_target = (int(pos[0]), int(pos[1]))
        if incident.get("pos") != new_target:
            incident["pos"] = new_target
            incident["entity_id"] = getattr(entity, "id", None)
            self._recompute_follow_path(new_target)
        elif not self._incident_path:
            self._recompute_follow_path(new_target)

    def _recompute_follow_path(self, target: Tuple[int, int]) -> None:
        """Rebuild the path deque towards an updated follow target."""
        walkable = self._global_walkable or self._walkable_cells or {self.position}
        path = self._shortest_path(self.position, target, walkable)
        if path and len(path) > 1:
            self._incident_path = deque(path[1:])
        else:
            self._incident_path.clear()

    # ---------------------------------------------------------------
    #   PATROL LOOP
    # ---------------------------------------------------------------

    async def _tick_patrol(self, behaviour: "DroneAgent.PatrolBehaviour") -> None:
        """Advance the drone state machine for patrol, incidents, or charging.

        Args:
            behaviour (PatrolBehaviour): Periodic behaviour driving the loop.
        """
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

        # 3) Seguimento direto de incidente (follow mode)
        elif self._active_incident and self._active_incident.get("follow"):
            self._refresh_follow_target()
            if self._incident_path:
                nxt = self._incident_path.popleft()
                self.position = nxt
                self._consume_battery_for_move(previous, nxt)
                self.log(
                    "Tracking incident",
                    self._active_incident.get("id"),
                    "→",
                    self.position,
                    f"{len(self._incident_path)} steps left",
                )
            else:
                self.log(
                    "Tracking incident",
                    self._active_incident.get("id"),
                    "holding position",
                    self.position,
                )

        # 4) Modo incidente normal
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
                if self._active_incident and not self._active_incident.get("follow"):
                    self._active_incident = None
                    self._schedule_rejoin_patrol()

        # 5) Caminho de regresso ao setor após incidente
        elif self._rejoin_path:
            nxt = self._rejoin_path.popleft()
            self.position = nxt
            self._consume_battery_for_move(previous, nxt)
            self.log(
                "Rejoining sector",
                self.position,
                f"{len(self._rejoin_path)} steps left",
            )
            if not self._rejoin_path:
                self.log("Back at patrol base, resuming route.")
                self._route_index = -1
                self.position = self.base_position

        # 6) Tentar apanhar incidente da fila
        elif self._incident_queue:
            self._plan_incident_path()

        # 7) Patrulha normal
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

        await self._scan_events_nearby(behaviour)

        # TELEMETRIA
        await self._broadcast_patrol_status(behaviour)

    # ---------------------------------------------------------------
    #   TELEMETRY + ATTACHMENTS
    # ---------------------------------------------------------------

    async def _broadcast_patrol_status(
        self, behaviour: "DroneAgent.PatrolBehaviour"
    ) -> None:
        """Emit telemetry updates to the ranger containing path and battery info.

        Args:
            behaviour (PatrolBehaviour): Behaviour used to send the message.
        """
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
        """Occasionally raise a synthetic alert while patrolling for extra noise.

        Args:
            behaviour (PatrolBehaviour): Behaviour used to send alerts.
        """
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

    async def _scan_events_nearby(self, behaviour: "DroneAgent.PatrolBehaviour") -> None:
        """Look for simulated world entities and send real-time alerts.

        Args:
            behaviour (PatrolBehaviour): Behaviour used for message sending.
        """
        engine = getattr(self.reserve, "events", None)
        if engine is None:
            return
        nearby = engine.nearby_entities(
            self.position,
            radius=2,
            kinds=("poacher", "herd"),
        )
        for category in ("poacher", "herd"):
            for pos, _, entity in nearby.get(category, []):
                entity_id = getattr(entity, "id", None) or f"{category}-{pos[0]}-{pos[1]}"
                if entity_id in self._observed_events:
                    continue
                self._observed_events.add(entity_id)
                payload = {
                    "sensor": str(self.jid),
                    "id": f"{self.jid}-visual-{entity_id}",
                    "pos": list(pos),
                    "confidence": 0.85,
                    "ts": dt.datetime.utcnow().isoformat() + "Z",
                    "category": category,
                    "sensor_pos": self.position,
                }
                msg = make_inform_alert(self.ranger_jid, payload)
                await behaviour.send(msg)
                self.log("Visual alert", category, "at", pos)

    async def _collect_attachments(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Generate mock attachment payloads for the alert being escalated.

        Args:
            payload (Dict[str, Any]): Alert payload forwarded to samplers.

        Returns:
            Dict[str, str]: Base64-encoded blobs for photo and IR channels.
        """
        photo_task = asyncio.to_thread(self.photo_sampler, payload)
        ir_task = asyncio.to_thread(self.ir_sampler, payload)
        photo_bytes, ir_bytes = await asyncio.gather(photo_task, ir_task)

        return {
            "photo_base64": base64.b64encode(photo_bytes).decode(),
            "ir_base64": base64.b64encode(ir_bytes).decode(),
        }

    def _confirm_sensor_alert(self, payload: Dict[str, Any]) -> bool:
        """
        Drone faz uma confirmação rápida usando o estado do motor de eventos
        antes de escalar para o ranger.
        """
        engine = getattr(self.reserve, "events", None)
        if engine is None:
            return True

        raw_pos = payload.get("pos")
        if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 2:
            return False
        try:
            target = (int(raw_pos[0]), int(raw_pos[1]))
        except (TypeError, ValueError):
            return False

        category = payload.get("category")
        nearby = engine.nearby_entities(
            target,
            radius=2,
            kinds=("poacher", "herd"),
        )

        if isinstance(category, str) and category in nearby:
            if nearby[category]:
                return True

        return any(nearby_list for nearby_list in nearby.values())

    async def _reply_to_sensor(
        self,
        behaviour: "DroneAgent.AlertRelayBehaviour",
        sensor: str,
        ack_payload: Dict[str, Any],
    ) -> None:
        """Send an acknowledgement back to the originating sensor.

        Args:
            behaviour (AlertRelayBehaviour): Behaviour used to send the reply.
            sensor (str): Sensor JID.
            ack_payload (Dict[str, Any]): Telemetry payload to send.
        """
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
        """Forward a sensor alert, attachments, and receipt data to the ranger.

        Args:
            behaviour (AlertRelayBehaviour): Behaviour responsible for sends.
            sensor (str): Source sensor JID.
            original_payload (Dict[str, Any]): Alert payload from the sensor.
            ack_payload (Dict[str, Any]): Telemetry acknowledgement data.
            attachments (Dict[str, str]): Evidence base64 blobs.
        """
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
        """Listen for `ALERT_ANOMALY` messages coming from sensors."""

        def __init__(self, drone: "DroneAgent"):
            """Store the owning drone so callbacks can be invoked."""
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            """Receive alerts and hand them to the drone for processing."""
            msg = await self.receive(timeout=0.2)
            if msg:
                await self.drone.handle_sensor_alert(self, msg)

    class PatrolBehaviour(PeriodicBehaviour):
        """Heartbeat behaviour that drives the patrol state machine."""

        def __init__(self, drone: "DroneAgent", period: float) -> None:
            """Attach the drone reference and set the desired interval."""
            super().__init__(period=period)
            self.drone = drone

        async def run(self) -> None:
            """Delegate to the drone's patrol tick implementation."""
            await self.drone._tick_patrol(self)

    # ---------------------------------------------------------------
    #   CNP BEHAVIOURS
    # ---------------------------------------------------------------

    # NOVO #
    class CNPParticipationBehaviour(CyclicBehaviour):
        """Handle CNP CFP invitations by replying with a PROPOSE message."""

        def __init__(self, drone: "DroneAgent"):
            """Remember the owning drone to reuse helper methods."""
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            """Evaluate CFPs and emit proposals when the mission is feasible."""
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
                self.drone.position,
                target,
                walkable=self.drone._global_walkable,
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
        """Process ACCEPT/REJECT responses to previously sent proposals."""

        def __init__(self, drone: "DroneAgent"):
            """Hold on to the drone reference."""
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            """Enqueue incidents on ACCEPT and log rejections."""
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

    class DirectAssignmentBehaviour(CyclicBehaviour):
        """Handle direct follow/stand-down commands sent by the ranger."""

        def __init__(self, drone: "DroneAgent"):
            """Persist the drone reference for later callbacks."""
            super().__init__()
            self.drone = drone

        async def run(self) -> None:
            """Interpret commands and enqueue or cancel incidents accordingly."""
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            payload = self.drone._safe_load(msg.body)
            action = payload.get("action", "follow")
            if action == "follow":
                payload["follow"] = True
                self.drone._enqueue_incident(payload)
                if not self.drone._incident_path and not self.drone._active_incident:
                    self.drone._plan_incident_path()
            elif action == "stand_down":
                self.drone._cancel_incident(payload.get("alert_id"))

    # ---------------------------------------------------------------
    #   LOGGING
    # ---------------------------------------------------------------

    def log(self, *args: Any) -> None:
        """Emit tagged log messages to aid with debugging and demos."""
        print(f"[DRONE-{self.callsign}]", *args)
