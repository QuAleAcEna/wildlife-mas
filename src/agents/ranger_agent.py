"""SPADe ranger command agent that orchestrates alerts, dispatches and drones."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template


from core.messages import (
    ALERT_ANOMALY,
    TELEMETRY,
    INFORM,
    CFP,
    PROPOSE,
    ACCEPT,
    REJECT,
    CNP_ALERT,
    DIRECT_ALERT,
    json_dumps,
    json_loads,
)

from core.env import EnvironmentClock

__all__ = ["RangerAgent"]


class RangerAgent(Agent):
    """Ranger command agent that receives alerts and issues responses."""

    def __init__(
        self,
        jid: str,
        password: str,
        dispatch_delay_s: float = 12.0,
        clock: Optional[EnvironmentClock] = None,
    ):
        """Initialize the ranger with timing, fuel, and collaboration settings.

        Args:
            jid (str): XMPP identifier for the ranger agent.
            password (str): XMPP password for the ranger agent.
            dispatch_delay_s (float, optional): Delay applied before starting a patrol.
            clock (EnvironmentClock | None, optional): Shared simulation clock.
        """
        super().__init__(jid, password)
        self.dispatch_delay_s = dispatch_delay_s
        self.alert_history: List[Dict[str, Any]] = []
        self.telemetry_history: List[Dict[str, Any]] = []
        self.clock = clock
        self._base_position: Tuple[int, int] = (0, 0)
        self._current_position: Tuple[int, int] = self._base_position
        self.max_fuel: float = 200.0
        self.fuel_per_step: float = 1.0
        self.fuel_level: float = self.max_fuel
        self._fuel_return_margin_steps: int = 5
        self.travel_time_per_step_s: float = 1.0

        
        # Métricas simples de categoria de alertas e despachos
        self.alert_counts = {"poacher": 0, "herd": 0, "unknown": 0}
        self.dispatch_counts = {"poacher": 0, "herd": 0, "unknown": 0}

        # Lista de drones para o CNP (setada em run_all.py)
        self.drone_jids: List[str] = []

        # Estado do CNP: alert_id -> {incident, proposals, expected}
        self._cnp_pending: Dict[str, Dict[str, Any]] = {}
        

        # Ligação opcional a um writer externo (dashboard Week 6)
        self.metrics_writer: Optional[Any] = None
        self._drone_positions: Dict[str, Tuple[int, int]] = {}
        self._direct_assignments: Dict[str, str] = {}
        self.reserve: Optional[Any] = None
        self._alert_backlog: List[Dict[str, Any]] = []
        self._processing_alerts: bool = False

    async def setup(self) -> None:
        """Register behaviours to receive alerts, telemetry, and CNP proposals."""
        self.log("Ranger ready for alerts…")

        # ALERTS
        alert_behaviour = self.AlertReceptionBehaviour(self)
        alert_template = Template()
        alert_template.set_metadata("performative", INFORM)
        alert_template.set_metadata("type", ALERT_ANOMALY)
        self.add_behaviour(alert_behaviour, alert_template)

        # TELEMETRY
        telemetry_behaviour = self.TelemetryReceptionBehaviour(self)
        telemetry_template = Template()
        telemetry_template.set_metadata("performative", INFORM)
        telemetry_template.set_metadata("type", TELEMETRY)
        self.add_behaviour(telemetry_behaviour, telemetry_template)

        
        # CNP: Propostas dos drones
        cnp_behaviour = self.CNPProposalReceptionBehaviour(self)
        cnp_template = Template()
        cnp_template.set_metadata("performative", PROPOSE)
        cnp_template.set_metadata("type", CNP_ALERT)
        self.add_behaviour(cnp_behaviour, cnp_template)
        

    # ================================
    #   ALERT HANDLING + START CNP
    # ================================
    async def handle_drone_notification(
        self,
        behaviour: "RangerAgent.AlertReceptionBehaviour",
        msg: Message,
    ) -> None:
        """React to incoming anomaly notifications from drones or sensors.

        Args:
            behaviour (AlertReceptionBehaviour): Behaviour relaying the message.
            msg (Message): SPADE message that triggered the callback.
        """
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty notification from", msg.sender)
            return

        self.alert_history.append(payload)

        
        # Extrair info do alerta e categoria
        alert_block = payload.get("alert", {}) if isinstance(payload.get("alert"), dict) else {}
        category = alert_block.get("category") or payload.get("category") or "unknown"
        if category not in self.alert_counts:
            category = "unknown"
        self.alert_counts[category] += 1
        alert_id = alert_block.get("id")
        

        
        # Iniciar CNP apenas para POACHER
        confidence = alert_block.get("confidence") or payload.get("confidence")
        high_conf_poacher = (
            category == "poacher"
            and isinstance(confidence, (int, float))
            and confidence >= 0.7
        )

        if category == "poacher" and not high_conf_poacher:
            await self._start_cnp_for_alert(alert_block, category, behaviour)
        elif high_conf_poacher:
            await self._order_drone_follow(alert_block)
        

        # Política de horário (poacher ignora janelas)
        if not self._can_dispatch_now(category) and not high_conf_poacher:
            self.log(
                "Deferring alert",
                alert_id,
                "category",
                category,
                "outside operating window (hour",
                self._current_hour(),
                ").",
            )
            return

        # Enfileirar alerta e processar pela proximidade
        self._alert_backlog.append(
            {
                "payload": payload,
                "behaviour": behaviour,
                "category": category,
                "high_conf": high_conf_poacher,
                "alert_id": alert_id,
                "sender": str(msg.sender),
            }
        )
        await self._process_alert_backlog()

    # ================================
    #   CNP MANAGEMENT
    # ================================

    
    async def _start_cnp_for_alert(
        self,
        alert_block: Dict[str, Any],
        category: str,
        behaviour: "RangerAgent.AlertReceptionBehaviour",
    ) -> None:
        """Inicia o CNP para escolher qual drone responde ao alerta."""
        if not self.drone_jids:
            self.log("No drone JIDs for CNP – skipping.")
            return

        alert_id = alert_block.get("id")
        pos = alert_block.get("pos")
        if not alert_id or not isinstance(pos, (list, tuple)) or len(pos) != 2:
            self.log("Cannot start CNP: invalid alert payload", alert_block)
            return

        try:
            target = (int(pos[0]), int(pos[1]))
        except (TypeError, ValueError):
            self.log("Cannot start CNP – invalid numeric pos")
            return

        incident = {
            "alert_id": alert_id,
            "category": category,
            "pos": target,
            "manager": str(self.jid),
        }

        self._cnp_pending[alert_id] = {
            "incident": incident,
            "proposals": [],
            "expected": len(self.drone_jids),
        }

        self.log(
            "Starting CNP for alert",
            alert_id,
            "category",
            category,
            "pos",
            target,
            "expecting",
            len(self.drone_jids),
            "proposals.",
        )

        # Enviar CFP para todos os drones via behaviour
        for djid in self.drone_jids:
            msg = Message(to=djid)
            msg.set_metadata("performative", CFP)
            msg.set_metadata("type", CNP_ALERT)
            msg.body = json_dumps(incident)
            await behaviour.send(msg)

    async def handle_cnp_proposal(
        self,
        behaviour: "RangerAgent.CNPProposalReceptionBehaviour",
        msg: Message,
    ) -> None:
        """Processa mensagens PROPOSE dos drones."""
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Empty CNP PROPOSE")
            return

        alert_id = payload.get("alert_id")
        if not alert_id or alert_id not in self._cnp_pending:
            self.log("PROPOSE for unknown alert", alert_id)
            return

        pending = self._cnp_pending[alert_id]
        pending["proposals"].append(payload)

        self.log(
            "CNP PROPOSE received:",
            "alert",
            alert_id,
            "drone",
            payload.get("drone"),
            "cost",
            payload.get("cost"),
            "can_serve",
            payload.get("can_serve"),
        )

        if len(pending["proposals"]) >= pending["expected"]:
            await self._finalise_cnp(alert_id, behaviour)

    async def _finalise_cnp(
        self,
        alert_id: str,
        behaviour: "RangerAgent.CNPProposalReceptionBehaviour",
    ) -> None:
        """Escolhe vencedor e envia ACCEPT/REJECT."""
        pending = self._cnp_pending.pop(alert_id, None)
        if not pending:
            return

        incident = pending["incident"]
        proposals: List[Dict[str, Any]] = pending["proposals"]

        feasible = [
            p
            for p in proposals
            if p.get("can_serve") and p.get("cost", float("inf")) < float("inf")
        ]

        if not feasible:
            self.log("CNP: No feasible proposals for alert", alert_id)
            return

        winner = min(feasible, key=lambda p: p.get("cost"))
        winner_jid = winner.get("drone")

        self.log(
            "CNP winner for alert",
            alert_id,
            "is",
            winner_jid,
            "with cost",
            winner.get("cost"),
        )

        for p in proposals:
            drone_jid = p.get("drone")
            if not drone_jid:
                continue
            msg = Message(to=drone_jid)
            msg.set_metadata("type", CNP_ALERT)
            if drone_jid == winner_jid:
                msg.set_metadata("performative", ACCEPT)
                msg.body = json_dumps(incident)
            else:
                msg.set_metadata("performative", REJECT)
                msg.body = json_dumps(
                    {"alert_id": alert_id, "reason": "winner_better"}
                )
            await behaviour.send(msg)
    

    # ================================
    #   TELEMETRY + UTILITIES
    # ================================

    def _safe_load(self, body: str | None) -> Dict[str, Any]:
        """Parse JSON payloads defensively, surfacing malformed bodies as-is.

        Args:
            body (str | None): Raw message body.

        Returns:
            Dict[str, Any]: Parsed dictionary or a `raw_body` placeholder.
        """
        if not body:
            return {}
        try:
            return json_loads(body)
        except Exception:
            return {"raw_body": body}

    async def _confirm_dispatch(
        self,
        behaviour: "RangerAgent.AlertReceptionBehaviour",
        drone: str,
        payload: Dict[str, Any],
    ) -> None:
        """Send a telemetry acknowledgement confirming that a dispatch started.

        Args:
            behaviour (AlertReceptionBehaviour): Behaviour used to reply.
            drone (str): Target drone JID for the confirmation.
            payload (Dict[str, Any]): Original alert payload for context.
        """
        response = {
            "ranger": str(self.jid),
            "alert_id": payload.get("alert", {}).get("id"),
            "ack_id": payload.get("ack", {}).get("alert_id"),
            "received_at": dt.datetime.utcnow().isoformat() + "Z",
            "status": "dispatching",
            "estimated_departure_s": self.dispatch_delay_s,
        }
        msg = Message(to=drone)
        msg.set_metadata("performative", INFORM)
        msg.set_metadata("type", TELEMETRY)
        msg.body = json_dumps(response)
        await behaviour.send(msg)

    async def handle_drone_telemetry(
        self,
        _: "RangerAgent.TelemetryReceptionBehaviour",
        msg: Message,
    ) -> None:
        """Persist and log telemetry updates shared by drones.

        Args:
            _ (TelemetryReceptionBehaviour): Behaviour placeholder.
            msg (Message): Incoming message containing telemetry.
        """
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Empty telemetry")
            return

        self.telemetry_history.append(payload)
        self.log(
            "Telemetry from",
            payload.get("drone", str(msg.sender)),
            "position",
            payload.get("position"),
        )
        drone_id = str(payload.get("drone") or msg.sender)
        pos = payload.get("position")
        coords: Optional[Tuple[int, int]] = None
        if isinstance(pos, dict):
            x = pos.get("x")
            y = pos.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                coords = (int(x), int(y))
        elif isinstance(pos, (list, tuple)) and len(pos) == 2:
            try:
                coords = (int(pos[0]), int(pos[1]))
            except Exception:
                coords = None
        if coords:
            self._drone_positions[drone_id] = coords

    def log(self, *args: Any) -> None:
        """Central logging helper that prefixes records with the agent role."""
        print("[RANGER]", *args)

    def _current_hour(self) -> int:
        """Return the current simulation hour, falling back to wall-clock time."""
        if self.clock:
            return self.clock.current_hour % 24
        return dt.datetime.utcnow().hour

    def _within_operating_hours(self) -> bool:
        """Check whether the ranger's schedule permits dispatches right now."""
        hour = self._current_hour()
        if 9 <= hour < 17:
            return True
        if hour >= 19 or hour <= 3:
            return True
        return False

    def _can_dispatch_now(self, category: str) -> bool:
        """Determine if an alert category can be serviced immediately.

        Args:
            category (str): Alert category tag.

        Returns:
            bool: True when deployment is allowed for the current hour.
        """
        if category == "poacher":
            return True
        return self._within_operating_hours()

    # ================================
    #   PATH / FUEL
    # ================================

    def _plan_path_to_alert(self, payload: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Translate an alert payload into a traversable path for the ranger.

        Args:
            payload (Dict[str, Any]): Alert containing a nested `alert.pos`.

        Returns:
            List[Tuple[int, int]]: Manhattan path including current position.
        """
        alert_pos = payload.get("alert", {}).get("pos")
        if not isinstance(alert_pos, (list, tuple)) or len(alert_pos) != 2:
            self.log("Invalid alert position", alert_pos)
            return []

        try:
            target = (int(alert_pos[0]), int(alert_pos[1]))
        except Exception:
            self.log("Non-numeric alert pos", alert_pos)
            return []

        if not self._ensure_fuel_for_mission(target):
            self.log("Skipping alert due to fuel")
            return []

        return self._build_manhattan_path(self._current_position, target)

    async def _order_drone_follow(self, alert_block: Dict[str, Any]) -> None:
        """Instruct the nearest drone to shadow a high-confidence poacher alert.

        Args:
            alert_block (Dict[str, Any]): Canonical alert dictionary.
        """
        alert_id = alert_block.get("id")
        pos = alert_block.get("pos")
        if (
            not alert_id
            or not isinstance(pos, (list, tuple))
            or len(pos) != 2
        ):
            return
        try:
            target = (int(pos[0]), int(pos[1]))
        except Exception:
            return
        drone_jid = self._nearest_drone_jid(target)
        if not drone_jid:
            self.log("No drone available for direct follow on", alert_id)
            return
        payload = {
            "alert_id": alert_id,
            "category": alert_block.get("category", "poacher"),
            "pos": target,
            "action": "follow",
        }
        msg = Message(to=drone_jid)
        msg.set_metadata("performative", INFORM)
        msg.set_metadata("type", DIRECT_ALERT)
        msg.body = json_dumps(payload)
        await self.send(msg)
        self._direct_assignments[alert_id] = drone_jid
        self.log("Ordered direct follow on", alert_id, "by", drone_jid)

    async def _notify_drone_relief(self, alert_id: Optional[str]) -> None:
        """Signal a drone to stand down once the ranger resolves a direct alert.

        Args:
            alert_id (str | None): Tracking identifier for the alert.
        """
        if not alert_id:
            return
        drone_jid = self._direct_assignments.pop(alert_id, None)
        if not drone_jid:
            return
        payload = {"alert_id": alert_id, "action": "stand_down"}
        msg = Message(to=drone_jid)
        msg.set_metadata("performative", INFORM)
        msg.set_metadata("type", DIRECT_ALERT)
        msg.body = json_dumps(payload)
        await self.send(msg)
        self.log("Notified", drone_jid, "to stand down for", alert_id)

    def _nearest_drone_jid(self, target: Tuple[int, int]) -> Optional[str]:
        """Select the drone positioned closest to the provided coordinates.

        Args:
            target (Tuple[int, int]): Target grid coordinates.

        Returns:
            str | None: Drone JID or None when no drones registered.
        """
        best: Optional[str] = None
        best_dist = float("inf")
        for jid, pos in self._drone_positions.items():
            dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            if dist < best_dist:
                best = jid
                best_dist = dist
        if best:
            return best
        # fallback to the first registered drone
        return self.drone_jids[0] if self.drone_jids else None

    async def _process_alert_backlog(self) -> None:
        """Continuously process queued alerts sorted by distance."""
        if self._processing_alerts:
            return
        self._processing_alerts = True
        try:
            while self._alert_backlog:
                idx = self._select_closest_alert_index()
                record = self._alert_backlog.pop(idx)
                await self._handle_single_alert(record)
        finally:
            self._processing_alerts = False

    def _select_closest_alert_index(self) -> int:
        """Pick the index of the alert nearest to the current ranger location."""
        if not self._alert_backlog:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for idx, record in enumerate(self._alert_backlog):
            alert = record.get("payload", {}).get("alert", {})
            pos = alert.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                continue
            try:
                target = (int(pos[0]), int(pos[1]))
            except Exception:
                continue
            dist = self._manhattan_distance(self._current_position, target)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    async def _handle_single_alert(self, record: Dict[str, Any]) -> None:
        """Resolve a single alert by travelling, logging metrics, and confirming.

        Args:
            record (Dict[str, Any]): Cached backlog entry.
        """
        payload = record["payload"]
        behaviour = record["behaviour"]
        category = record["category"]
        high_conf_poacher = record["high_conf"]
        alert_id = record.get("alert_id")

        if not self._can_dispatch_now(category) and not high_conf_poacher:
            self.log(
                "Deferring alert",
                alert_id,
                "category",
                category,
                "outside operating window (hour",
                self._current_hour(),
                ").",
            )
            self._alert_backlog.append(record)
            await asyncio.sleep(1)
            return

        self.log(
            "Dispatching to alert",
            alert_id,
            "from",
            payload.get("sensor"),
            "category",
            category,
        )

        path = self._plan_path_to_alert(payload)
        if path:
            route_str = " -> ".join(f"({x},{y})" for x, y in path)
            self.log("Ranger route", route_str)
            await self._travel_path(path)
            self.log(
                "Ranger arrived at alert",
                alert_id,
                "after",
                len(path) - 1,
                "steps.",
            )
            self._update_field_position(path[-1])
            self._consume_fuel(len(path) - 1)
            self._check_refuel_need()
            captured_poacher = False
            if category == "poacher":
                captured_poacher = self._capture_poacher_near(path[-1])
                if captured_poacher:
                    self.log("Poacher apprehended at", path[-1], "- transporting to base.")
                    await self._escort_poacher_to_base()
            self.dispatch_counts[category] += 1
            if self.metrics_writer and path:
                steps = max(0, len(path) - 1)
                self.metrics_writer.record_response(steps)
                self.metrics_writer.record_energy("RANGER", steps)
            await self._confirm_dispatch(behaviour, record.get("sender", ""), payload)
            if high_conf_poacher:
                await self._notify_drone_relief(alert_id)
        else:
            self.log(
                "No dispatch sent for alert",
                alert_id,
                "due to constraints.",
            )
            if high_conf_poacher:
                await self._notify_drone_relief(alert_id)

    def _capture_poacher_near(self, position: Tuple[int, int], radius: int = 1) -> bool:
        """Attempt to deactivate poachers inside a radius around the ranger.

        Args:
            position (Tuple[int, int]): Ranger position where the search begins.
            radius (int, optional): Search radius in Manhattan steps.

        Returns:
            bool: True when at least one poacher is neutralised.
        """
        reserve = getattr(self, "reserve", None)
        engine = getattr(reserve, "events", None) if reserve else None
        if engine is None:
            return False
        captured = False
        px, py = position
        for poacher in list(engine.poachers):
            dist = abs(poacher.pos[0] - px) + abs(poacher.pos[1] - py)
            if dist <= radius:
                poacher.active = False
                captured = True
                self.log("Apprehended poacher", poacher.id, "at", poacher.pos)
        if captured:
            engine.poachers = [p for p in engine.poachers if p.active]
        return captured

    async def _escort_poacher_to_base(self) -> None:
        """Simulate transporting an apprehended poacher back to base."""
        if self._current_position == self._base_position:
            self.log("Already at base with detainee.")
            return
        path = self._build_manhattan_path(self._current_position, self._base_position)
        self.log("Escorting detainee to base over", len(path) - 1, "steps.")
        for step, waypoint in enumerate(path[1:], start=1):
            await asyncio.sleep(self.travel_time_per_step_s)
            self._update_field_position(waypoint)
            self._consume_fuel(1)
            self.log("Escort step", step, "->", waypoint)
        self.log("Poacher delivered to base.")
        self._refuel()

    def _update_field_position(self, new_position: Tuple[int, int]) -> None:
        """Persist the ranger's last known position for future planning.

        Args:
            new_position (Tuple[int, int]): Coordinates reached by the ranger.
        """
        self._current_position = new_position

    async def _travel_path(self, path: List[Tuple[int, int]]) -> None:
        """Walk a path step-by-step while respecting dispatch and travel delays.

        Args:
            path (List[Tuple[int, int]]): Ordered list of coordinates to visit.
        """
        if len(path) <= 1:
            return
        if self.dispatch_delay_s > 0:
            self.log("Preparing to depart in", f"{self.dispatch_delay_s:.1f}s")
            await asyncio.sleep(self.dispatch_delay_s)
        for step, waypoint in enumerate(path[1:], start=1):
            await asyncio.sleep(self.travel_time_per_step_s)
            self._update_field_position(waypoint)
            self.log("Ranger step", step, "->", waypoint)

    def _manhattan_distance(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> int:
        """Return the Manhattan distance between two cells.

        Args:
            start (Tuple[int, int]): Origin coordinates.
            target (Tuple[int, int]): Destination coordinates.

        Returns:
            int: Number of axis-aligned steps separating the cells.
        """
        return abs(start[0] - target[0]) + abs(start[1] - target[1])

    def _build_manhattan_path(
         self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Create a straight-line path restricted to horizontal/vertical moves.

        Args:
            start (Tuple[int, int]): Starting coordinates.
            target (Tuple[int, int]): Destination coordinates.

        Returns:
            List[Tuple[int, int]]: Intermediate steps along the path.
        """
        path = [start]
        cx, cy = start
        tx, ty = target

        if cx != tx:
            step = 1 if tx > cx else -1
            for x in range(cx + step, tx + step, step):
                path.append((x, cy))
        cx = tx

        if cy != ty:
            step = 1 if ty > cy else -1
            for y in range(cy + step, ty + step, step):
                path.append((cx, y))

        return path

    def _fuel_needed_for_mission(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> float:
        """Estimate the fuel requirements for a round trip to the alert.

        Args:
            start (Tuple[int, int]): Origin coordinates.
            target (Tuple[int, int]): Target coordinates.

        Returns:
            float: Consumption estimate in fuel units.
        """
        to_target = self._manhattan_distance(start, target)
        to_base = self._manhattan_distance(target, self._base_position)
        return (to_target + to_base) * self.fuel_per_step

    def _ensure_fuel_for_mission(self, target: Tuple[int, int]) -> bool:
        """Guarantee enough fuel exists to travel to the alert and return.

        Args:
            target (Tuple[int, int]): Coordinates of the alert.

        Returns:
            bool: True when a mission can start safely.
        """
        required = self._fuel_needed_for_mission(self._current_position, target)

        if self.fuel_level >= required:
            return True

        if self._current_position != self._base_position:
            self.log("Fuel insufficient – returning to base")
            self._travel_to_base()
            required = self._fuel_needed_for_mission(self._current_position, target)

        if self.fuel_level >= required:
            return True

        if self._current_position == self._base_position and self.fuel_level < self.max_fuel:
            self._refuel()
            required = self._fuel_needed_for_mission(self._current_position, target)
            if self.fuel_level >= required:
                return True

        return False

    def _consume_fuel(self, steps: int) -> None:
        """Decrease the fuel level based on traversed steps.

        Args:
            steps (int): Number of movement steps performed.
        """
        if steps <= 0:
            return
        consumed = steps * self.fuel_per_step
        self.fuel_level = max(0.0, self.fuel_level - consumed)
        self.log("Fuel:", f"{self.fuel_level:.1f}/{self.max_fuel:.1f}")

    def _check_refuel_need(self) -> None:
        """Force a return to base when fuel margins shrink below the buffer."""
        if self._current_position == self._base_position:
            return
        dist = self._manhattan_distance(self._current_position, self._base_position)
        need = (dist + self._fuel_return_margin_steps) * self.fuel_per_step
        if self.fuel_level <= need:
            self.log("Fuel low – returning to base")
            self._travel_to_base()

    def _travel_to_base(self) -> None:
        """Relocate the ranger to base, consuming fuel along the way."""
        if self._current_position == self._base_position:
            self._refuel()
            return

        path = self._build_manhattan_path(self._current_position, self._base_position)
        self._consume_fuel(len(path) - 1)
        self._current_position = self._base_position
        self.log("Ranger arrived at base.")
        self._refuel()

    def _refuel(self) -> None:
        """Reset the fuel level to maximum capacity."""
        self.fuel_level = self.max_fuel
        self.log("Ranger refueled.")

    # ================================
    #   BEHAVIOURS
    # ================================

    class AlertReceptionBehaviour(CyclicBehaviour):
        """Background behaviour that waits for incoming anomaly alerts."""

        def __init__(self, ranger: "RangerAgent"):
            """Link the behaviour to its owning ranger agent."""
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            """Pull messages from the queue and hand them to the ranger handler."""
            msg = await self.receive(timeout=0.5)
            if msg:
                await self.ranger.handle_drone_notification(self, msg)

    class TelemetryReceptionBehaviour(CyclicBehaviour):
        """Continuously ingests telemetry updates sent by drones."""

        def __init__(self, ranger: "RangerAgent"):
            """Persist the ranger reference for later callbacks."""
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            """Receive telemetry and forward it to the ranger processor."""
            msg = await self.receive(timeout=0.5)
            if msg:
                await self.ranger.handle_drone_telemetry(self, msg)

    # NOVO #
    class CNPProposalReceptionBehaviour(CyclicBehaviour):
        """Receive and buffer PROPOSE replies from drones during the CNP."""

        def __init__(self, ranger: "RangerAgent"):
            """Store the owning ranger instance for message handling."""
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            """Deliver CNP proposals to the ranger as they arrive."""
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_cnp_proposal(self, msg)
    # NOVO #
