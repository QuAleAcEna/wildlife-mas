from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

# NOVO #
from core.messages import (
    ALERT_ANOMALY,
    TELEMETRY,
    INFORM,
    CFP,
    PROPOSE,
    ACCEPT,
    REJECT,
    CNP_ALERT,
    json_dumps,
    json_loads,
)
# NOVO #
from core.env import EnvironmentClock


class RangerAgent(Agent):
    """Ranger command agent that receives alerts and issues responses."""

    def __init__(
        self,
        jid: str,
        password: str,
        dispatch_delay_s: float = 12.0,
        clock: Optional[EnvironmentClock] = None,
    ):
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

        # NOVO #
        # Métricas simples de categoria de alertas e despachos
        self.alert_counts = {"poacher": 0, "herd": 0, "unknown": 0}
        self.dispatch_counts = {"poacher": 0, "herd": 0, "unknown": 0}

        # Lista de drones para o CNP (setada em run_all.py)
        self.drone_jids: List[str] = []

        # Estado do CNP: alert_id -> {incident, proposals, expected}
        self._cnp_pending: Dict[str, Dict[str, Any]] = {}
        # NOVO #

    async def setup(self) -> None:
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

        # NOVO #
        # CNP: Propostas dos drones
        cnp_behaviour = self.CNPProposalReceptionBehaviour(self)
        cnp_template = Template()
        cnp_template.set_metadata("performative", PROPOSE)
        cnp_template.set_metadata("type", CNP_ALERT)
        self.add_behaviour(cnp_behaviour, cnp_template)
        # NOVO #

    # ================================
    #   ALERT HANDLING + START CNP
    # ================================
    async def handle_drone_notification(
        self,
        behaviour: "RangerAgent.AlertReceptionBehaviour",
        msg: Message,
    ) -> None:
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty notification from", msg.sender)
            return

        self.alert_history.append(payload)

        # NOVO #
        # Extrair info do alerta e categoria
        alert_block = payload.get("alert", {}) if isinstance(payload.get("alert"), dict) else {}
        category = alert_block.get("category") or payload.get("category") or "unknown"
        if category not in self.alert_counts:
            category = "unknown"
        self.alert_counts[category] += 1
        alert_id = alert_block.get("id")
        # NOVO #

        # NOVO #
        # Iniciar CNP apenas para POACHER
        if category == "poacher":
            await self._start_cnp_for_alert(alert_block, category, behaviour)
        # NOVO #

        # Política de horário (poacher ignora janelas)
        if not self._can_dispatch_now(category):
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

        # Despacho físico do “ranger” em terra
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
            self.dispatch_counts[category] += 1
            await self._confirm_dispatch(behaviour, str(msg.sender), payload)
        else:
            self.log(
                "No dispatch sent for alert",
                alert_id,
                "due to constraints.",
            )

    # ================================
    #   CNP MANAGEMENT
    # ================================

    # NOVO #
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
    # NOVO #

    # ================================
    #   TELEMETRY + UTILITIES
    # ================================

    def _safe_load(self, body: str | None) -> Dict[str, Any]:
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

    def log(self, *args: Any) -> None:
        print("[RANGER]", *args)

    def _current_hour(self) -> int:
        if self.clock:
            return self.clock.current_hour % 24
        return dt.datetime.utcnow().hour

    def _within_operating_hours(self) -> bool:
        hour = self._current_hour()
        if 9 <= hour < 17:
            return True
        if hour >= 19 or hour <= 3:
            return True
        return False

    def _can_dispatch_now(self, category: str) -> bool:
        # Poacher ignora restrição de horário
        if category == "poacher":
            return True
        return self._within_operating_hours()

    # ================================
    #   PATH / FUEL
    # ================================

    def _plan_path_to_alert(self, payload: Dict[str, Any]) -> List[Tuple[int, int]]:
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

    def _update_field_position(self, new_position: Tuple[int, int]) -> None:
        self._current_position = new_position

    def _manhattan_distance(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> int:
        return abs(start[0] - target[0]) + abs(start[1] - target[1])

    def _build_manhattan_path(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
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
        to_target = self._manhattan_distance(start, target)
        to_base = self._manhattan_distance(target, self._base_position)
        return (to_target + to_base) * self.fuel_per_step

    def _ensure_fuel_for_mission(self, target: Tuple[int, int]) -> bool:
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
        if steps <= 0:
            return
        consumed = steps * self.fuel_per_step
        self.fuel_level = max(0.0, self.fuel_level - consumed)
        self.log("Fuel:", f"{self.fuel_level:.1f}/{self.max_fuel:.1f}")

    def _check_refuel_need(self) -> None:
        if self._current_position == self._base_position:
            return
        dist = self._manhattan_distance(self._current_position, self._base_position)
        need = (dist + self._fuel_return_margin_steps) * self.fuel_per_step
        if self.fuel_level <= need:
            self.log("Fuel low – returning to base")
            self._travel_to_base()

    def _travel_to_base(self) -> None:
        if self._current_position == self._base_position:
            self._refuel()
            return

        path = self._build_manhattan_path(self._current_position, self._base_position)
        self._consume_fuel(len(path) - 1)
        self._current_position = self._base_position
        self.log("Ranger arrived at base.")
        self._refuel()

    def _refuel(self) -> None:
        self.fuel_level = self.max_fuel
        self.log("Ranger refueled.")

    # ================================
    #   BEHAVIOURS
    # ================================

    class AlertReceptionBehaviour(CyclicBehaviour):
        def __init__(self, ranger: "RangerAgent"):
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if msg:
                await self.ranger.handle_drone_notification(self, msg)

    class TelemetryReceptionBehaviour(CyclicBehaviour):
        def __init__(self, ranger: "RangerAgent"):
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if msg:
                await self.ranger.handle_drone_telemetry(self, msg)

    # NOVO #
    class CNPProposalReceptionBehaviour(CyclicBehaviour):
        """Recebe PROPOSE dos drones no âmbito do CNP."""

        def __init__(self, ranger: "RangerAgent"):
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_cnp_proposal(self, msg)
    # NOVO #
