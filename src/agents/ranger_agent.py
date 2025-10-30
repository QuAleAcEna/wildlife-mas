from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from core.messages import ALERT_ANOMALY, INFORM, TELEMETRY, json_dumps, json_loads
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

    async def setup(self) -> None:
        self.log("Ranger ready for alerts…")
        alert_behaviour = self.AlertReceptionBehaviour(self)
        alert_template = Template()
        alert_template.set_metadata("performative", INFORM)
        alert_template.set_metadata("type", ALERT_ANOMALY)
        self.add_behaviour(alert_behaviour, alert_template)

        telemetry_behaviour = self.TelemetryReceptionBehaviour(self)
        telemetry_template = Template()
        telemetry_template.set_metadata("performative", INFORM)
        telemetry_template.set_metadata("type", TELEMETRY)
        self.add_behaviour(telemetry_behaviour, telemetry_template)

    async def handle_drone_notification(
        self, behaviour: "RangerAgent.AlertReceptionBehaviour", msg: Message
    ) -> None:
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty notification from", msg.sender)
            return

        self.alert_history.append(payload)

        if not self._within_operating_hours():
            current_hour = self._current_hour()
            self.log(
                "Outside operating hours (hour",
                current_hour,
                "). Deferring response to alert",
                payload.get("alert", {}).get("id"),
            )
            return

        self.log(
            "Dispatching to alert",
            payload.get("alert", {}).get("id"),
            "from",
            payload.get("sensor"),
        )

        await self._confirm_dispatch(behaviour, str(msg.sender), payload)

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
        self, _: "RangerAgent.TelemetryReceptionBehaviour", msg: Message
    ) -> None:
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty telemetry from", msg.sender)
            return

        self.telemetry_history.append(payload)
        self.log(
            "Telemetry update from",
            payload.get("drone", str(msg.sender)),
            "position",
            payload.get("position"),
            f"route {payload.get('route_index')} / {payload.get('route_length')}",
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

    class AlertReceptionBehaviour(CyclicBehaviour):
        def __init__(self, ranger: "RangerAgent") -> None:
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_drone_notification(self, msg)

    class TelemetryReceptionBehaviour(CyclicBehaviour):
        def __init__(self, ranger: "RangerAgent") -> None:
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_drone_telemetry(self, msg)
