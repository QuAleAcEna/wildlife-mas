from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message

from core.messages import INFORM, TELEMETRY, json_dumps, json_loads


class RangerAgent(Agent):
    """Ranger command agent that receives alerts and issues responses."""

    def __init__(self, jid: str, password: str, dispatch_delay_s: float = 12.0):
        super().__init__(jid, password)
        self.dispatch_delay_s = dispatch_delay_s
        self.alert_history: List[Dict[str, Any]] = []

    async def setup(self) -> None:
        self.log("Ranger ready for alerts…")
        self.add_behaviour(self.AlertReceptionBehaviour(self))

    async def handle_drone_notification(self, msg: Message) -> None:
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty notification from", msg.sender)
            return

        self.alert_history.append(payload)
        self.log(
            "Dispatching to alert",
            payload.get("alert", {}).get("id"),
            "from",
            payload.get("sensor"),
        )

        await self._confirm_dispatch(str(msg.sender), payload)

    def _safe_load(self, body: str | None) -> Dict[str, Any]:
        if not body:
            return {}
        try:
            return json_loads(body)
        except Exception:
            return {"raw_body": body}

    async def _confirm_dispatch(self, drone: str, payload: Dict[str, Any]) -> None:
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
        await self.send(msg)

    def log(self, *args: Any) -> None:
        print("[RANGER]", *args)

    class AlertReceptionBehaviour(CyclicBehaviour):
        def __init__(self, ranger: "RangerAgent") -> None:
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_drone_notification(msg)