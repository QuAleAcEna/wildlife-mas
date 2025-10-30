from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

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
        """Initialise ranger state, including fuel levels and simulated clock."""
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

    async def setup(self) -> None:
        """Register behaviours so the ranger reacts to alerts and telemetry."""
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
        """Process new anomaly alerts and decide whether to dispatch the ranger."""
        payload = self._safe_load(msg.body)
        if not payload:
            self.log("Received empty notification from", msg.sender)
            return

        self.alert_history.append(payload)

        if not self._within_operating_hours():
            # Avoid dispatching during downtime windows; alert stays logged for later.
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
        path = self._plan_path_to_alert(payload)
        if path:
            route_str = " -> ".join(f"({x},{y})" for x, y in path)
            self.log(
                "Ranger route",
                route_str,
            )
            self.log(
                "Ranger arrived at alert",
                payload.get("alert", {}).get("id"),
                "after",
                len(path) - 1,
                "steps.",
            )
            self._update_field_position(path[-1])
            self._consume_fuel(len(path) - 1)
            self._check_refuel_need()
            await self._confirm_dispatch(behaviour, str(msg.sender), payload)
        else:
            self.log(
                "No dispatch sent for alert",
                payload.get("alert", {}).get("id"),
                "due to constraints.",
            )

    def _safe_load(self, body: str | None) -> Dict[str, Any]:
        """Best-effort JSON decode that keeps raw payloads for debugging."""
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
        """Tell the requesting drone that a ranger team is en route."""
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
        """Record patrol telemetry to keep situational awareness up to date."""
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
        """Consistent logging helper for ranger output."""
        print("[RANGER]", *args)

    def _current_hour(self) -> int:
        """Read the current clock hour from simulation or real time."""
        if self.clock:
            return self.clock.current_hour % 24
        return dt.datetime.utcnow().hour

    def _within_operating_hours(self) -> bool:
        """Return True when the ranger is allowed to deploy."""
        hour = self._current_hour()
        if 9 <= hour < 17:
            return True
        if hour >= 19 or hour <= 3:
            # Evening window covers rapid response during peak poaching hours.
            return True
        return False

    def _plan_path_to_alert(self, payload: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Validate coordinates and build the movement plan to the alert site."""
        alert_pos = payload.get("alert", {}).get("pos")
        if not isinstance(alert_pos, (list, tuple)) or len(alert_pos) != 2:
            self.log(
                "Cannot plan path: invalid alert position",
                alert_pos,
            )
            return []
        try:
            target = (int(alert_pos[0]), int(alert_pos[1]))
        except (TypeError, ValueError):
            self.log("Cannot plan path: non-numeric alert position", alert_pos)
            return []

        if not self._ensure_fuel_for_mission(target):
            self.log(
                "Skipping alert",
                payload.get("alert", {}).get("id"),
                "due to fuel constraints.",
            )
            return []

        return self._build_manhattan_path(self._current_position, target)

    def _update_field_position(self, new_position: Tuple[int, int]) -> None:
        """Track the ranger's last known position to inform future routing."""
        self._current_position = new_position

    def _manhattan_distance(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> int:
        """Return the grid distance between two points (cheapest orthogonal path)."""
        return abs(start[0] - target[0]) + abs(start[1] - target[1])

    def _build_manhattan_path(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Construct a simple orthogonal path by walking rows first then columns."""
        path: List[Tuple[int, int]] = [start]
        current_x, current_y = start

        if current_x != target[0]:
            step_x = 1 if target[0] > current_x else -1
            for x in range(current_x + step_x, target[0] + step_x, step_x):
                path.append((x, current_y))
        current_x = target[0]

        if current_y != target[1]:
            step_y = 1 if target[1] > current_y else -1
            for y in range(current_y + step_y, target[1] + step_y, step_y):
                path.append((current_x, y))

        return path

    def _fuel_needed_for_mission(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> float:
        """Calculate the round-trip consumption for a mission."""
        distance_to_target = self._manhattan_distance(start, target)
        distance_to_base = self._manhattan_distance(target, self._base_position)
        return (distance_to_target + distance_to_base) * self.fuel_per_step

    def _ensure_fuel_for_mission(self, target: Tuple[int, int]) -> bool:
        """Check fuel, attempt refuel if possible, and block missions when unsafe."""
        required = self._fuel_needed_for_mission(self._current_position, target)
        if self.fuel_level >= required:
            return True

        if self._current_position != self._base_position:
            self.log(
                "Fuel insufficient for mission. Returning to base to refuel first."
            )
            self._travel_to_base()
            required = self._fuel_needed_for_mission(self._current_position, target)

        if self.fuel_level >= required:
            return True

        if self._current_position == self._base_position and self.fuel_level < self.max_fuel:
            self._refuel()
            required = self._fuel_needed_for_mission(self._current_position, target)
            if self.fuel_level >= required:
                return True

        if required > self.max_fuel:
            self.log(
                "Alert target requires",
                f"{required:.1f}",
                "fuel but maximum is",
                f"{self.max_fuel:.1f}.",
            )
        else:
            self.log(
                "Fuel level still insufficient for mission. Current:",
                f"{self.fuel_level:.1f}/{self.max_fuel:.1f}",
            )
        return False

    def _consume_fuel(self, steps: int) -> None:
        """Reduce onboard fuel after moving between grid cells."""
        if steps <= 0 or self.fuel_per_step <= 0:
            return
        consumed = steps * self.fuel_per_step
        previous = self.fuel_level
        self.fuel_level = max(0.0, self.fuel_level - consumed)
        if self.fuel_level < previous:
            self.log(
                "Fuel status",
                f"{self.fuel_level:.1f}/{self.max_fuel:.1f}",
            )

    def _check_refuel_need(self) -> None:
        """Trigger an immediate return if the ranger cannot safely reach base."""
        if self._current_position == self._base_position:
            return
        distance_to_base = self._manhattan_distance(
            self._current_position, self._base_position
        )
        minimum_required = (
            distance_to_base + self._fuel_return_margin_steps
        ) * self.fuel_per_step
        if self.fuel_level <= minimum_required:
            self.log("Fuel low after mission. Returning to base to refuel.")
            self._travel_to_base()

    def _travel_to_base(self) -> None:
        """Walk the manhattan route back to base and refuel on arrival."""
        if self._current_position == self._base_position:
            self._refuel()
            return

        path = self._build_manhattan_path(
            self._current_position, self._base_position
        )
        if len(path) <= 1:
            self._current_position = self._base_position
            self._refuel()
            return
        route_str = " -> ".join(f"({x},{y})" for x, y in path)
        self.log("Returning to base", route_str)
        self._consume_fuel(len(path) - 1)
        self._current_position = self._base_position
        self.log("Ranger arrived at base to refuel.")
        self._refuel()

    def _refuel(self) -> None:
        """Top up the ranger's fuel reserves at base."""
        if self.fuel_level >= self.max_fuel:
            return
        self.fuel_level = self.max_fuel
        self.log("Ranger refueled to full capacity.")

    class AlertReceptionBehaviour(CyclicBehaviour):
        """Cyclic behaviour that waits for new drone anomaly reports."""
        def __init__(self, ranger: "RangerAgent") -> None:
            """Keep a reference to the parent agent for delegating message handling."""
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            """Poll the inbox and forward anomaly alerts to the ranger logic."""
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_drone_notification(self, msg)

    class TelemetryReceptionBehaviour(CyclicBehaviour):
        """Cyclic behaviour that listens for routine drone telemetry updates."""
        def __init__(self, ranger: "RangerAgent") -> None:
            """Retain parent reference for telemetry forwarding."""
            super().__init__()
            self.ranger = ranger

        async def run(self) -> None:
            """Relay telemetry messages to the parent agent for processing."""
            msg = await self.receive(timeout=0.5)
            if not msg:
                return
            await self.ranger.handle_drone_telemetry(self, msg)
