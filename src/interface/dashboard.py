from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.animaltracker_agent import AnimalTrackerAgent
from agents.drone_agent import DroneAgent
from agents.ranger_agent import RangerAgent
from agents.sensor_agent import SensorAgent
from core.env import Reserve
from core.events import Herd, Poacher, WorldEventEngine

Coord = Tuple[int, int]


def _safe_tuple(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    if len(value) != 2:
        return None
    return int(value[0]), int(value[1])


@dataclass
class DashboardStateWriter:
    """
    Periodicamente gera uma fotografia JSON do estado do sistema para a interface web.

    Produz um ficheiro ``state.json`` (por omissão em ``interface/static``)
    que é lido pelo frontend HTML/JS.
    """

    reserve: Reserve
    events: WorldEventEngine
    ranger: RangerAgent
    drones: Sequence[DroneAgent]
    sensors: Sequence[SensorAgent]
    trackers: Sequence[AnimalTrackerAgent]
    output_path: Path
    interval: float = 1.0

    def __post_init__(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        """Main loop that serializes state snapshots at fixed cadence."""
        while True:
            data = self._build_snapshot()
            tmp_path = self.output_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.replace(self.output_path)
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # Snapshot assembly helpers
    # ------------------------------------------------------------------
    def _build_snapshot(self) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat() + "Z"
        clock = getattr(self.reserve, "clock", None)
        snapshot = {
            "generated_at": now,
            "clock": {
                "day": getattr(clock, "current_day", None),
                "hour": getattr(clock, "current_hour", None),
                "seconds_per_hour": getattr(clock, "seconds_per_hour", None),
            },
            "reserve": {
                "width": self.reserve.width,
                "height": self.reserve.height,
                "no_fly": list(self.reserve.no_fly),
            },
            "agents": {
                "ranger": self._snapshot_ranger(),
                "drones": [self._snapshot_drone(d) for d in self.drones],
                "sensors": [self._snapshot_sensor(s) for s in self.sensors],
                "trackers": [self._snapshot_tracker(t) for t in self.trackers],
            },
            "events": {
                "poachers": [self._snapshot_poacher(p) for p in self.events.poachers],
                "herds": [self._snapshot_herd(h) for h in self.events.herds],
            },
            "alerts": {
                "recent": self._recent_alerts(limit=12),
                "cnp_pending": self._cnp_pending(),
            },
            "metrics": self._metrics_payload(),
        }
        return snapshot

    def _metrics_payload(self) -> Dict[str, Any]:
        alert_breakdown = dict(self.ranger.alert_counts)
        dispatch_breakdown = dict(self.ranger.dispatch_counts)
        pending_cnp = getattr(self.ranger, "_cnp_pending", {})
        metrics = {
            "alerts_total": len(self.ranger.alert_history),
            "dispatch_total": sum(dispatch_breakdown.values()),
            "alert_breakdown": alert_breakdown,
            "dispatch_breakdown": dispatch_breakdown,
            "cnp_active": len(pending_cnp),
            "drones_total": len(self.drones),
            "sensors_total": len(self.sensors),
            "trackers_total": len(self.trackers),
            "poachers_active": len(self.events.poachers),
            "herds_active": len(self.events.herds),
        }
        return metrics

    def _recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        alerts = self.ranger.alert_history[-limit:]
        return [self._slim_alert(a) for a in alerts]

    def _cnp_pending(self) -> List[Dict[str, Any]]:
        pending = getattr(self.ranger, "_cnp_pending", {})
        out: List[Dict[str, Any]] = []
        for alert_id, block in pending.items():
            out.append(
                {
                    "alert_id": alert_id,
                    "category": block.get("incident", {}).get("category"),
                    "pos": block.get("incident", {}).get("pos"),
                    "received_proposals": len(block.get("proposals", [])),
                    "expected_proposals": block.get("expected"),
                }
            )
        return out

    def _slim_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        payload = alert.copy()
        category = payload.get("category")
        sensor_pos = payload.get("sensor_pos") or payload.get("sensor_position")
        return {
            "id": payload.get("id") or payload.get("alert", {}).get("id"),
            "sensor": payload.get("sensor"),
            "category": category,
            "pos": payload.get("pos") or payload.get("alert", {}).get("pos"),
            "confidence": payload.get("confidence"),
            "timestamp": payload.get("ts") or payload.get("timestamp"),
            "sensor_pos": sensor_pos,
        }

    def _snapshot_ranger(self) -> Dict[str, Any]:
        pos = getattr(self.ranger, "_current_position", (0, 0))
        base = getattr(self.ranger, "_base_position", (0, 0))
        return {
            "jid": str(self.ranger.jid),
            "position": list(pos),
            "base": list(base),
            "fuel_level": self.ranger.fuel_level,
            "max_fuel": self.ranger.max_fuel,
        }

    def _snapshot_drone(self, drone: DroneAgent) -> Dict[str, Any]:
        if drone.max_battery > 0:
            battery_pct = round((drone.battery_level / drone.max_battery) * 100, 1)
        else:
            battery_pct = 0.0
        status = "patrolling"
        if getattr(drone, "is_charging", False):
            status = "charging"
        elif getattr(drone, "is_returning_to_base", False):
            status = "returning"
        elif getattr(drone, "_active_incident", None):
            status = "responding"

        queue = getattr(drone, "_incident_queue", [])
        active_incident = getattr(drone, "_active_incident", None)
        return {
            "jid": str(drone.jid),
            "callsign": drone.callsign,
            "position": list(drone.position),
            "base": list(drone.base_position),
            "patrol_sector": list(drone._sector_bounds) if drone._sector_bounds else None,
            "status": status,
            "battery_pct": battery_pct,
            "incident_queue": len(queue),
            "active_incident": active_incident,
        }

    def _snapshot_sensor(self, sensor: SensorAgent) -> Dict[str, Any]:
        return {
            "jid": str(sensor.jid),
            "position": list(sensor.position),
            "radius": getattr(sensor, "detection_radius", None),
        }

    def _snapshot_tracker(self, tracker: AnimalTrackerAgent) -> Dict[str, Any]:
        goal = getattr(tracker, "_goal", None)
        return {
            "jid": str(tracker.jid),
            "animal_id": tracker.animal_id,
            "position": list(tracker.position),
            "goal": list(goal) if goal else None,
        }

    def _snapshot_poacher(self, poacher: Poacher) -> Dict[str, Any]:
        return {
            "id": poacher.id,
            "position": list(poacher.pos),
            "active": poacher.active,
        }

    def _snapshot_herd(self, herd: Herd) -> Dict[str, Any]:
        return {
            "id": herd.id,
            "center": list(herd.center),
            "size": herd.size,
            "goal": list(herd.migration_goal) if herd.migration_goal else None,
            "active": herd.active,
        }
