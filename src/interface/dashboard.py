"""State writer that powers the dashboard JSON feed and historical exports."""

from __future__ import annotations

import asyncio
import json
import atexit
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

from agents.animaltracker_agent import AnimalTrackerAgent
from agents.drone_agent import DroneAgent
from agents.ranger_agent import RangerAgent
from agents.sensor_agent import SensorAgent
from core.env import Reserve
from core.events import Herd, Poacher, WorldEventEngine

Coord = Tuple[int, int]


def _safe_tuple(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    """Convert arbitrary sequences into 2D tuples when valid."""
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
    history_limit: int = 900  # ~15 min de histórico

    def __post_init__(self) -> None:
        """Create output directories and initialize internal caches."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._response_times: List[int] = []
        self._energy_log: List[Tuple[str, float]] = []
        self._visited_cells: Set[Tuple[int, int]] = set()
        self._history_file: Optional[Path] = None
        self._kpi_file: Optional[Path] = None
        self._exported_once = False
        clock = getattr(self.events, "clock", None) or getattr(self.reserve, "clock", None)
        self._clock_start_day = getattr(clock, "current_day", 1)
        self._clock_start_hour = getattr(clock, "current_hour", 0)
        self._seconds_per_hour = getattr(clock, "seconds_per_hour", 60.0) or 60.0
        self._clock_start_ts = datetime.utcnow()
        if clock and getattr(clock, "current_day", None) is not None:
            self._last_recorded_hour: Optional[Tuple[int, int]] = (clock.current_day, clock.current_hour)
        else:
            self._last_recorded_hour = None

    def record_response(self, steps: int) -> None:
        """Register the number of steps taken by the ranger to reach an alert."""
        if steps >= 0:
            self._response_times.append(steps)

    def record_energy(self, callsign: str, delta: float) -> None:
        """Track energy consumption deltas keyed by agent callsign."""
        self._energy_log.append((callsign, delta))

    async def run(self) -> None:
        """Main loop that serializes state snapshots at fixed cadence."""
        while True:
            data = self._build_snapshot()
            self.output_path.write_text(json.dumps(data, indent=2))
            if self._should_record_history(data):
                self._history.append(data)
                if len(self._history) > self.history_limit:
                    self._history.pop(0)
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # Snapshot assembly helpers
    # ------------------------------------------------------------------
    def _current_clock_state(self) -> Tuple[int, int, float]:
        """Determine the current simulated day/hour even if the clock isn't running."""
        clock = getattr(self.events, "clock", None) or getattr(self.reserve, "clock", None)
        if clock and getattr(clock, "current_day", None) is not None:
            day = getattr(clock, "current_day", self._clock_start_day)
            hour = getattr(clock, "current_hour", self._clock_start_hour)
            elapsed = getattr(clock, "total_hours_elapsed", 0.0)
            return day, hour, elapsed
        elapsed_seconds = (datetime.utcnow() - self._clock_start_ts).total_seconds()
        sim_hours = int(elapsed_seconds / self._seconds_per_hour)
        day = self._clock_start_day + (self._clock_start_hour + sim_hours) // 24
        hour = (self._clock_start_hour + sim_hours) % 24
        return day, hour, sim_hours

    def _build_snapshot(self) -> Dict[str, Any]:
        """Assemble the full snapshot dictionary consumed by the frontend."""
        now = datetime.utcnow().isoformat() + "Z"
        day, hour, elapsed_hours = self._current_clock_state()
        snapshot = {
            "generated_at": now,
            "clock": {
                "day": day,
                "hour": hour,
                "seconds_per_hour": self._seconds_per_hour,
                "total_hours_elapsed": elapsed_hours,
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
        snapshot["kpis"] = self._kpi_payload(snapshot)
        self._record_coverage(snapshot)
        return snapshot

    def _metrics_payload(self) -> Dict[str, Any]:
        """Collect lightweight counters for display on the dashboard."""
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

    def _record_coverage(self, snapshot: Dict[str, Any]) -> None:
        """Update the set of visited cells based on the latest snapshot."""
        for drone in snapshot["agents"]["drones"]:
            pos = drone.get("position")
            if isinstance(pos, list) and len(pos) == 2:
                self._visited_cells.add(tuple(pos))
        ranger_pos = snapshot["agents"]["ranger"].get("position")
        if isinstance(ranger_pos, list) and len(ranger_pos) == 2:
            self._visited_cells.add(tuple(ranger_pos))

    def _should_record_history(self, snapshot: Dict[str, Any]) -> bool:
        """Record at most one snapshot per simulated hour."""
        clock = snapshot.get("clock", {})
        key = (clock.get("day"), clock.get("hour"))
        if key != self._last_recorded_hour:
            self._last_recorded_hour = key
            return True
        return False

    def _kpi_payload(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Compute derived KPIs such as coverage and response time averages."""
        drones = snapshot["agents"]["drones"]
        avg_battery = (
            round(sum(d.get("battery_pct") or 0.0 for d in drones) / len(drones), 2)
            if drones
            else 0.0
        )
        mean_response = (
            round(sum(self._response_times) / len(self._response_times), 2)
            if self._response_times
            else 0.0
        )
        energy_per_threat = (
            round(sum(delta for _, delta in self._energy_log) / len(self._energy_log), 2)
            if self._energy_log
            else 0.0
        )
        coverage = (
            round((len(self._visited_cells) / (self.reserve.width * self.reserve.height)) * 100.0, 2)
            if self._visited_cells
            else 0.0
        )

        return {
            "alerts_total": snapshot["metrics"]["alerts_total"],
            "dispatch_total": snapshot["metrics"]["dispatch_total"],
            "mean_response_steps": mean_response,
            "avg_drone_battery_pct": avg_battery,
            "ranger_fuel": self.ranger.fuel_level,
            "energy_per_threat": energy_per_threat,
            "coverage_rate": coverage,
        }

    def export_history(self, path: Path) -> None:
        """Persist accumulated snapshots to disk as JSON."""
        if not self._history:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self._history, fh, indent=2)

    def export_kpis_csv(self, path: Path) -> None:
        """Persist KPI history to CSV for plotting."""
        import csv

        if not self._history:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "generated_at",
            "alerts_total",
            "dispatch_total",
            "mean_response_steps",
            "avg_drone_battery_pct",
            "ranger_fuel",
            "energy_per_threat",
            "poachers_active",
            "herds_active",
            "coverage_rate",
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._history:
                kpis = entry.get("kpis", {})
                writer.writerow(
                    {
                        "generated_at": entry.get("generated_at"),
                        "alerts_total": kpis.get("alerts_total"),
                        "dispatch_total": kpis.get("dispatch_total"),
                        "mean_response_steps": kpis.get("mean_response_steps"),
                        "avg_drone_battery_pct": kpis.get("avg_drone_battery_pct"),
                        "ranger_fuel": kpis.get("ranger_fuel"),
                        "energy_per_threat": kpis.get("energy_per_threat"),
                        "poachers_active": len(entry.get("events", {}).get("poachers", [])),
                        "herds_active": len(entry.get("events", {}).get("herds", [])),
                        "coverage_rate": kpis.get("coverage_rate"),
                    }
                )

    def register_export_paths(self, history_path: Path, csv_path: Path) -> None:
        """Register output files that will be flushed automatically on exit."""
        self._history_file = history_path
        self._kpi_file = csv_path
        atexit.register(self._export_at_exit)

    def _export_at_exit(self) -> None:
        """Write pending exports if they haven't been flushed yet."""
        if self._exported_once:
            return
        history_path = self._history_file
        csv_path = self._kpi_file
        if history_path:
            try:
                self.export_history(history_path)
            except Exception:
                pass
        if csv_path:
            try:
                self.export_kpis_csv(csv_path)
            except Exception:
                pass
        self._exported_once = True

    def _recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent alerts slimmed down to key fields."""
        alerts = self.ranger.alert_history[-limit:]
        return [self._slim_alert(a) for a in alerts]

    def _cnp_pending(self) -> List[Dict[str, Any]]:
        """Expose pending CNP negotiations to the dashboard."""
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
        """Normalize alert payloads irrespective of their origin."""
        payload = alert.copy()
        alert_block = payload.get("alert")
        if not isinstance(alert_block, dict):
            alert_block = {}
        category = payload.get("category") or alert_block.get("category") or "unknown"
        sensor_pos = (
            payload.get("sensor_pos")
            or payload.get("sensor_position")
            or alert_block.get("sensor_pos")
        )
        return {
            "id": payload.get("id") or alert_block.get("id"),
            "sensor": payload.get("sensor"),
            "category": category,
            "pos": payload.get("pos") or alert_block.get("pos"),
            "confidence": payload.get("confidence") or alert_block.get("confidence"),
            "timestamp": payload.get("ts")
            or payload.get("timestamp")
            or alert_block.get("ts")
            or alert_block.get("timestamp"),
            "sensor_pos": sensor_pos,
        }

    def _snapshot_ranger(self) -> Dict[str, Any]:
        """Serialize ranger state into a JSON-friendly structure."""
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
        """Serialize a single drone, including patrol metadata and queue depth."""
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
        """Serialize a ground sensor for frontend display."""
        return {
            "jid": str(sensor.jid),
            "position": list(sensor.position),
            "radius": getattr(sensor, "detection_radius", None),
        }

    def _snapshot_tracker(self, tracker: AnimalTrackerAgent) -> Dict[str, Any]:
        """Serialize a tracked animal including its next goal."""
        goal = getattr(tracker, "_goal", None)
        return {
            "jid": str(tracker.jid),
            "animal_id": tracker.animal_id,
            "position": list(tracker.position),
            "goal": list(goal) if goal else None,
        }

    def _snapshot_poacher(self, poacher: Poacher) -> Dict[str, Any]:
        """Serialize a poacher entity for the dashboard."""
        return {
            "id": poacher.id,
            "position": list(poacher.pos),
            "active": poacher.active,
        }

    def _snapshot_herd(self, herd: Herd) -> Dict[str, Any]:
        """Serialize a herd entity for the dashboard."""
        return {
            "id": herd.id,
            "center": list(herd.center),
            "size": herd.size,
            "goal": list(herd.migration_goal) if herd.migration_goal else None,
            "active": herd.active,
        }
