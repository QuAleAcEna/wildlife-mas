"""Standalone scenario simulator used to validate KPI trends."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.env import Reserve
from core.events import EventConfig, WorldEventEngine


Coord = Tuple[int, int]

SENSOR_POSITIONS: Tuple[Coord, ...] = ((2, 2), (17, 2), (2, 17), (17, 17))
SENSOR_RADIUS = 4
DRONE_SENSOR_RADIUS = 2


@dataclass
class Scenario:
    """Configuration bundle for a deterministic simulation run."""

    name: str
    description: str
    ticks: int
    seed: int
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatRecord:
    """Record describing a spawned threat along with detection timestamps."""

    id: str
    kind: str
    spawn_tick: int
    pos: Coord
    detected_tick: Optional[int] = None
    responded_tick: Optional[int] = None


@dataclass
class PatrolDrone:
    """Lightweight patrol drone used for deterministic simulations."""

    name: str
    route: List[Coord]
    idx: int = 0
    position: Coord = (0, 0)
    energy_used: int = 0

    def __post_init__(self) -> None:
        """Initialize the starting position from the chosen route index."""
        self.position = self.route[self.idx]

    def step(self) -> None:
        """Advance to the next waypoint and accumulate energy usage."""
        self.idx = (self.idx + 1) % len(self.route)
        self.position = self.route[self.idx]
        self.energy_used += 1


def _build_config(overrides: Dict[str, Any]) -> EventConfig:
    """Create an EventConfig copy applying CLI overrides."""
    cfg = EventConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown EventConfig field '{key}'")
        setattr(cfg, key, value)
    return cfg


def _manhattan(a: Coord, b: Coord) -> int:
    """Return the Manhattan distance between two coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _build_patrol_route(width: int, height: int) -> List[Coord]:
    """Generate a serpentine route that visits every cell of the grid."""
    route: List[Coord] = []
    for y in range(height):
        xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        for x in xs:
            route.append((x, y))
    return route


def _spawn_drones(reserve: Reserve, count: int = 3) -> List[PatrolDrone]:
    """Instantiate evenly spaced PatrolDrone objects over the shared route."""
    base_route = _build_patrol_route(reserve.width, reserve.height)
    spacing = max(1, len(base_route) // count)
    drones: List[PatrolDrone] = []
    for i in range(count):
        idx = (i * spacing) % len(base_route)
        drones.append(PatrolDrone(name=f"D{i + 1}", route=base_route, idx=idx))
    return drones


def _default_scenarios(ticks: int) -> Iterable[Scenario]:
    """Yield canonical scenarios covering baseline, single, and mixed threats."""
    return (
        Scenario(
            name="patrol_only",
            description="Baseline patrol without poachers (coverage + energy metrics).",
            ticks=ticks,
            seed=5,
            overrides={
                "max_poachers": 0,
                "spawn_prob_poacher": 0.0,
                "max_herds": 0,
                "spawn_prob_herd": 0.0,
            },
        ),
        Scenario(
            name="single_poacher",
            description="One poacher at a time; validates detection and response latency.",
            ticks=ticks,
            seed=17,
            overrides={
                "max_poachers": 1,
                "spawn_prob_poacher": 0.25,
                "max_herds": 0,
                "spawn_prob_herd": 0.0,
            },
        ),
        Scenario(
            name="stress_mixed",
            description="Multiple poachers + herds competing for limited coverage.",
            ticks=ticks,
            seed=47,
            overrides={
                "max_poachers": 4,
                "spawn_prob_poacher": 0.35,
                "max_herds": 4,
                "spawn_prob_herd": 0.3,
            },
        ),
    )


def _is_detected(threat: ThreatRecord, drones: List[PatrolDrone]) -> bool:
    """Check if a threat is within range of any static sensor or drone."""
    for sensor in SENSOR_POSITIONS:
        if _manhattan(sensor, threat.pos) <= SENSOR_RADIUS:
            return True
    return any(_manhattan(drone.position, threat.pos) <= DRONE_SENSOR_RADIUS for drone in drones)


def _run_single_scenario(scenario: Scenario) -> Dict[str, Any]:
    """Run a scenario to completion and compute summary KPIs."""
    reserve = Reserve()
    config = _build_config(scenario.overrides)
    engine = WorldEventEngine(reserve=reserve, config=config, seed=scenario.seed)

    drones = _spawn_drones(reserve, count=3)
    visited_cells: set[Coord] = set(drone.position for drone in drones)
    threats: Dict[str, ThreatRecord] = {}
    herd_ids: Dict[str, bool] = {}

    proposals_total = 0
    proposals_accepted = 0
    response_delays: List[float] = []
    energy_spent = 0

    for tick in range(1, scenario.ticks + 1):
        engine.tick()

        for drone in drones:
            drone.step()
            visited_cells.add(drone.position)

        for poacher in engine.poachers:
            if poacher.id not in threats:
                threats[poacher.id] = ThreatRecord(
                    id=poacher.id,
                    kind="poacher",
                    spawn_tick=tick,
                    pos=poacher.pos,
                )

        for herd in engine.herds:
            if herd.id not in herd_ids:
                herd_ids[herd.id] = herd.active
            else:
                herd_ids[herd.id] = herd_ids[herd.id] and herd.active

        for threat in threats.values():
            if threat.detected_tick is None and _is_detected(threat, drones):
                threat.detected_tick = tick
                proposals_total += len(drones)
                proposals_accepted += 1

            if threat.detected_tick is not None and threat.responded_tick is None:
                nearest = min(_manhattan(drone.position, threat.pos) for drone in drones)
                threat.responded_tick = threat.detected_tick + nearest
                response_delays.append(nearest)
                energy_spent += nearest

    total_cells = reserve.width * reserve.height
    coverage_rate = round((len(visited_cells) / total_cells) * 100.0, 2)

    poacher_records = [t for t in threats.values() if t.kind == "poacher"]
    detected = [t for t in poacher_records if t.detected_tick is not None]
    responded = [t for t in poacher_records if t.responded_tick is not None]

    detection_rate = round((len(detected) / max(1, len(poacher_records))) * 100.0, 2)
    mean_response = round(statistics.mean(response_delays), 2) if response_delays else 0.0
    energy_per_threat = round(energy_spent / max(1, len(responded)), 2)
    task_efficiency = round((proposals_accepted / max(1, proposals_total)) * 100.0, 2)

    herd_spawned = len(herd_ids)
    herd_losses = len([hid for hid, alive in herd_ids.items() if not alive])
    herd_safety = round(100.0 - (herd_losses / max(1, herd_spawned)) * 100.0, 2)

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "ticks": scenario.ticks,
        "poachers_spawned": len(poacher_records),
        "threat_detection_rate": detection_rate,
        "mean_response_time": mean_response,
        "coverage_rate": coverage_rate,
        "energy_per_threat": energy_per_threat,
        "task_allocation_efficiency": task_efficiency,
        "herds_spawned": herd_spawned,
        "herd_safety_pct": herd_safety,
    }


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write aggregated KPI rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "description",
        "ticks",
        "poachers_spawned",
        "threat_detection_rate",
        "mean_response_time",
        "coverage_rate",
        "energy_per_threat",
        "task_allocation_efficiency",
        "herds_spawned",
        "herd_safety_pct",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Create a bar chart comparing detection and coverage per scenario."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib não disponível – a plot não será gerada.")
        return

    names = [row["scenario"] for row in rows]
    detection = [row["threat_detection_rate"] for row in rows]
    coverage = [row["coverage_rate"] for row in rows]

    x = range(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([xi - width / 2 for xi in x], detection, width, label="Detection (%)")
    ax.bar([xi + width / 2 for xi in x], coverage, width, label="Coverage (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage")
    ax.set_title("Scenario KPIs")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Gráfico guardado em {output_path}")


def main() -> None:
    """CLI entry point for orchestrating the scenario experiments."""
    parser = argparse.ArgumentParser(
        description="Run KPI-oriented scenarios (patrol baseline, single poacher, stress test) and export metrics."
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=240,
        help="Number of ticks to simulate per scenario (default: 240).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_scenarios.csv"),
        help="Destination CSV path (default: ./test_scenarios.csv).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Optional path to export a bar chart comparing detection and coverage rates.",
    )
    args = parser.parse_args()

    scenarios = list(_default_scenarios(args.ticks))
    results = [_run_single_scenario(s) for s in scenarios]

    _write_csv(results, args.output)
    print(f"Saved {len(results)} scenario summaries to {args.output}")
    for row in results:
        print(
            f"- {row['scenario']}: detection {row['threat_detection_rate']}%, "
            f"response {row['mean_response_time']} ticks, coverage {row['coverage_rate']}%"
        )

    if args.plot:
        _export_plot(results, args.plot)


if __name__ == "__main__":
    main()
