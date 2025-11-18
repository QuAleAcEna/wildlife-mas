from __future__ import annotations

import asyncio
import logging
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional, Tuple

import spade
from agents.animaltracker_agent import AnimalTrackerAgent
from agents.drone_agent import DroneAgent
from agents.ranger_agent import RangerAgent
from agents.sensor_agent import SensorAgent, plan_sensor_grid
from core.env import EnvironmentClock, Reserve
from core.events import EventConfig, WorldEventEngine

try:  # Support running as script or module
    from .dashboard import DashboardStateWriter
except ImportError:  # pragma: no cover - best effort fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from interface.dashboard import DashboardStateWriter  # type: ignore
from run_all import (
    DRONE_JID,
    DRONE_PASS,
    RANGER_JID,
    RANGER_PASS,
    SENSOR_JID,
    SENSOR_PASS,
    _patch_pyjabber_handle_user,
    _patch_pyjabber_xml_protocol,
    _require_env_vars,
    _with_resource,
)


def _start_static_server(directory: Path, preferred_port: int = 8000) -> Tuple[ThreadingHTTPServer, threading.Thread, str]:
    """
    Launch a simple HTTP server that serves the dashboard static assets.

    Returns the server instance, its thread, and the URL to open.
    """

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    last_error: Optional[Exception] = None
    for port in (preferred_port, 0):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as exc:
            last_error = exc
            continue
    else:
        assert last_error is not None
        raise last_error

    thread = threading.Thread(target=server.serve_forever, name="DashboardHTTPServer", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    return server, thread, url


async def _run_events(engine: WorldEventEngine) -> None:
    """Run the world event engine in the background."""
    try:
        while True:
            engine.tick()
            await asyncio.sleep(engine.cfg.tick_seconds)
    except asyncio.CancelledError:
        pass


async def main(args: Any = None) -> None:
    """Start the MAS stack alongside the dashboard state writer."""
    _require_env_vars()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reserve = Reserve(clock=EnvironmentClock(seconds_per_hour=10.0))
    reserve.clock.start()

    events = WorldEventEngine(
        reserve=reserve,
        clock=reserve.clock,
        config=EventConfig(),
    )
    event_task = asyncio.create_task(_run_events(events), name="WorldEventEngine")

    ranger = RangerAgent(RANGER_JID, RANGER_PASS, clock=reserve.clock)
    ranger.reserve = reserve

    # --- Drones setup (mirrors run_all.py) ---
    drone_count = 4
    callsigns = ["1", "2", "3", "4"]
    half_w = max(1, reserve.width // 2)
    half_h = max(1, reserve.height // 2)
    sector_defs = [
        ((0, 0), (0, 0, half_w - 1, half_h - 1)),
        ((0, reserve.height - 1), (0, half_h, half_w - 1, reserve.height - 1)),
        ((reserve.width - 1, 0), (half_w, 0, reserve.width - 1, half_h - 1)),
        (
            (reserve.width - 1, reserve.height - 1),
            (half_w, half_h, reserve.width - 1, reserve.height - 1),
        ),
    ]
    drones: List[DroneAgent] = []
    for idx, (base_position, sector) in enumerate(sector_defs):
        jid = _with_resource(DRONE_JID, f"drone{idx + 1}")
        drone = DroneAgent(
            jid,
            DRONE_PASS,
            ranger_jid=RANGER_JID,
            reserve=reserve,
            base_position=base_position,
            patrol_sector=sector,
            patrol_seed=idx + 1,
            callsign=callsigns[idx],
        )
        drones.append(drone)

    def _nearest_drone_jid(cell: Tuple[int, int]) -> str:
        best = drones[0].jid
        best_dist = float("inf")
        for drone in drones:
            bx, by = drone.base_position
            dist = abs(bx - cell[0]) + abs(by - cell[1])
            if dist < best_dist:
                best_dist = dist
                best = drone.jid
        return best

    sensors: List[SensorAgent] = []
    placements = plan_sensor_grid(reserve)
    for idx, (position, bounds) in enumerate(placements):
        sensor_jid = _with_resource(SENSOR_JID, f"sensor{idx + 1}")
        sensor = SensorAgent(
            sensor_jid,
            SENSOR_PASS,
            reserve,
            target_drone=_nearest_drone_jid(position),
            target_ranger=RANGER_JID,
            position=position,
            coverage_bounds=bounds,
        )
        sensors.append(sensor)

    trackers: List[AnimalTrackerAgent] = []
    tracker_count = 2
    for idx in range(tracker_count):
        tracker_jid = _with_resource(DRONE_JID, f"tracker{idx + 1}")
        tracker = AnimalTrackerAgent(
            tracker_jid,
            DRONE_PASS,
            reserve=reserve,
            target_jid=RANGER_JID,
        )
        trackers.append(tracker)

    ranger.drone_jids = [d.jid for d in drones]

    await ranger.start(auto_register=True)
    for drone in drones:
        await drone.start(auto_register=True)
    for sensor in sensors:
        await sensor.start(auto_register=True)
    for tracker in trackers:
        await tracker.start(auto_register=True)

    output_path = Path(__file__).with_name("static").joinpath("state.json")
    writer = DashboardStateWriter(
        reserve=reserve,
        events=events,
        ranger=ranger,
        drones=drones,
        sensors=sensors,
        trackers=trackers,
        output_path=output_path,
        interval=1.0,
    )
    setattr(reserve, "dashboard_writer", writer)
    ranger.metrics_writer = writer
    reports_dir = Path(__file__).with_name("reports")
    writer.register_export_paths(
        reports_dir / "dashboard_history.json",
        reports_dir / "dashboard_metrics.csv",
    )
    writer_task = asyncio.create_task(writer.run(), name="DashboardStateWriter")

    print("Agents + dashboard writer running. Preparing dashboard web server…")

    http_server: Optional[ThreadingHTTPServer] = None
    http_thread: Optional[threading.Thread] = None
    static_dir = Path(__file__).with_name("static")
    try:
        http_server, http_thread, url = _start_static_server(static_dir, preferred_port=8000)
        print(f"Dashboard web UI available at {url}")
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception:
            print("Não consegui abrir o browser automaticamente; abre o link acima manualmente.")
    except OSError as exc:
        logging.warning("Could not start built-in HTTP server: %s", exc)
        print(f"Serve {static_dir} manually with `python3 -m http.server 8000`.")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        writer_task.cancel()
        try:
            await writer_task
        except asyncio.CancelledError:
            pass
        await reserve.clock.stop()
        if event_task:
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
        for sensor in sensors:
            await sensor.stop()
        for drone in drones:
            await drone.stop()
        for tracker in trackers:
            await tracker.stop()
        await ranger.stop()
        if http_server:
            http_server.shutdown()
            http_server.server_close()
        try:
            writer.export_history(reports_dir / "dashboard_history.json")
            writer.export_kpis_csv(reports_dir / "dashboard_metrics.csv")
            print(f"Dashboard metrics exported to {reports_dir}")
        except Exception as exc:
            logging.warning("Could not export dashboard metrics: %s", exc)


if __name__ == "__main__":
    _patch_pyjabber_handle_user()
    _patch_pyjabber_xml_protocol()
    spade.run(main(), embedded_xmpp_server=True)
