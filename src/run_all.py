"""Utility script that boots every agent for end-to-end simulations."""

import asyncio
import json
import logging
import os
from typing import Any, List, Tuple

import spade
from dotenv import load_dotenv

from agents.drone_agent import DroneAgent
from agents.sensor_agent import SensorAgent, plan_sensor_grid
from agents.ranger_agent import RangerAgent
from core.env import EnvironmentClock, Reserve
# NOVO #
from core.events import WorldEventEngine, EventConfig
from agents.animaltracker_agent import AnimalTrackerAgent
# NOVO #

load_dotenv()

SENSOR_JID = os.getenv("SENSOR_JID")
SENSOR_PASS = os.getenv("SENSOR_PASS")
DRONE_JID = os.getenv("DRONE_JID")
DRONE_PASS = os.getenv("DRONE_PASS")
RANGER_JID = os.getenv("RANGER_JID")
RANGER_PASS = os.getenv("RANGER_PASS")


def _with_resource(jid: str, resource: str) -> str:
    """Append/replace the resource part of a JID for multi-session use."""
    if "/" in jid:
        base, _ = jid.split("/", 1)
        return f"{base}/{resource}"
    return f"{jid}/{resource}"


def _pick_sector_base(reserve: Reserve, sector: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Choose a deterministic base cell within the patrol sector that is not no-fly."""
    x_min, y_min, x_max, y_max = sector
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if not reserve.is_no_fly((x, y)):
                return (x, y)
    # Fall back to sector origin even if it overlaps a no-fly area to avoid crashes.
    return (x_min, y_min)


def _require_env_vars() -> None:
    """Ensure the credentials required for SPADE login are available."""
    missing = [
        name
        for name, value in {
            "SENSOR_JID": SENSOR_JID,
            "SENSOR_PASS": SENSOR_PASS,
            "DRONE_JID": DRONE_JID,
            "DRONE_PASS": DRONE_PASS,
            "RANGER_JID": RANGER_JID,
            "RANGER_PASS": RANGER_PASS,
        }.items()
        if not value
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_str}")

# NOVO #
async def _run_events(engine: WorldEventEngine) -> None:
    """
    Tarefa assíncrona que avança o estado do motor de eventos.
    Usa o tick_seconds definido na configuração do engine.
    """
    try:
        while True:
            engine.tick()
            await asyncio.sleep(engine.cfg.tick_seconds)
    except asyncio.CancelledError:
        # Saída limpa quando o main desliga.
        pass
# NOVO #


async def main(args: Any = None) -> None:
    """Boot the simulated reserve, agents, and keep the event loop alive."""
    _require_env_vars()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reserve = Reserve(clock=EnvironmentClock(seconds_per_hour=10.0))
    print("Reserve no-fly cells:", sorted(reserve.no_fly))
    # Start the shared simulation clock before agents query it.
    reserve.clock.start()

    # NOVO #
    # Instanciar o motor de eventos (poachers + herds) ligado à mesma reserva/clock.
    events = WorldEventEngine(
        reserve=reserve,
        clock=reserve.clock,
        config=EventConfig(),  # podes ajustar aqui os parâmetros default
    )
    # Arrancar tarefa que faz tick do motor de eventos em background.
    event_task = asyncio.create_task(_run_events(events), name="WorldEventEngine")
    # NOVO #

    ranger = RangerAgent(RANGER_JID, RANGER_PASS, clock=reserve.clock)
    ranger.reserve = reserve
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
        logging.info(
            "Assigning drone %s (callsign %s) to sector %s with base %s",
            jid,
            callsigns[idx],
            sector,
            base_position,
        )
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
        """Pick the closest drone (by base) for a sensor placement."""
        best_jid = drones[0].jid
        best_dist = float("inf")
        for drone in drones:
            bx, by = drone.base_position
            dist = abs(bx - cell[0]) + abs(by - cell[1])
            if dist < best_dist:
                best_dist = dist
                best_jid = drone.jid
        return best_jid

    sensors = []
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

    # NOVO #
    # Registar os drones no Ranger para que ele possa lançar o CNP.
    ranger.drone_jids = [d.jid for d in drones]
    # NOVO #

    # NOVO #
    # Criar alguns animais com colar de tracking que reportam direto ao Ranger.
    trackers: List[AnimalTrackerAgent] = []
    tracker_count = 2
    for idx in range(tracker_count):
        tracker_jid = _with_resource(DRONE_JID, f"tracker{idx + 1}")
        tracker = AnimalTrackerAgent(
            tracker_jid,
            DRONE_PASS,    # reutilizamos a mesma password/base JID que os drones
            reserve=reserve,
            target_jid=RANGER_JID,
        )
        trackers.append(tracker)
    # NOVO #

    await ranger.start(auto_register=True)
    for drone in drones:
        await drone.start(auto_register=True)
    for sensor in sensors:
        await sensor.start(auto_register=True)
    # NOVO #
    for tracker in trackers:
        await tracker.start(auto_register=True)
    # NOVO #

    print("Agents running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await reserve.clock.stop()
        # NOVO #
        # Parar task do motor de eventos antes de desligar agentes.
        if event_task:
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass
        # NOVO #
        for sensor in sensors:
            await sensor.stop()
        for drone in drones:
            await drone.stop()
        # NOVO #
        for tracker in trackers:
            await tracker.stop()
        # NOVO #
        await ranger.stop()


# Run against SPADE's embedded XMPP server to skip external authentication.
def _patch_pyjabber_handle_user() -> None:
    """Patch pyjabber admin endpoint for SQLAlchemy 2.x compatibility."""
    try:
        from sqlalchemy import text

        from pyjabber.webpage.api import api as api_module
    except Exception:
        return

    if getattr(api_module.handleUser, "__patched__", False):
        return

    DB = api_module.DB  # reuse module singletons
    web = api_module.web

    async def handle_user(request):  # pragma: no cover - runtime side effect
        """Serve the embedded SPADE credential table for the dashboard UI."""
        with DB.connection() as con:
            result = con.execute(text("SELECT id, jid FROM credentials"))
            rows = result.fetchall()
        users = [{"id": row[0], "jid": row[1]} for row in rows]
        return web.Response(text=json.dumps(users))

    handle_user.__patched__ = True  # type: ignore[attr-defined]
    api_module.handleUser = handle_user


def _patch_pyjabber_xml_protocol() -> None:
    """Mute noisy SAXParseException logs caused by non-XMPP probes."""
    try:
        from loguru import logger
        from xml import sax

        from pyjabber.network import XMLProtocol as xml_protocol_module
    except Exception:
        return

    if getattr(xml_protocol_module.XMLProtocol.data_received, "__patched__", False):
        return

    original = xml_protocol_module.XMLProtocol.data_received

    def data_received(self, data):  # pragma: no cover - runtime side effect
        """Ignore malformed probes while keeping normal XMPP traffic intact."""
        try:
            return original(self, data)
        except sax.SAXParseException as exc:
            peer = getattr(self, "_peer", "unknown")
            logger.warning("Dropped non-XMPP payload from {}: {}", peer, exc)
            transport = getattr(self, "_transport", None)
            if transport:
                try:
                    transport.close()
                except Exception:
                    pass

    data_received.__patched__ = True  # type: ignore[attr-defined]
    xml_protocol_module.XMLProtocol.data_received = data_received


if __name__ == "__main__":
    _patch_pyjabber_handle_user()
    _patch_pyjabber_xml_protocol()

    spade.run(main(), embedded_xmpp_server=True)
