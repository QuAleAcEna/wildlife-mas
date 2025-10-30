import asyncio
import json
import logging
import os
from typing import Any

import spade
from dotenv import load_dotenv

from agents.drone_agent import DroneAgent
from agents.sensor_agent import SensorAgent
from agents.ranger_agent import RangerAgent
from core.env import Reserve

load_dotenv()

SENSOR_JID = os.getenv("SENSOR_JID")
SENSOR_PASS = os.getenv("SENSOR_PASS")
DRONE_JID = os.getenv("DRONE_JID")
DRONE_PASS = os.getenv("DRONE_PASS")
RANGER_JID = os.getenv("RANGER_JID")
RANGER_PASS = os.getenv("RANGER_PASS")


def _require_env_vars() -> None:
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


async def main(args: Any = None) -> None:
    _require_env_vars()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reserve = Reserve()
    print("Reserve no-fly cells:", sorted(reserve.no_fly))
    reserve.clock.start()
    ranger = RangerAgent(RANGER_JID, RANGER_PASS, clock=reserve.clock)
    drone = DroneAgent(DRONE_JID, DRONE_PASS, ranger_jid=RANGER_JID, reserve=reserve)
    sensor = SensorAgent(SENSOR_JID, SENSOR_PASS, reserve, target_drone=DRONE_JID)

    await ranger.start(auto_register=True)
    await drone.start(auto_register=True)
    await sensor.start(auto_register=True)

    print("Agents running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await reserve.clock.stop()
        await sensor.stop()
        await drone.stop()
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
