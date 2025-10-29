import asyncio
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

    reserve = Reserve()
    drone = DroneAgent(DRONE_JID, DRONE_PASS, ranger_jid=RANGER_JID)
    sensor = SensorAgent(SENSOR_JID, SENSOR_PASS, reserve, target_drone=DRONE_JID)
    ranger = RangerAgent(RANGER_JID, RANGER_PASS)

    await drone.start(auto_register=True)
    await sensor.start(auto_register=True)
    await ranger.start(auto_register=True)

    print("Agents running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await sensor.stop()
        await drone.stop()
        await ranger.stop()

if __name__ == "__main__":
    spade.run(main())
