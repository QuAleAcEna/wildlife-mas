import random
import time
import uuid

from spade import agent
from spade.behaviour import PeriodicBehaviour

from core.env import Reserve
from core.messages import make_inform_alert


class SensorAgent(agent.Agent):
    """Simulated sensor that periodically raises anomalies to a target drone."""

    def __init__(self, jid: str, password: str, reserve: Reserve, target_drone: str):
        """Store references to the reserve map and the drone that handles alerts."""
        super().__init__(jid, password)
        self.reserve = reserve
        self.target_drone = target_drone

    class SenseAndAlert(PeriodicBehaviour):
        """Periodic behaviour that probabilistically emits anomaly alerts."""

        async def run(self):
            """Fire alerts at random to drive the demonstration flow."""
            if random.random() < 0.5:  # Elevated odds to keep the demo busy.
                cell = self.agent.reserve.random_cell()
                alert_id = f"{self.agent.jid}-{uuid.uuid4().hex}"
                payload = {
                    "sensor": str(self.agent.jid),
                    "id": alert_id,
                    "pos": cell,
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                    "ts": self.current_time(),
                }
                msg = make_inform_alert(self.agent.target_drone, payload)
                await self.send(msg)
                self.agent.log(f"ALERT -> {self.agent.target_drone} :: {payload}")

        def current_time(self):
            """Expose a small wrapper so behaviour remains test-friendly."""
            return int(time.time())

    async def setup(self):
        """Attach the periodic sensing behaviour to the agent."""
        self.log("Sensor starting…")
        behaviour = self.SenseAndAlert(period=1.0)  # every 1s for faster alert cadence
        self.add_behaviour(behaviour)

    def log(self, *args):
        """Emit tagged log messages for sensor events."""
        print("[SENSOR]", *args)
