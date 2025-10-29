import random
import time
import uuid

from spade import agent
from spade.behaviour import PeriodicBehaviour

from core.env import Reserve
from core.messages import make_inform_alert


class SensorAgent(agent.Agent):
    def __init__(self, jid: str, password: str, reserve: Reserve, target_drone: str):
        super().__init__(jid, password)
        self.reserve = reserve
        self.target_drone = target_drone

    class SenseAndAlert(PeriodicBehaviour):
        async def run(self):
            # Simulate sensing with low probability for anomaly
            if random.random() < 0.15:  # ~15% tick anomaly
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
            return int(time.time())

    async def setup(self):
        self.log("Sensor starting…")
        behaviour = self.SenseAndAlert(period=2.0)  # every 2s
        self.add_behaviour(behaviour)

    def log(self, *args):
        print("[SENSOR]", *args)
