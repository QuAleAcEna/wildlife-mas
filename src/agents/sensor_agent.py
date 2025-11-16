import random
import time
import uuid

from spade import agent
from spade.behaviour import PeriodicBehaviour

from core.env import Reserve
from core.messages import make_inform_alert

# NOVO #
# Parâmetros simples de deteção dos sensores
SENSOR_DETECTION_RADIUS = 5          # em células
SENSOR_MIN_DET_PROB = 0.2            # probabilidade mínima de deteção
SENSOR_POACHER_BASE_DET = 0.9        # base para caçadores
SENSOR_HERD_BASE_DET = 0.7           # base para bandos
SENSOR_FALLBACK_RANDOM_PROB = 0.03   # prob. de alerta aleatório quando nada é visto
# NOVO #


class SensorAgent(agent.Agent):
    """Simulated sensor that periodically raises anomalies to a target drone."""

    def __init__(self, jid: str, password: str, reserve: Reserve, target_drone: str):
        """Store references to the reserve map and the drone that handles alerts."""
        super().__init__(jid, password)
        self.reserve = reserve
        self.target_drone = target_drone
        # NOVO #
        # Cada sensor fica fixo numa célula do mapa (simula um sensor físico no terreno).
        self.position = self.reserve.random_cell()
        # Raio de deteção configurável
        self.detection_radius = SENSOR_DETECTION_RADIUS
        # NOVO #

    class SenseAndAlert(PeriodicBehaviour):  # sending alerts at random times, change to be based on movement, sound, etc
        """Periodic behaviour that probabilistically emits anomaly alerts."""

        async def run(self):
            """Fire alerts with base on nearby dynamic entities (poachers/herds)."""
            # NOVO #
            engine = getattr(self.agent.reserve, "events", None)

            # Primeiro tenta detetar entidades reais (poachers / herds)
            if engine is not None:
                nearby = engine.nearby_entities(
                    self.agent.position,
                    radius=self.agent.detection_radius,
                    kinds=("poacher", "herd"),
                )

                candidates = []
                for kind in ("poacher", "herd"):
                    for pos, dist, ent in nearby.get(kind, []):
                        candidates.append((kind, pos, dist))

                if candidates:
                    # Escolhe a entidade mais próxima
                    category, pos, dist = min(candidates, key=lambda t: t[2])
                    max_r = max(1, self.agent.detection_radius)

                    if category == "poacher":
                        base = SENSOR_POACHER_BASE_DET
                    else:
                        base = SENSOR_HERD_BASE_DET

                    # Probabilidade decresce com a distância, mas nunca abaixo do mínimo
                    prob = max(
                        SENSOR_MIN_DET_PROB,
                        base * (1.0 - (dist / max_r)),
                    )

                    if random.random() < prob:
                        alert_id = f"{self.agent.jid}-{uuid.uuid4().hex}"
                        # Confiança também decresce com a distância
                        confidence = 0.5 + 0.5 * (1.0 - dist / max_r)
                        confidence = round(max(0.0, min(1.0, confidence)), 2)

                        payload = {
                            "sensor": str(self.agent.jid),
                            "id": alert_id,
                            "pos": pos,
                            "confidence": confidence,
                            "ts": self.current_time(),
                            "category": category,              # "poacher" | "herd"
                            "distance_cells": dist,
                            "distance_m": dist * 100.0,        # simples escala p/ o drone usar se quiser
                            "sensor_pos": self.agent.position,
                        }
                        msg = make_inform_alert(self.agent.target_drone, payload)
                        await self.send(msg)
                        self.agent.log(
                            f"ALERT ({category}) -> {self.agent.target_drone} :: {payload}"
                        )
                        return  # só um alerta por tick

            # Se não houver motor de eventos ou nada perto → fallback aleatório (mantém demo viva)
            if random.random() < SENSOR_FALLBACK_RANDOM_PROB:
                cell = self.agent.reserve.random_cell()
                alert_id = f"{self.agent.jid}-{uuid.uuid4().hex}"
                payload = {
                    "sensor": str(self.agent.jid),
                    "id": alert_id,
                    "pos": cell,
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                    "ts": self.current_time(),
                    "category": "unknown",
                    "sensor_pos": self.agent.position,
                }
                msg = make_inform_alert(self.agent.target_drone, payload)
                await self.send(msg)
                self.agent.log(
                    f"ALERT (fallback) -> {self.agent.target_drone} :: {payload}"
                )
            # NOVO #

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
