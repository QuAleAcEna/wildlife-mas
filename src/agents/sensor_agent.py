"""Passive ground sensor agent that watches the reserve and emits alerts."""

import time
import uuid
from typing import List, Optional, Tuple

from spade import agent
from spade.behaviour import PeriodicBehaviour

from core.env import Reserve
from core.messages import make_inform_alert

# NOVO #
# Parâmetros simples de deteção dos sensores
SENSOR_COVERAGE_SIZE = 5              # sensores cobrem blocos 5x5
SENSOR_MIN_DET_PROB = 0.2             # probabilidade mínima de deteção
SENSOR_POACHER_BASE_DET = 0.9         # base para caçadores
SENSOR_HERD_BASE_DET = 0.7            # base para bandos
# NOVO #

__all__ = [
    "SENSOR_COVERAGE_SIZE",
    "SENSOR_MIN_DET_PROB",
    "SENSOR_POACHER_BASE_DET",
    "SENSOR_HERD_BASE_DET",
    "plan_sensor_grid",
    "_distance_factor",
    "SensorAgent",
]


def plan_sensor_grid(reserve: Reserve, coverage_size: int = SENSOR_COVERAGE_SIZE) -> List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]]:
    """
    Gera posições/limites para sensores de forma a cobrir toda a reserva com blocos coverage_size x coverage_size.
    Retorna lista de tuplos (posição, bounds).
    """
    placements: List[Tuple[Tuple[int, int], Tuple[int, int, int, int]]] = []
    width, height = reserve.width, reserve.height
    for y_min in range(0, height, coverage_size):
        y_max = min(y_min + coverage_size - 1, height - 1)
        for x_min in range(0, width, coverage_size):
            x_max = min(x_min + coverage_size - 1, width - 1)
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            placements.append(((cx, cy), (x_min, y_min, x_max, y_max)))
    return placements


def _distance_factor(distance: float, detection_radius: int) -> float:
    """
    Fatoriza o impacto da distância na deteção.
    Ao estar mais perto (distância 0) devolve 1; ao atingir o limite, aproxima-se de 0.
    Usa uma curva quadrática suave para favorecer observações muito próximas.
    """
    if detection_radius <= 0:
        return 1.0
    closeness = max(0.0, 1.0 - (distance / (detection_radius + 1)))
    return closeness * closeness


class SensorAgent(agent.Agent):
    """Simulated sensor that periodically raises anomalies to a target drone."""

    def __init__(
        self,
        jid: str,
        password: str,
        reserve: Reserve,
        target_drone: str,
        position: Optional[Tuple[int, int]] = None,
        coverage_bounds: Optional[Tuple[int, int, int, int]] = None,
        target_ranger: Optional[str] = None,
    ):
        """Store references to the reserve map and the drone that handles alerts."""
        super().__init__(jid, password)
        self.reserve = reserve
        self.target_drone = target_drone
        self.target_ranger = target_ranger
        # NOVO #
        # Cada sensor fica fixo numa célula do mapa (simula um sensor físico no terreno).
        self.position = position or self.reserve.random_cell()
        self.coverage_bounds = coverage_bounds
        self.detection_radius = self._compute_detection_radius()
        # NOVO #

    def _compute_detection_radius(self) -> int:
        """Derive a Chebyshev-like coverage radius from bounds or defaults."""
        if not self.coverage_bounds:
            return SENSOR_COVERAGE_SIZE
        x_min, y_min, x_max, y_max = self.coverage_bounds
        corners = [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
        ]
        cx, cy = self.position
        radius = max(abs(cx - x) + abs(cy - y) for x, y in corners)
        return max(1, radius)

    def _in_coverage(self, cell: Tuple[int, int]) -> bool:
        """Return True if a grid cell is inside the sensor's coverage bounds."""
        if not self.coverage_bounds:
            return True
        x_min, y_min, x_max, y_max = self.coverage_bounds
        x, y = cell
        return x_min <= x <= x_max and y_min <= y <= y_max

    class SenseAndAlert(PeriodicBehaviour):
        """Periodic behaviour that emits anomaly alerts for nearby entities."""

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
                        if not self.agent._in_coverage(pos):
                            continue
                        candidates.append((kind, pos, dist))

                if candidates:
                    # Escolhe a entidade mais próxima
                    category, pos, dist = min(candidates, key=lambda t: t[2])
                    max_r = max(1, self.agent.detection_radius)

                    if category == "poacher":
                        base = SENSOR_POACHER_BASE_DET
                    else:
                        base = SENSOR_HERD_BASE_DET

                    # Probabilidade agora é sempre 1 dentro do raio; apenas confiança varia
                    distance_factor = _distance_factor(dist, max_r)
                    confidence = base * distance_factor
                    confidence = round(max(0.0, min(1.0, confidence)), 2)

                    alert_id = f"{self.agent.jid}-{uuid.uuid4().hex}"
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
                    recipient = self.agent.target_drone
                    route = "drone"
                    if (
                        category == "poacher"
                        and confidence >= 0.7
                        and self.agent.target_ranger
                    ):
                        recipient = self.agent.target_ranger
                        route = "ranger"
                    msg = make_inform_alert(recipient, payload)
                    await self.send(msg)
                    self.agent.log(
                        f"ALERT ({category}) -> {route} :: {payload}"
                    )
                    return  # só um alerta por tick


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
