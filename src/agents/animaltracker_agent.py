"""Agent that simulates a collared animal broadcasting telemetry to the ranger."""

# NOVO #
from __future__ import annotations

import datetime as dt
import random
from typing import Tuple

from spade import agent
from spade.behaviour import PeriodicBehaviour
from spade.message import Message

from core.env import Reserve
from core.messages import INFORM, TELEMETRY, json_dumps


Coord = Tuple[int, int]


class AnimalTrackerAgent(agent.Agent):
    """
    Agente que simula um animal com colar de tracking (GPS).
    Move-se no mapa e envia periodicamente a sua posição.
    """

    def __init__(self, jid: str, password: str, reserve: Reserve, target_jid: str):
        """Configure the mock animal tracker with destinations and recipients.

        Args:
            jid (str): SPADE identifier for the tracker agent.
            password (str): SPADE password for authentication.
            reserve (Reserve): Map providing bounds and no-fly zones.
            target_jid (str): Recipient (usually the ranger) for telemetry.
        """
        super().__init__(jid, password)
        self.reserve = reserve
        self.target_jid = target_jid  # tipicamente o Ranger
        # Identificador lógico do animal (podes usar outra convenção se quiseres)
        self.animal_id = str(jid)
        # Estado de movimento
        self.position: Coord = self.reserve.random_cell()
        self._goal: Coord = self._sample_goal()
        self._speed: int = 1
        self._rng = random.Random()

    def _sample_goal(self) -> Coord:
        """Escolhe um novo objetivo aleatório dentro da reserva, fora de no-fly."""
        for _ in range(32):
            cell = self.reserve.random_cell()
            if not self.reserve.is_no_fly(cell):
                return cell
        # fallback improvável
        return (0, 0)

    def _step_towards_goal(self) -> None:
        """Move uma célula na direção do objetivo, evitando no-fly se possível."""
        cx, cy = self.position
        gx, gy = self._goal

        if (cx, cy) == (gx, gy):
            # Já chegou, escolher novo objetivo
            self._goal = self._sample_goal()
            gx, gy = self._goal

        dx = 0 if cx == gx else (1 if gx > cx else -1)
        dy = 0 if cy == gy else (1 if gy > cy else -1)

        # Decide se anda em X ou em Y neste tick (alternando um bocado)
        if self._rng.random() < 0.5:
            trial = (cx + dx * self._speed, cy)
        else:
            trial = (cx, cy + dy * self._speed)

        trial = self._clamp(trial)
        if self.reserve.is_no_fly(trial):
            # Tenta vizinhos ortogonais
            candidates = [
                (cx + dx * self._speed, cy),
                (cx - dx * self._speed, cy),
                (cx, cy + dy * self._speed),
                (cx, cy - dy * self._speed),
            ]
            self._rng.shuffle(candidates)
            for cand in candidates:
                cand = self._clamp(cand)
                if not self.reserve.is_no_fly(cand):
                    self.position = cand
                    return
            # Se não conseguir, fica parado neste tick
            return
        else:
            self.position = trial

    def _clamp(self, pos: Coord) -> Coord:
        """Keep coordinates inside reserve boundaries."""
        x = max(0, min(self.reserve.width - 1, pos[0]))
        y = max(0, min(self.reserve.height - 1, pos[1]))
        return (x, y)

    class MigrateAndReport(PeriodicBehaviour):
        """Comportamento periódico: mover o animal e reportar posição."""

        async def run(self):
            """Move the animal, compose telemetry, and send it to the ranger."""
            # Atualiza posição
            self.agent._step_towards_goal()

            x, y = self.agent.position
            gx, gy = self.agent._goal

            payload = {
                "animal": str(self.agent.jid),
                "animal_id": self.agent.animal_id,
                "position": {"x": x, "y": y},
                "goal": {"x": gx, "y": gy},
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
                "source": "animal_tracker",
            }

            msg = Message(to=self.agent.target_jid)
            msg.set_metadata("performative", INFORM)
            msg.set_metadata("type", TELEMETRY)
            msg.body = json_dumps(payload)
            await self.send(msg)

            self.agent.log(
                f"Telemetry -> {self.agent.target_jid} :: pos={self.agent.position} goal={self.agent._goal}"
            )

    async def setup(self):
        """Start the periodic migration behaviour."""
        self.log("Animal tracker starting… at", self.position, "goal", self._goal)
        behaviour = self.MigrateAndReport(period=3.0)  # a cada 3 segundos
        self.add_behaviour(behaviour)

    def log(self, *args):
        """Emit helper logs with a consistent [ANIMAL] prefix."""
        print("[ANIMAL]", *args)
# NOVO #
