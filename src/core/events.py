"""Stochastic world event engine that spawns poachers and herds within the reserve."""

# NOVO #
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Iterable, Dict

from core.env import Reserve, EnvironmentClock

__all__ = [
    "EventConfig",
    "Poacher",
    "Herd",
    "WorldEventEngine",
]

Coord = Tuple[int, int]


@dataclass
class EventConfig:
    """Parâmetros de alto nível do motor de eventos."""
    # Frequência do tick externo (apenas informativo; o agendamento é feito fora)
    tick_seconds: float = 1.0

    # Limites de entidades
    max_poachers: int = 3
    max_herds: int = 4

    # Probabilidades de spawn por tick (se abaixo do máximo)
    spawn_prob_poacher: float = 0.05
    spawn_prob_herd: float = 0.08

    # Dinâmica de movimento (células por tick)
    poacher_speed: int = 1
    herd_speed: int = 1

    # Migração de rebanhos
    herd_have_goal_prob: float = 0.7
    herd_goal_min_dist: int = 6
    herd_goal_max_dist: int = 12

    # Segurança / sanidade
    max_retries_relocate: int = 16


@dataclass
class Poacher:
    """Moving adversary that enters from the border, hunts, and attempts an exit."""

    id: str
    pos: Coord
    speed: int = 1
    active: bool = True
    target: Optional[Coord] = None
    exit_target: Optional[Coord] = None
    returning: bool = False
    move_cooldown: float = 0.0

    def __repr__(self) -> str:
        """Readable representation for debugging dashboards."""
        return (
            "Poacher("
            f"id={self.id}, pos={self.pos}, speed={self.speed}, active={self.active}, "
            f"target={self.target}, exit_target={self.exit_target}, returning={self.returning})"
        )


@dataclass
class Herd:
    """Group of animals migrating between goals while avoiding no-fly zones."""

    id: str
    center: Coord
    size: int = 8
    speed: int = 1
    migration_goal: Optional[Coord] = None
    active: bool = True

    def __repr__(self) -> str:
        """Readable representation for debugging dashboards."""
        return (
            f"Herd(id={self.id}, center={self.center}, size={self.size}, "
            f"speed={self.speed}, migration_goal={self.migration_goal}, active={self.active})"
        )


class WorldEventEngine:
    """
    Motor de eventos do ambiente: gere caçadores (poachers) e bandos (herds),
    atualiza as suas posições, assegura limites e evita no-fly zones.
    """

    def __init__(
        self,
        reserve: Reserve,
        clock: Optional[EnvironmentClock] = None,
        config: Optional[EventConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create a new engine bound to the given reserve and optional seed.

        Args:
            reserve (Reserve): Environment grid shared with agents.
            clock (EnvironmentClock | None): Simulation clock to derive pacing from.
            config (EventConfig | None): Behavioural settings; defaults to EventConfig().
            seed (int | None): Optional RNG seed for deterministic behaviour.
        """
        self.reserve = reserve
        self.clock = clock or reserve.clock
        self.cfg = config or EventConfig()
        self._rng = random.Random(seed)
        self.poachers: List[Poacher] = []
        self.herds: List[Herd] = []
        self._poacher_move_interval = getattr(self.clock, "seconds_per_hour", 10.0) or 10.0

        # Liga o engine ao reserve para fácil acesso a partir dos agentes (não quebra API existente)
        # Os agentes podem fazer: getattr(reserve, "events", None)
        setattr(self.reserve, "events", self)

    # ---------- API Pública ----------

    def tick(self) -> None:
        """Avança o estado do mundo um passo."""
        # Spawns
        self._maybe_spawn_poacher()
        self._maybe_spawn_herd()

        # Movimento
        for p in list(self.poachers):
            if p.active:
                p.move_cooldown = max(0.0, p.move_cooldown - self.cfg.tick_seconds)
                if p.move_cooldown > 0.0:
                    continue
                self._step_poacher(p)
                if p.active:
                    p.move_cooldown = self._poacher_move_interval

        for h in list(self.herds):
            if h.active:
                self._step_herd(h)

        # Limpeza (se algum for desativado)
        self.poachers = [p for p in self.poachers if p.active]
        self.herds = [h for h in self.herds if h.active]

    def nearby_entities(
        self,
        pos: Coord,
        radius: int,
        kinds: Iterable[str] = ("poacher", "herd"),
    ) -> Dict[str, List[Tuple[Coord, float, object]]]:
        """
        Consulta entidades próximas de uma posição.
        Retorna dicionário com listas de tuplos (posicao, distancia, entidade).
        """
        out: Dict[str, List[Tuple[Coord, float, object]]] = {"poacher": [], "herd": []}
        if "poacher" in kinds:
            for p in self.poachers:
                d = self._manhattan(pos, p.pos)
                if d <= radius:
                    out["poacher"].append((p.pos, d, p))
        if "herd" in kinds:
            for h in self.herds:
                d = self._manhattan(pos, h.center)
                if d <= radius:
                    out["herd"].append((h.center, d, h))
        return out

    # ---------- Spawns ----------

    def _maybe_spawn_poacher(self) -> None:
        """Probabilistically introduce a poacher entering from the border."""
        if len(self.poachers) >= self.cfg.max_poachers:
            return
        if self._rng.random() > self.cfg.spawn_prob_poacher:
            return
        pos = self._sample_border_cell()
        target = self._sample_goal_from(pos)
        exit_target = self._sample_border_cell()
        poacher = Poacher(
            id=f"poacher-{self._rng.getrandbits(32):08x}",
            pos=pos,
            speed=max(1, self.cfg.poacher_speed),
            target=target,
            exit_target=exit_target,
            move_cooldown=0.0,
        )
        self.poachers.append(poacher)

    def _maybe_spawn_herd(self) -> None:
        """Probabilistically introduce a new herd within the reserve interior."""
        if len(self.herds) >= self.cfg.max_herds:
            return
        if self._rng.random() > self.cfg.spawn_prob_herd:
            return
        center = self._sample_free_cell()
        goal = None
        if self._rng.random() < self.cfg.herd_have_goal_prob:
            goal = self._sample_goal_from(center)
        size = self._rng.randint(5, 12)
        herd = Herd(
            id=f"herd-{self._rng.getrandbits(32):08x}",
            center=center,
            size=size,
            speed=max(1, self.cfg.herd_speed),
            migration_goal=goal,
        )
        self.herds.append(herd)

    # ---------- Movimento ----------

    def _step_poacher(self, p: Poacher) -> None:
        """
        Poacher percorre um objetivo interno e depois regressa à borda.
        Cada passo respeita as no-fly zones para evitar soft-locks.
        """
        if not p.returning and p.target is None:
            p.target = self._sample_goal_from(p.pos)
        if p.returning and p.exit_target is None:
            p.exit_target = self._sample_border_cell()

        destination = p.exit_target if p.returning else p.target
        if destination is None:
            p.active = False
            return

        moved = self._move_towards(p, destination)
        if not moved:
            p.active = False
            return

        if p.pos == destination:
            if not p.returning:
                # Objetivo interno alcançado → apontar para saída
                p.returning = True
            else:
                # Chegou à borda → sai da reserva
                p.active = False

    def _step_herd(self, h: Herd) -> None:
        """Herd migra suavemente para o objetivo; se não houver, random-walk suave."""
        target = h.migration_goal
        if target is None:
            # Sem objetivo → random-walk leve
            for _ in range(self.cfg.max_retries_relocate):
                nx, ny = self._random_neighbor(h.center, step=h.speed, inward_bias=False)
                npos = self._clamp((nx, ny))
                if not self.reserve.is_no_fly(npos):
                    h.center = npos
                    return
            h.active = False
            return

        # Com objetivo → mover 1 célula na direção do objetivo, evitando no-fly
        cx, cy = h.center
        tx, ty = target
        dx = 0 if cx == tx else (1 if tx > cx else -1)
        dy = 0 if cy == ty else (1 if ty > cy else -1)

        # Permite passo diagonal em dois ticks (prioriza eixo mais distante)
        if abs(tx - cx) >= abs(ty - cy):
            trial = (cx + dx * h.speed, cy)
        else:
            trial = (cx, cy + dy * h.speed)

        trial = self._clamp(trial)
        if self.reserve.is_no_fly(trial):
            # Contorna: tenta vizinhos ortogonais
            candidates = [
                (cx + dx * h.speed, cy),
                (cx - dx * h.speed, cy),
                (cx, cy + dy * h.speed),
                (cx, cy - dy * h.speed),
            ]
            self._rng.shuffle(candidates)
            moved = False
            for cand in candidates:
                cand = self._clamp(cand)
                if not self.reserve.is_no_fly(cand):
                    h.center = cand
                    moved = True
                    break
            if not moved:
                # Último recurso: pequeno random-walk
                for _ in range(self.cfg.max_retries_relocate):
                    nx, ny = self._random_neighbor(h.center, step=1, inward_bias=False)
                    npos = self._clamp((nx, ny))
                    if not self.reserve.is_no_fly(npos):
                        h.center = npos
                        moved = True
                        break
                if not moved:
                    h.active = False
        else:
            h.center = trial

        # Chegada ao objetivo → escolhe novo objetivo distante
        if h.active and h.migration_goal and h.center == h.migration_goal:
            h.migration_goal = self._sample_goal_from(h.center)

    # ---------- Utilitários ----------

    def _sample_free_cell(self, edge_bias: bool = False) -> Coord:
        """
        Amostra uma célula livre (não no-fly). Se edge_bias=True, favorece bordas
        do mapa (útil para spawn de poachers a entrarem na reserva).
        """
        w, h = self.reserve.width, self.reserve.height
        for _ in range(self.cfg.max_retries_relocate):
            if edge_bias and self._rng.random() < 0.75:
                # Escolhe uma das bordas
                side = self._rng.choice(("top", "bottom", "left", "right"))
                if side in ("top", "bottom"):
                    x = self._rng.randrange(w)
                    y = 0 if side == "top" else (h - 1)
                else:
                    y = self._rng.randrange(h)
                    x = 0 if side == "left" else (w - 1)
            else:
                x = self._rng.randrange(w)
                y = self._rng.randrange(h)
            cell = (x, y)
            if not self.reserve.is_no_fly(cell):
                return cell
        # fallback (muito improvável): origem
        return (0, 0)

    def _sample_goal_from(self, origin: Coord) -> Coord:
        """Escolhe um objetivo a uma distância razoável do ponto de origem."""
        min_d = self.cfg.herd_goal_min_dist
        max_d = self.cfg.herd_goal_max_dist
        for _ in range(self.cfg.max_retries_relocate):
            theta = self._rng.random() * 2.0 * math.pi
            r = self._rng.randint(min_d, max_d)
            ox, oy = origin
            gx = ox + int(round(math.cos(theta) * r))
            gy = oy + int(round(math.sin(theta) * r))
            goal = self._clamp((gx, gy))
            if not self.reserve.is_no_fly(goal):
                return goal
        # Se falhar repetidamente, cai no centro do mapa
        return self._clamp((self.reserve.width // 2, self.reserve.height // 2))

    def _random_neighbor(self, pos: Coord, step: int = 1, inward_bias: bool = False) -> Coord:
        """Vizinhança ortogonal aleatória; com viés para o interior se pedido."""
        x, y = pos
        candidates = [
            (x + step, y),
            (x - step, y),
            (x, y + step),
            (x, y - step),
        ]
        if inward_bias:
            def edge_dist(c: Coord) -> int:
                """Return the distance of candidate `c` from the map edges."""
                cx, cy = self._clamp(c)
                return min(cx, self.reserve.width - 1 - cx, cy, self.reserve.height - 1 - cy)
            candidates.sort(key=edge_dist, reverse=True)
        else:
            self._rng.shuffle(candidates)
        return candidates[0]

    def _move_towards(self, poacher: Poacher, target: Coord) -> bool:
        """Move poacher um passo em direção ao target contornando no-fly zones."""
        cx, cy = poacher.pos
        tx, ty = target
        dx = 0 if cx == tx else (1 if tx > cx else -1)
        dy = 0 if cy == ty else (1 if ty > cy else -1)

        primary_axis_is_x = abs(tx - cx) >= abs(ty - cy)
        candidates = []
        if primary_axis_is_x:
            candidates.append((cx + dx * poacher.speed, cy))
            candidates.append((cx, cy + dy * poacher.speed))
        else:
            candidates.append((cx, cy + dy * poacher.speed))
            candidates.append((cx + dx * poacher.speed, cy))

        # Complementar: tenta diagonais simples e pequenos desvios
        candidates.extend(
            [
                (cx + dx * poacher.speed, cy + dy * poacher.speed),
                (cx - dx * poacher.speed, cy),
                (cx, cy - dy * poacher.speed),
            ]
        )
        for cand in candidates:
            cand = self._clamp(cand)
            if not self.reserve.is_no_fly(cand):
                poacher.pos = cand
                return True
        return False

    def _clamp(self, pos: Coord) -> Coord:
        """Ensure coordinates remain within the reserve bounds."""
        x = max(0, min(self.reserve.width - 1, pos[0]))
        y = max(0, min(self.reserve.height - 1, pos[1]))
        return (x, y)

    @staticmethod
    def _manhattan(a: Coord, b: Coord) -> int:
        """Return the Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _sample_border_cell(self) -> Coord:
        """Escolhe sempre uma célula na borda da reserva."""
        w, h = self.reserve.width, self.reserve.height
        for _ in range(self.cfg.max_retries_relocate):
            if self._rng.random() < 0.5:
                x = self._rng.randrange(w)
                y = 0 if self._rng.random() < 0.5 else (h - 1)
            else:
                y = self._rng.randrange(h)
                x = 0 if self._rng.random() < 0.5 else (w - 1)
            cell = (x, y)
            if not self.reserve.is_no_fly(cell):
                return cell
        return (0, 0)

