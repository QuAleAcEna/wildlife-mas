"""Shared simulation primitives such as the global clock and reserve map."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class EnvironmentClock:
    """Async clock that advances simulated days and hours at a configurable rate."""

    def __init__(
        self,
        seconds_per_hour: float = 60.0,
        start_day: int = 1,
        start_hour: int = 8,
    ) -> None:
        """Create a new clock anchored at the provided day/hour."""
        self.seconds_per_hour = seconds_per_hour
        self._start_day = start_day
        self._start_hour = start_hour
        self.total_hours_elapsed: int = 0
        self.current_day: int = start_day
        self.current_hour: int = start_hour
        self._task: Optional[asyncio.Task[None]] = None
        self._logger = logging.getLogger("environment.clock")

    def start(self) -> None:
        """Begin ticking by scheduling the internal asyncio task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="EnvironmentClock")

    async def stop(self) -> None:
        """Stop the background task and wait for it to cancel cleanly."""
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        """Sleep for the configured interval and advance time forever."""
        while True:
            await asyncio.sleep(self.seconds_per_hour)
            self._advance_time()

    def _advance_time(self) -> None:
        """Increment hours/days while logging the simulated instant."""
        self.total_hours_elapsed += 1
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        self._logger.info(
            "Environment time Day %s %02d:00",
            self.current_day,
            self.current_hour,
        )

    def reset(self) -> None:
        """Reset the clock to its initial state."""
        self.total_hours_elapsed = 0
        self.current_day = self._start_day
        self.current_hour = self._start_hour


@dataclass
class Reserve:
    """Grid representation of the reserve with random no-fly blocks."""

    width: int = 20
    height: int = 20
    no_fly: List[Tuple[int, int]] = None
    clock: EnvironmentClock = field(default_factory=EnvironmentClock)

    def __post_init__(self) -> None:
        """Populate a pseudo-random rectangular no-fly area if none provided."""
        if self.no_fly is None:
            seed = time.time_ns()
            rng = random.Random(seed)
            block_width = min(4, self.width)
            block_height = min(4, self.height)
            start_x = rng.randint(0, self.width - block_width)
            start_y = rng.randint(0, self.height - block_height)
            self.no_fly = [
                (start_x + dx, start_y + dy)
                for dx in range(block_width)
                for dy in range(block_height)
            ]

    def random_cell(self) -> Tuple[int, int]:
        """Return a random coordinate inside the reserve."""
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def is_no_fly(self, cell: Tuple[int, int]) -> bool:
        """Check whether a cell is part of the restricted no-fly list."""
        return cell in self.no_fly
