import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class EnvironmentClock:
    def __init__(
        self,
        seconds_per_hour: float = 60.0,
        start_day: int = 1,
        start_hour: int = 8,
    ) -> None:
        self.seconds_per_hour = seconds_per_hour
        self._start_day = start_day
        self._start_hour = start_hour
        self.total_hours_elapsed: int = 0
        self.current_day: int = start_day
        self.current_hour: int = start_hour
        self._task: Optional[asyncio.Task[None]] = None
        self._logger = logging.getLogger("environment.clock")

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="EnvironmentClock")

    async def stop(self) -> None:
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
        while True:
            await asyncio.sleep(self.seconds_per_hour)
            self._advance_time()

    def _advance_time(self) -> None:
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
    width: int = 20
    height: int = 20
    no_fly: List[Tuple[int, int]] = None
    clock: EnvironmentClock = field(default_factory=EnvironmentClock)

    def __post_init__(self) -> None:
        if self.no_fly is None:
            # Mark a simple 4x4 square as no-fly
            self.no_fly = [(x, y) for x in range(8, 12) for y in range(8, 12)]

    def random_cell(self) -> Tuple[int, int]:
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def is_no_fly(self, cell: Tuple[int, int]) -> bool:
        return cell in self.no_fly
