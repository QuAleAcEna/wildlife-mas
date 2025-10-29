from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class Reserve:
    width: int = 20
    height: int = 20
    no_fly: List[Tuple[int, int]] = None


    def __post_init__(self):
        if self.no_fly is None:
            # Mark a simple 4x4 square as no-fly
            self.no_fly = [(x, y) for x in range(8, 12) for y in range(8, 12)]


    def random_cell(self) -> Tuple[int, int]:
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))


    def is_no_fly(self, cell: Tuple[int, int]) -> bool:
        return cell in self.no_fly
