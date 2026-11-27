"""Per-run record of model calls, with a hard cap on billed tokens."""

import json
from pathlib import Path

from aomma.backends.vlm import Usage


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, max_billed_tokens: int):
        self.path = path
        self.max_billed_tokens = max_billed_tokens

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def billed_tokens(self) -> int:
        return sum(
            e["input_tokens"] + e["output_tokens"] for e in self.entries() if e["billed"]
        )

    def check_budget(self) -> None:
        """Call before every model request; raises once the run's cap is spent."""
        spent = self.billed_tokens()
        if spent >= self.max_billed_tokens:
            raise BudgetExceeded(
                f"run has used {spent} billed tokens (cap {self.max_billed_tokens})"
            )

    def record(self, stage: str, usage: Usage) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"stage": stage, **usage.to_dict()}) + "\n")
