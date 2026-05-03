from dataclasses import dataclass
from typing import List


# ----------------------------
# Logical operator definitions
# ----------------------------

@dataclass(frozen=True)
class LogicalOperator:
    name: str
    prompt: str


@dataclass(frozen=True)
class SemanticFilter(LogicalOperator):
    pass


@dataclass(frozen=True)
class SemanticMap(LogicalOperator):
    pass


# ----------------------------
# Logical plan
# ----------------------------

@dataclass
class LogicalPlan:
    operators: List[LogicalOperator]
    buffer_size: int
    semantic_boundary: str  # e.g. "buffer_full"


# ----------------------------
# Query Planner
# ----------------------------

class QueryPlanner:
    """
    Produces a logical execution plan.
    Does NOT execute anything.
    """

    def __init__(self):
        self.operators: List[LogicalOperator] = []

    def add_filter(self, prompt: str):
        self.operators.append(
            SemanticFilter(name="sem_filter", prompt=prompt)
        )

    def add_map(self, prompt: str):
        self.operators.append(
            SemanticMap(name="sem_map", prompt=prompt)
        )

    def plan(self, buffer_size: int = 64) -> LogicalPlan:
        """
        Analyze operator sequence and emit a logical plan.
        """
        return LogicalPlan(
            operators=self.operators,
            buffer_size=buffer_size,
            semantic_boundary="buffer_full",
        )


planner = QueryPlanner()

planner.add_filter("Is the candidate capable of GPU programming?")
planner.add_map("Summarize resume with skill set")

plan = planner.plan(buffer_size=64)