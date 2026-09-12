"""Cap the number of model calls per run (the original loop's ``max_steps``)."""

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware


class AgentStepLimitError(Exception):
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        super().__init__("Agent exceeded max tool turns")


class StepLimitMiddleware(AgentMiddleware):
    """Raise once the run would make more than ``max_steps`` model calls."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        calls = int(state.get("model_calls") or 0) + 1
        if calls > self.max_steps:
            raise AgentStepLimitError(self.max_steps)
        return {"model_calls": calls}

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
