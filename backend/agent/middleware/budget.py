# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Abort a variant that would keep spending past ``GENERATION_MAX_COST_USD``.

Cost comes from OpenRouter's usage accounting (``usage: {include: true}``
puts a USD ``cost`` in the usage block; ``ChatOpenRouter`` surfaces it as
``response_metadata["cost"]``). When OpenRouter reports no cost (e.g. BYOK
endpoints), the per-million-token table in ``costs.pricing`` is used; models
with neither are unbounded — the same rule the original providers applied.
"""

from typing import Any, Optional

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage

from costs.pricing import MODEL_PRICING
from costs.token_usage import TokenUsage, token_usage_from_message


class BudgetExceededError(Exception):
    """Shown verbatim to end users (variantError), so no cost figures here."""

    def __init__(self) -> None:
        super().__init__("Generation stopped: this variant exceeded its resource limit.")


def message_cost_usd(message: AIMessage, pricing_key: Optional[str]) -> Optional[float]:
    metadata = message.response_metadata or {}
    cost = metadata.get("cost")
    if isinstance(cost, (int, float)):
        return float(cost)
    if pricing_key:
        pricing = MODEL_PRICING.get(pricing_key)
        usage: Optional[TokenUsage] = token_usage_from_message(message)
        if pricing is not None and usage is not None:
            return usage.cost(pricing)
    return None


class BudgetMiddleware(AgentMiddleware):
    def __init__(self, max_cost_usd: float, pricing_key: Optional[str] = None) -> None:
        self.max_cost_usd = max_cost_usd
        self.pricing_key = pricing_key

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        last: AIMessage = messages[-1]
        cost = message_cost_usd(last, self.pricing_key)
        spent = float(state.get("spent_usd") or 0.0)
        if cost is not None:
            spent += cost
        # Abort only when the run would otherwise continue: a run that just
        # produced its final answer is already paid for.
        if last.tool_calls and cost is not None and spent > self.max_cost_usd:
            print(f"[BUDGET] Aborting variant: ${spent:.2f} > ${self.max_cost_usd:.2f}")
            raise BudgetExceededError()
        return {"spent_usd": spent}

    async def aafter_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
