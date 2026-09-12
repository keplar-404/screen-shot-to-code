"""Deep Agents harness profile for OpenRouter-backed models.

Registered once at import time. It removes the parts of the default deep-agent
harness this app does not want:

- ``task`` (general-purpose subagent): the code-generation loop is single-agent.
- ``read_file``: the model writes through ``create_file`` / ``edit_file`` and
  never needs the built-in filesystem view (``FilesystemMiddleware`` requires
  ``read_file`` in its allowlist, so it is hidden here instead).
- ``SummarizationMiddleware``: it would compact the conversation (including
  the input screenshot) on long runs; the original loop never summarised and
  is bounded by ``StepLimitMiddleware`` / ``BudgetMiddleware`` instead.
"""

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

OPENROUTER_HARNESS_PROFILE = HarnessProfile(
    excluded_tools=frozenset({"read_file"}),
    excluded_middleware=frozenset({"SummarizationMiddleware"}),
    general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
)

_registered = False


def ensure_profiles_registered() -> None:
    global _registered
    if _registered:
        return
    register_harness_profile("openrouter", OPENROUTER_HARNESS_PROFILE)
    _registered = True


ensure_profiles_registered()
