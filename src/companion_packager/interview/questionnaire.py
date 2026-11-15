"""Interactive and headless user needs questionnaire with custom intent parser."""

from __future__ import annotations

from collections import Counter
import re
from typing import List, Optional
from rich.prompt import Confirm, Prompt

from companion_packager.schemas import UserNeeds

VALID_TASKS = (
    "coding",
    "creative_writing",
    "general_desktop",
    "reasoning_math",
    "summarization",
    "custom",
    "reasoning",
    "chat",
)

TASK_ALIASES: dict[str, str] = {}

VALID_PRIORITIES = ("quality", "balanced", "speed", "max_iq")
VALID_CONTEXTS = ("16384", "32768", "8192", "4096")

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "want", "need", "like",
    "use", "make", "help", "from", "some", "tool", "model", "best", "good", "fast",
}


def parse_custom_intent(user_text: str) -> List[str]:
    """Parse freeform intent text and return top 3 domain keywords.

    Uses regex tokenization supporting programming languages and symbols (e.g. c++, c#, sql),
    strips common stopwords, and ranks by frequency and token length.
    """
    if not user_text:
        return ["instruct", "assistant", "general"]

    tokens = re.findall(r"(?:\b[a-z0-9]+[+#]*|\b[a-z0-9+#]{2,}\b)", user_text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS]

    if not filtered:
        return ["instruct", "assistant", "general"]

    counts = Counter(filtered)
    unique_sorted = sorted(counts.keys(), key=lambda w: (-counts[w], -len(w), w))
    result = unique_sorted[:3]

    return result if result else ["instruct", "assistant", "general"]


def ask_user_needs(
    interactive: bool = True,
    task: Optional[str] = None,
    priority: Optional[str] = None,
    min_context_tokens: Optional[int] = None,
    require_permissive_license: Optional[bool] = None,
    custom_intent: Optional[str] = None,
) -> UserNeeds:
    """Collect user operational constraints and workload preferences.

    Supports both interactive Rich prompts and headless CLI parameters.
    """
    if interactive:
        if task is None:
            task = Prompt.ask(
                "[bold cyan]Select primary task domain[/bold cyan]",
                choices=list(VALID_TASKS),
                default="coding",
            )
        if priority is None:
            priority = Prompt.ask(
                "[bold cyan]Select optimization goal[/bold cyan]",
                choices=["quality", "balanced", "speed"],
                default="balanced",
            )
        if min_context_tokens is None:
            ctx_choice = Prompt.ask(
                "[bold cyan]Select context window token length[/bold cyan]",
                choices=list(VALID_CONTEXTS),
                default="16384",
            )
            min_context_tokens = int(ctx_choice)
        if require_permissive_license is None:
            require_permissive_license = Confirm.ask(
                "[bold cyan]Require strictly permissive license (Apache-2.0 / MIT)?[/bold cyan]",
                default=False,
            )
        if str(task).lower().strip() == "custom" and custom_intent is None:
            custom_intent = Prompt.ask(
                "[bold cyan]Enter custom workload intent / description[/bold cyan]",
                default="",
            )
    else:
        task = task or "coding"
        priority = priority or "balanced"
        min_context_tokens = 16384 if min_context_tokens is None else int(min_context_tokens)
        require_permissive_license = (
            False if require_permissive_license is None else bool(require_permissive_license)
        )
        custom_intent = custom_intent or ""

    task_clean = str(task).lower().strip()
    priority_clean = str(priority).lower().strip()
    custom_intent_clean = str(custom_intent or "").strip()

    # Map legacy aliases
    if task_clean in TASK_ALIASES:
        task_clean = TASK_ALIASES[task_clean]

    if task_clean not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task_clean}'. Must be one of: {', '.join(VALID_TASKS)}")

    if priority_clean not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority_clean}'. Must be one of: {', '.join(VALID_PRIORITIES)}"
        )

    if int(min_context_tokens) <= 0:
        raise ValueError(f"min_context_tokens must be positive, got {min_context_tokens}")

    return UserNeeds(
        task=task_clean,
        priority=priority_clean,
        min_context_tokens=int(min_context_tokens),
        require_permissive_license=bool(require_permissive_license),
        custom_intent=custom_intent_clean,
    )
