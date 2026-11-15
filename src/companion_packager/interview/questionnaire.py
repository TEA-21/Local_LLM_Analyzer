"""Interactive and headless user needs questionnaire."""

from typing import Optional
from rich.prompt import Confirm, Prompt

from companion_packager.schemas import UserNeeds

VALID_TASKS = ("coding", "reasoning", "chat")
VALID_PRIORITIES = ("quality", "balanced", "speed", "max_iq")
VALID_CONTEXTS = ("16384", "32768", "8192", "4096")


def ask_user_needs(
    interactive: bool = True,
    task: Optional[str] = None,
    priority: Optional[str] = None,
    min_context_tokens: Optional[int] = None,
    require_permissive_license: Optional[bool] = None,
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
    else:
        task = task or "coding"
        priority = priority or "balanced"
        min_context_tokens = 16384 if min_context_tokens is None else int(min_context_tokens)
        require_permissive_license = (
            False if require_permissive_license is None else bool(require_permissive_license)
        )

    task_clean = str(task).lower().strip()
    priority_clean = str(priority).lower().strip()

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
    )
