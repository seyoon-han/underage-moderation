from .policy import DEFAULT_AGE_THRESHOLD, ModerationPolicy, decide_underage


def run_underage_moderation(*args, **kwargs):
    from .pipeline import run_underage_moderation as _run_underage_moderation

    return _run_underage_moderation(*args, **kwargs)

__all__ = [
    "DEFAULT_AGE_THRESHOLD",
    "ModerationPolicy",
    "decide_underage",
    "run_underage_moderation",
]
