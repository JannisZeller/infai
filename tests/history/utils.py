from src.history.models import History, UserPrompt


def compare_user_prompt(prompt: UserPrompt, other_prompt: UserPrompt):
    assert prompt.id == other_prompt.id
    assert prompt.history_id == other_prompt.history_id
    assert prompt.created_at == other_prompt.created_at
    assert prompt.prompt == other_prompt.prompt


def compare_history(history: History, other_history: History):
    assert history.id == other_history.id
    assert history.created_at == other_history.created_at
    assert len(history.items) == len(other_history.items)
    for hi1, hi2 in zip(history.items, other_history.items):
        if not isinstance(hi1, UserPrompt) or not isinstance(hi2, UserPrompt):
            raise NotImplementedError("Only UserPrompt is supported for now")
        compare_user_prompt(hi1, hi2)
