from uuid import uuid4

from src.ai.pydantic_ai.mapper import PydanticAiMapper


def test_map_user_prompt_out_joins_sequence_parts():
    history_id = uuid4()

    user_prompt = PydanticAiMapper.map_user_prompt_out(
        pai_user_prompt=["Call ", "the tool"],
        id=uuid4(),
        history_id=history_id,
    )

    assert user_prompt is not None
    assert user_prompt.history_id == history_id
    assert user_prompt.prompt == "Call the tool"


def test_map_user_prompt_out_returns_none_for_missing_prompt():
    user_prompt = PydanticAiMapper.map_user_prompt_out(
        pai_user_prompt=None,
        id=uuid4(),
        history_id=uuid4(),
    )

    assert user_prompt is None
