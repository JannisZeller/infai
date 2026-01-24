from dataclasses import dataclass
from enum import Enum
from time import time_ns
from typing import Any
from uuid import UUID, uuid4

from src.ai.models import ModelResponseDelta, PartStart, ThinkingDelta
from src.history.models import ModelResponse, ThinkingStep


class PartState(Enum):
    THINKING = "thinking"
    TALKING = "talking"
    NO_STREAM = "no_stream"
    TOOL_CALL_PREP = "tool_call_prep"


ModelRequestFullItemYields = ModelResponse | ThinkingStep
ModelRequestYields = ModelRequestFullItemYields | ModelResponseDelta | ThinkingDelta


@dataclass
class ModelRequestCurrentPart:
    """
    Tracks the current part being streamed in the the models request, i.e.
    the "request" of the model back to the user or a tool call.
    This is used to collapse consecutive parts of the same type.
    """

    history_id: UUID
    id: UUID | None = None
    state: PartState = PartState.NO_STREAM
    content: str = ""
    provider_details: dict[str, Any] | None = None

    def is_streaming_but_not_in_state(self, state: PartState) -> bool:
        """Check if we're currently tracking a part but not in the given state."""
        return self.state != PartState.NO_STREAM and self.state != state

    def is_not_streaming(self) -> bool:
        """Check if we're not currently tracking a part."""
        return self.state == PartState.NO_STREAM

    def add_content_and_yield_delta(self, content: str) -> ModelRequestYields:
        """Add content to the current part and yield the corresponding delta.

        Args:
            content: str - The content to add to the current part
            flow_id: FlowId - The flow id to yield the delta for
            separator: str - The separator to use between the current content and the new content.
                Typically only used when collapsing consecutive parts of the same type.

        Returns:
            ModelRequestNodeYields - The delta to yield
        """
        assert self.state != PartState.NO_STREAM, (
            "ModelRequestNodeCurrentPart must be in state NO_STREAM when adding content"
        )
        assert self.id is not None, "flow_item_id must be set when part is active when adding content"

        self.content += content

        match self.state:
            case PartState.THINKING:
                return ThinkingDelta(
                    id=self.id,
                    history_id=self.history_id,
                    created_at=time_ns(),
                    delta=content,
                )
            case PartState.TALKING:
                return ModelResponseDelta(
                    id=self.id,
                    history_id=self.history_id,
                    created_at=time_ns(),
                    delta=content,
                )
            case PartState.TOOL_CALL_PREP:
                raise ValueError("Tool call prep part should not be added to the current part")

    def reset_to_state_and_get_part_start(self, state: PartState) -> PartStart:
        """Reset the part to the given state and a new flow item id."""
        self.id = uuid4()
        self.state = state
        self.content = ""
        match state:
            case PartState.THINKING:
                part_type = "thinking"
            case PartState.TALKING:
                part_type = "response"
            case PartState.TOOL_CALL_PREP:
                part_type = "tool_call_prep"
            case PartState.NO_STREAM:
                raise ValueError(
                    "ModelRequestCurrentPart should never be `reset_to_state_and_get_part_start`'ed to state=NO_STREAM"
                )
        return PartStart(
            id=self.id,
            history_id=self.history_id,
            created_at=time_ns(),
            part_type=part_type,
        )

    def reset_to_no_stream(self) -> None:
        """Reset the part to no stream."""
        self.id = uuid4()
        self.state = PartState.NO_STREAM
        self.content = ""

    def flush(self) -> ModelRequestFullItemYields | None:
        """Flush the current part as a final flow item if active."""

        match self.state:
            case PartState.THINKING:
                assert self.id is not None, "flow_item_id must be set when flushing a thinking part"
                flow_item = ThinkingStep(
                    id=self.id,
                    history_id=self.history_id,
                    created_at=time_ns(),
                    thoughts=self.content,
                )
            case PartState.TALKING:
                assert self.id is not None, "flow_item_id must be set when flushing a talking part"
                flow_item = ModelResponse(
                    id=self.id,
                    history_id=self.history_id,
                    created_at=time_ns(),
                    response=self.content,
                )
            case PartState.TOOL_CALL_PREP:
                raise ValueError("Tool call prep part should not be flushed")
            case PartState.NO_STREAM:
                flow_item = None

        return flow_item
