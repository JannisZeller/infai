from typing import Any, Literal

from pydantic import BaseModel


class ChatTurnRequest(BaseModel):
    prompt: str


class ToolApprovalDecisionRequest(BaseModel):
    tool_call_id: str
    approved: bool
    denial_message: str | None = None


class ChatResumeRequest(BaseModel):
    resume_token: str
    approvals: list[ToolApprovalDecisionRequest]


class ElicitationResponseRequest(BaseModel):
    elicitation_id: str
    action: Literal["accept", "decline", "cancel"]
    content: dict[str, Any] | None = None
