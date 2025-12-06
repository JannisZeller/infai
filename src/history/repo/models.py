from time import time_ns
from typing import Any
from uuid import UUID, uuid4

from sqlmodel import JSON, Column, Field, Relationship, SQLModel  # type: ignore


class HistoryItemDb(SQLModel, table=True):
    __tablename__ = "history_items"  # type: ignore

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    history_id: UUID = Field(foreign_key="history.id", nullable=False, index=True)
    created_at: int = Field(default_factory=time_ns, nullable=False, index=True)
    kind: str = Field(nullable=False, index=True)
    content: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))

    history: "HistoryDb" = Relationship(back_populates="items")


class HistoryDb(SQLModel, table=True):
    __tablename__ = "history"  # type: ignore

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    created_at: int = Field(default_factory=time_ns, nullable=False, index=True)

    items: list["HistoryItemDb"] = Relationship(back_populates="history")
