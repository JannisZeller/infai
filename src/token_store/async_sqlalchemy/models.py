from time import time_ns

from sqlalchemy import LargeBinary
from sqlmodel import Column, Field, SQLModel


class TokenStoreEntryDb(SQLModel, table=True):
    __tablename__ = "token_store_entries"

    collection: str = Field(primary_key=True)
    key: str = Field(primary_key=True)
    ciphertext: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    expires_at: int | None = Field(default=None, nullable=True, index=True)
    created_at: int = Field(default_factory=time_ns, nullable=False)
    updated_at: int = Field(default_factory=time_ns, nullable=False)
