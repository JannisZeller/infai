"""create token store entries table

Revision ID: 0002_token_store_entries
Revises: 0001_history_tables
Create Date: 2026-04-04 00:00:01.000000

"""

from typing import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_token_store_entries"
down_revision: str | Sequence[str] | None = "0001_history_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "token_store_entries",
        sa.Column("collection", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("expires_at", sa.BigInteger(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("collection", "key"),
    )
    op.create_index(op.f("ix_token_store_entries_expires_at"), "token_store_entries", ["expires_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_token_store_entries_expires_at"), table_name="token_store_entries")
    op.drop_table("token_store_entries")
