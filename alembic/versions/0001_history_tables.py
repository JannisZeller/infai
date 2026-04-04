"""create history tables

Revision ID: 0001_history_tables
Revises:
Create Date: 2026-04-04 00:00:00.000000

"""

from typing import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_history_tables"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "history",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_history_created_at"), "history", ["created_at"], unique=False)
    op.create_index(op.f("ix_history_id"), "history", ["id"], unique=False)

    op.create_table(
        "history_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("history_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["history_id"], ["history.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_history_items_created_at"), "history_items", ["created_at"], unique=False)
    op.create_index(op.f("ix_history_items_history_id"), "history_items", ["history_id"], unique=False)
    op.create_index(op.f("ix_history_items_id"), "history_items", ["id"], unique=False)
    op.create_index(op.f("ix_history_items_kind"), "history_items", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_history_items_kind"), table_name="history_items")
    op.drop_index(op.f("ix_history_items_id"), table_name="history_items")
    op.drop_index(op.f("ix_history_items_history_id"), table_name="history_items")
    op.drop_index(op.f("ix_history_items_created_at"), table_name="history_items")
    op.drop_table("history_items")

    op.drop_index(op.f("ix_history_id"), table_name="history")
    op.drop_index(op.f("ix_history_created_at"), table_name="history")
    op.drop_table("history")
