"""initial: api_keys and usage_log

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False, index=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("rpm_limit", sa.Integer, nullable=False),
        sa.Column("daily_token_budget", sa.Integer, nullable=True),
        sa.Column("daily_usd_budget", sa.Numeric(12, 6), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "usage_log",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("request_id", sa.String(64), nullable=False, index=True),
        sa.Column("api_key_id", sa.String(32), sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("requested_model", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(32), nullable=True),
        sa.Column("upstream_model", sa.String(128), nullable=True),
        sa.Column("status", sa.Integer, nullable=False),
        sa.Column("prompt_tokens", sa.Integer, nullable=False),
        sa.Column("completion_tokens", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("cache_hit", sa.Boolean, nullable=False),
        sa.Column("fallbacks", sa.Integer, nullable=False),
        sa.Column("retries", sa.Integer, nullable=False),
        sa.Column("stream", sa.Boolean, nullable=False),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_usage_key_created", "usage_log", ["api_key_id", "created_at"])


def downgrade() -> None:
    op.drop_table("usage_log")
    op.drop_table("api_keys")
