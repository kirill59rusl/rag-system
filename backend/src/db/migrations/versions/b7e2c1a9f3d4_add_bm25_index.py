"""add bm25 index

Revision ID: b7e2c1a9f3d4
Revises: d65c602a7b0d
Create Date: 2026-10-08 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b7e2c1a9f3d4'
down_revision: Union[str, Sequence[str], None] = 'd65c602a7b0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    # icu не режет идентификаторы (max_wal_size, pg_stat_statements) на части,
    # документация на английском — английские стеммер и стоп-слова
    op.execute(
        """
        CREATE INDEX chunks_bm25_idx ON chunks
        USING paradedb (
            id,
            (content::pdb.icu('stemmer=english', 'stopwords_language=english'))
        )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS chunks_bm25_idx")
