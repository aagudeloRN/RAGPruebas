"""Rename preview_image_url to cover_image_url in documents table

Revision ID: 364237e654b4
Revises: f488694a342e
Create Date: 2025-08-05 16:42:33.345656

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '364237e654b4'
down_revision: Union[str, Sequence[str], None] = 'f488694a342e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('documents', 'preview_image_url', new_column_name='cover_image_url', existing_type=sa.Text(), nullable=True)

def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('documents', 'cover_image_url', new_column_name='preview_image_url', existing_type=sa.Text(), nullable=True)
