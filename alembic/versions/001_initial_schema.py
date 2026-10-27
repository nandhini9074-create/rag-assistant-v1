"""initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-27 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. documents table
    op.create_table(
        'documents',
        sa.Column('document_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_type', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='UPLOADED', nullable=False),
        sa.Column('total_chunks', sa.Integer(), server_default='0', nullable=False),
        sa.Column('uploaded_by', sa.String(length=128), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('total_chunks >= 0', name='chk_doc_total_chunks_non_negative'),
        sa.CheckConstraint('file_size_bytes IS NULL OR file_size_bytes >= 0', name='chk_doc_file_size_non_negative'),
    )
    op.create_index('ix_documents_status', 'documents', ['status'])
    op.create_index('ix_documents_created_at', 'documents', ['created_at'])

    # 2. ingestion_jobs table
    op.create_table(
        'ingestion_jobs',
        sa.Column('job_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.document_id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='QUEUED', nullable=False),
        sa.Column('current_step', sa.String(length=64), nullable=True),
        sa.Column('parser', sa.String(length=64), nullable=True),
        sa.Column('total_chunks', sa.Integer(), server_default='0', nullable=False),
        sa.Column('processed_chunks', sa.Integer(), server_default='0', nullable=False),
        sa.Column('verified_chunks', sa.Integer(), server_default='0', nullable=False),
        sa.Column('attempt_number', sa.Integer(), server_default='1', nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_step', sa.String(length=64), nullable=True),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('document_id', 'attempt_number', name='uq_document_attempt'),
        sa.CheckConstraint('total_chunks >= 0', name='chk_job_total_chunks_non_neg'),
        sa.CheckConstraint('processed_chunks >= 0', name='chk_job_processed_chunks_non_neg'),
        sa.CheckConstraint('verified_chunks >= 0', name='chk_job_verified_chunks_non_neg'),
    )
    op.create_index('ix_ingestion_jobs_document_id_status', 'ingestion_jobs', ['document_id', 'status'])
    op.create_index('ix_ingestion_jobs_status', 'ingestion_jobs', ['status'])

    # 3. document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.document_id', ondelete='CASCADE'), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ingestion_jobs.job_id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content_type', sa.String(length=64), nullable=False),
        sa.Column('original_language', sa.String(length=16), nullable=True),
        sa.Column('translation_status', sa.String(length=32), server_default='PENDING', nullable=False),
        sa.Column('qdrant_point_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=32), server_default='PENDING', nullable=False),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('document_id', 'job_id', 'chunk_index', name='uq_document_job_chunk_index'),
    )
    op.create_index('ix_document_chunks_document_id_status', 'document_chunks', ['document_id', 'status'])
    op.create_index('ix_document_chunks_job_id_status', 'document_chunks', ['job_id', 'status'])
    op.create_index('ix_document_chunks_qdrant_point_id', 'document_chunks', ['qdrant_point_id'])
    op.create_index('ix_document_chunks_status', 'document_chunks', ['status'])


def downgrade() -> None:
    op.drop_table('document_chunks')
    op.drop_table('ingestion_jobs')
    op.drop_table('documents')
