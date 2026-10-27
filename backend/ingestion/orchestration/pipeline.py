import datetime
import logging
import uuid
from typing import Any
from sqlalchemy import select
from backend.config import get_settings
from backend.constants import ChunkStatus, DocumentStatus, JobStatus, PipelineStep
from backend.database.models import Document, DocumentChunk, IngestionJob
from backend.database.session import async_session_factory
from backend.ingestion.chunking.adaptive_chunker import AdaptiveChunker
from backend.ingestion.embeddings.arabic_service import ArabicEmbeddingService
from backend.ingestion.embeddings.english_service import EnglishEmbeddingService
from backend.ingestion.language.detector import LanguageDetectorService
from backend.ingestion.normalizers.normalizer import DataNormalizer
from backend.ingestion.parsers.csv_parser import CsvParser
from backend.ingestion.parsers.docx_parser import DocxParser
from backend.ingestion.parsers.ocr_parser import AzureDocumentIntelligenceParser
from backend.ingestion.parsers.pdf_parser import PyMuPDFParser
from backend.ingestion.parsers.router import FileRouter
from backend.ingestion.storage.private_store import PrivateStorageManager
from backend.ingestion.translation.gpt4o_service import TranslationService
from backend.ingestion.vector_store.collection import QdrantCollectionManager
from backend.ingestion.vector_store.qdrant_client import QdrantClientManager
from backend.ingestion.verification.verifier import QdrantStorageVerifier

logger = logging.getLogger(__name__)


class IngestionPipelineOrchestrator:
    """
    Coordinates Steps 2 through 11 of the Document Ingestion Pipeline.
    Maintains strict state transitions in PostgreSQL and validates dual vectors in Qdrant.
    """

    def __init__(
        self,
        storage_manager: PrivateStorageManager | None = None,
        translation_service: TranslationService | None = None,
        english_service: EnglishEmbeddingService | None = None,
        arabic_service: ArabicEmbeddingService | None = None,
        session_factory: Any | None = None,
        qdrant_client: Any | None = None,
    ) -> None:
        self.settings = get_settings()
        self.storage = storage_manager or PrivateStorageManager()
        self.translator = translation_service or TranslationService()
        self.en_emb = english_service or EnglishEmbeddingService()
        self.ar_emb = arabic_service or ArabicEmbeddingService()
        self.chunker = AdaptiveChunker()
        self.session_factory = session_factory or async_session_factory
        self.qdrant_client = qdrant_client

    async def execute_pipeline(self, document_id: uuid.UUID, job_id: uuid.UUID) -> dict[str, str]:
        async with self.session_factory() as session:
            doc_stmt = select(Document).where(Document.document_id == document_id)
            doc_res = await session.execute(doc_stmt)
            doc = doc_res.scalar_one_or_none()

            job_stmt = select(IngestionJob).where(IngestionJob.job_id == job_id)
            job_res = await session.execute(job_stmt)
            job = job_res.scalar_one_or_none()

            if not doc or not job:
                raise ValueError(f"Document {document_id} or Job {job_id} not found in database")

            try:
                job.status = JobStatus.PROCESSING.value
                job.started_at = datetime.datetime.now(datetime.timezone.utc)
                job.current_step = PipelineStep.FILE_ROUTING.value
                doc.status = DocumentStatus.PROCESSING.value
                await session.commit()

                # Step 2: File Routing
                file_path = self.storage.locate_existing_file(document_id)
                if not file_path:
                    raise FileNotFoundError(f"Source file for document {document_id} not found in private storage")

                routing_res = FileRouter.route_file(document_id, job_id, doc.file_type, file_path)
                parser_name = routing_res["parser"]
                job.parser = parser_name
                job.current_step = PipelineStep.CONTENT_EXTRACTION.value
                await session.commit()

                # Step 3: Content Extraction
                if parser_name == "pymupdf":
                    extracted_items = PyMuPDFParser.extract(file_path)
                elif parser_name == "azure_ocr":
                    extracted_items = AzureDocumentIntelligenceParser.extract(file_path)
                elif parser_name == "python-docx":
                    extracted_items = DocxParser.extract(file_path)
                elif parser_name == "openpyxl":
                    extracted_items = XlsxParser.extract(file_path)
                elif parser_name == "csv":
                    extracted_items = CsvParser.extract(file_path)
                else:
                    raise ValueError(f"Unrecognized parser '{parser_name}'")

                if not extracted_items:
                    raise ValueError(f"No content could be extracted from {doc.file_name}")

                # Step 4: Data Normalization
                job.current_step = PipelineStep.DATA_NORMALIZATION.value
                await session.commit()
                normalized_elements = DataNormalizer.normalize(document_id, extracted_items)

                # Step 5: Adaptive Chunking & Manifest Generation
                job.current_step = PipelineStep.ADAPTIVE_CHUNKING.value
                await session.commit()
                logical_chunks = self.chunker.chunk(document_id, normalized_elements)

                if not logical_chunks:
                    raise ValueError("Adaptive chunking produced 0 chunks")

                job.total_chunks = len(logical_chunks)
                doc.total_chunks = len(logical_chunks)

                for l_chunk in logical_chunks:
                    chk_stmt = select(DocumentChunk).where(
                        DocumentChunk.document_id == document_id,
                        DocumentChunk.job_id == job_id,
                        DocumentChunk.chunk_index == l_chunk.chunk_index,
                    )
                    existing_chk = (await session.execute(chk_stmt)).scalar_one_or_none()

                    if not existing_chk:
                        db_chunk = DocumentChunk(
                            chunk_id=l_chunk.chunk_id,
                            document_id=document_id,
                            job_id=job_id,
                            chunk_index=l_chunk.chunk_index,
                            content_type=l_chunk.content_type,
                            qdrant_point_id=l_chunk.qdrant_point_id,
                            source_metadata=l_chunk.source,
                            status=ChunkStatus.PENDING.value,
                        )
                        session.add(db_chunk)
                await session.commit()

                # Load and validate models
                self.en_emb.load_model()
                self.ar_emb.load_model()
                en_dim = self.en_emb.dimension
                ar_dim = self.ar_emb.dimension

                # Ensure Qdrant collection exists with dual named vectors
                qdrant_client = self.qdrant_client or QdrantClientManager.get_async_client()
                await QdrantCollectionManager.ensure_collection_exists(
                    client=qdrant_client,
                    collection_name=self.settings.QDRANT_COLLECTION,
                    english_dim=en_dim,
                    arabic_dim=ar_dim,
                )

                verified_count = 0
                processed_count = 0

                for l_chunk in logical_chunks:
                    chk_stmt = select(DocumentChunk).where(DocumentChunk.chunk_id == l_chunk.chunk_id)
                    db_chk = (await session.execute(chk_stmt)).scalar_one()

                    # Step 6: Language Detection
                    job.current_step = PipelineStep.LANGUAGE_DETECTION.value
                    orig_lang, conf = LanguageDetectorService.detect_language(l_chunk.text)
                    db_chk.original_language = orig_lang
                    if conf is not None:
                        meta = dict(db_chk.source_metadata or {})
                        meta["language_confidence"] = conf
                        db_chk.source_metadata = meta
                    await session.commit()

                    # Step 7: Translation & Bilingual Text Preparation
                    job.current_step = PipelineStep.TRANSLATION.value
                    try:
                        text_en, text_ar, trans_status = await self.translator.translate_chunk(
                            original_text=l_chunk.text,
                            original_language=orig_lang,
                        )
                    except Exception as trans_err:
                        logger.warning(f"Translation encountered unexpected error: {trans_err}. Using original text fallback.")
                        text_en = l_chunk.text
                        text_ar = l_chunk.text
                        trans_status = TranslationStatus.SKIPPED.value

                    db_chk.translation_status = trans_status
                    await session.commit()

                    # Step 8: Embedding Generation
                    job.current_step = PipelineStep.EMBEDDING_GENERATION.value
                    vec_en = self.en_emb.encode(text_en)
                    vec_ar = self.ar_emb.encode(text_ar)

                    meta = dict(db_chk.source_metadata or {})
                    meta["embedding_models"] = {
                        "text_english_vector": {
                            "model": self.en_emb.model_name,
                            "revision": self.en_emb.revision,
                            "dimensions": en_dim,
                        },
                        "text_arabic_vector": {
                            "model": self.ar_emb.model_name,
                            "revision": self.ar_emb.revision,
                            "dimensions": ar_dim,
                        },
                    }
                    db_chk.source_metadata = meta
                    await session.commit()

                    # Step 9: Store both vectors in one Qdrant point
                    job.current_step = PipelineStep.QDRANT_STORAGE.value
                    payload = {
                        "chunk_id": str(l_chunk.chunk_id),
                        "document_id": str(document_id),
                        "element_ids": [str(eid) for eid in l_chunk.element_ids],
                        "chunk_index": l_chunk.chunk_index,
                        "content_type": l_chunk.content_type,
                        "original_language": orig_lang,
                        "original_text": l_chunk.text,
                        "text_english": text_en,
                        "text_arabic": text_ar,
                        "translation_status": trans_status,
                        "source": l_chunk.source,
                        "embedding_models": meta["embedding_models"],
                    }

                    await QdrantCollectionManager.upsert_dual_vector_point(
                        client=qdrant_client,
                        collection_name=self.settings.QDRANT_COLLECTION,
                        point_id=l_chunk.qdrant_point_id,
                        text_english_vector=vec_en,
                        text_arabic_vector=vec_ar,
                        payload=payload,
                    )

                    db_chk.status = ChunkStatus.PROCESSED.value
                    processed_count += 1
                    job.processed_chunks = processed_count
                    await session.commit()

                    # Step 10: Deep Qdrant Verification
                    job.current_step = PipelineStep.QDRANT_VERIFICATION.value
                    is_verified, reason, _ = await QdrantStorageVerifier.verify_point(
                        client=qdrant_client,
                        collection_name=self.settings.QDRANT_COLLECTION,
                        expected_point_id=l_chunk.qdrant_point_id,
                        expected_chunk_id=l_chunk.chunk_id,
                        expected_document_id=document_id,
                        expected_en_dim=en_dim,
                        expected_ar_dim=ar_dim,
                        expected_en_model=self.en_emb.model_name,
                        expected_ar_model=self.ar_emb.model_name,
                    )

                    if not is_verified:
                        db_chk.status = ChunkStatus.FAILED.value
                        db_chk.failure_reason = reason
                        await session.commit()
                        raise RuntimeError(f"Chunk {l_chunk.chunk_id} failed Qdrant verification: {reason}")

                    db_chk.status = ChunkStatus.VERIFIED.value
                    verified_count += 1
                    job.verified_chunks = verified_count
                    await session.commit()

                # Step 11: Document Activation & Completion
                job.current_step = PipelineStep.PIPELINE_COMPLETION.value
                if verified_count == len(logical_chunks) and len(logical_chunks) > 0:
                    doc.status = DocumentStatus.ACTIVE.value
                    doc.total_chunks = verified_count
                    job.status = JobStatus.COMPLETED.value
                    job.current_step = PipelineStep.COMPLETED.value
                    job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                    await session.commit()
                    return {"document_status": "ACTIVE", "job_status": "COMPLETED"}
                else:
                    raise RuntimeError("Verification manifest count mismatch during completion")

            except Exception as e:
                logger.exception(f"Pipeline execution failed for document {document_id}")
                doc.status = DocumentStatus.FAILED.value
                doc.failure_reason = str(e)
                job.status = JobStatus.FAILED.value
                job.error_step = job.current_step or "UNKNOWN"
                job.failure_reason = str(e)
                job.completed_at = datetime.datetime.now(datetime.timezone.utc)
                await session.commit()
                return {"document_status": "FAILED", "job_status": "FAILED", "error": str(e)}
