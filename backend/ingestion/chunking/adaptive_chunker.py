import uuid
from typing import Any
from backend.config import get_settings
from backend.constants import ContentType
from backend.ingestion.chunking.tokenizer import TokenEstimator
from backend.ingestion.normalizers.normalizer import DataNormalizer
from backend.ingestion.normalizers.schema import LogicalChunk, NormalizedElement

CHUNK_NAMESPACE = uuid.UUID("b2c3d4e5-f6a7-8b9c-0d1e-2f3a4b5c6d7e")
POINT_NAMESPACE = uuid.UUID("c3d4e5f6-a7b8-9c0d-1e2f-3a4b5c6d7e8f")


class AdaptiveChunker:
    """
    Step 5: Adaptive Chunking.
    Splits normalized elements into token-bounded logical chunks.
    Maintains table row groups with repeated headers.
    Binds headings to subsequent content.
    Generates deterministic UUIDs for retry idempotency.
    """

    def __init__(
        self,
        max_chunk_tokens: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.max_tokens = max_chunk_tokens or settings.CHUNK_SIZE_TOKENS
        self.overlap_tokens = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS

    @classmethod
    def generate_chunk_id(cls, document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
        seed = f"{document_id}:chunk:{chunk_index}"
        return uuid.uuid5(CHUNK_NAMESPACE, seed)

    @classmethod
    def generate_point_id(cls, document_id: uuid.UUID, chunk_index: int) -> uuid.UUID:
        seed = f"{document_id}:qdrant:{chunk_index}"
        return uuid.uuid5(POINT_NAMESPACE, seed)

    def chunk(self, document_id: uuid.UUID, elements: list[NormalizedElement]) -> list[LogicalChunk]:
        chunks: list[LogicalChunk] = []
        chunk_index = 0

        pending_heading: NormalizedElement | None = None
        current_text_parts: list[str] = []
        current_element_ids: list[uuid.UUID] = []
        current_sources: list[dict[str, Any]] = []
        current_tokens = 0

        def flush_text_chunk(content_type: str = ContentType.NARRATIVE_TEXT.value) -> None:
            nonlocal chunk_index, current_text_parts, current_element_ids, current_sources, current_tokens, pending_heading
            if not current_text_parts:
                return

            full_text = "\n\n".join(current_text_parts).strip()
            if full_text:
                cid = self.generate_chunk_id(document_id, chunk_index)
                pid = self.generate_point_id(document_id, chunk_index)

                merged_source = {"sources": current_sources} if len(current_sources) > 1 else (current_sources[0] if current_sources else {})

                chunks.append(
                    LogicalChunk(
                        chunk_id=cid,
                        chunk_index=chunk_index,
                        element_ids=list(current_element_ids),
                        content_type=content_type,
                        text=full_text,
                        source=merged_source,
                        qdrant_point_id=pid,
                    )
                )
                chunk_index += 1

            current_text_parts = []
            current_element_ids = []
            current_sources = []
            current_tokens = 0
            pending_heading = None

        for el in elements:
            if el.content_type == ContentType.HEADING.value:
                flush_text_chunk()
                pending_heading = el
                continue

            if el.content_type == ContentType.FINANCIAL_TABLE.value:
                flush_text_chunk()

                raw = el.raw_data or {}
                headers = raw.get("headers", [])
                rows = raw.get("rows", [])

                if headers and rows:
                    header_line = " | ".join(headers)
                    header_tokens = TokenEstimator.count_tokens(header_line)

                    row_batch: list[list[str]] = []
                    row_batch_tokens = header_tokens

                    for row in rows:
                        row_line = " | ".join(row)
                        r_tokens = TokenEstimator.count_tokens(row_line)

                        if row_batch and (row_batch_tokens + r_tokens > self.max_tokens):
                            table_md = DataNormalizer.format_markdown_table(headers, row_batch)
                            if pending_heading:
                                table_md = f"### {pending_heading.text}\n\n{table_md}"
                                el_ids = [pending_heading.element_id, el.element_id]
                                pending_heading = None
                            else:
                                el_ids = [el.element_id]

                            cid = self.generate_chunk_id(document_id, chunk_index)
                            pid = self.generate_point_id(document_id, chunk_index)
                            chunks.append(
                                LogicalChunk(
                                    chunk_id=cid,
                                    chunk_index=chunk_index,
                                    element_ids=el_ids,
                                    content_type=ContentType.FINANCIAL_TABLE.value,
                                    text=table_md,
                                    source=el.source,
                                    qdrant_point_id=pid,
                                )
                            )
                            chunk_index += 1
                            row_batch = [row]
                            row_batch_tokens = header_tokens + r_tokens
                        else:
                            row_batch.append(row)
                            row_batch_tokens += r_tokens

                    if row_batch:
                        table_md = DataNormalizer.format_markdown_table(headers, row_batch)
                        if pending_heading:
                            table_md = f"### {pending_heading.text}\n\n{table_md}"
                            el_ids = [pending_heading.element_id, el.element_id]
                            pending_heading = None
                        else:
                            el_ids = [el.element_id]

                        cid = self.generate_chunk_id(document_id, chunk_index)
                        pid = self.generate_point_id(document_id, chunk_index)
                        chunks.append(
                            LogicalChunk(
                                chunk_id=cid,
                                chunk_index=chunk_index,
                                element_ids=el_ids,
                                content_type=ContentType.FINANCIAL_TABLE.value,
                                text=table_md,
                                source=el.source,
                                qdrant_point_id=pid,
                            )
                        )
                        chunk_index += 1
                else:
                    t_text = el.text
                    if pending_heading:
                        t_text = f"### {pending_heading.text}\n\n{t_text}"
                        el_ids = [pending_heading.element_id, el.element_id]
                        pending_heading = None
                    else:
                        el_ids = [el.element_id]

                    cid = self.generate_chunk_id(document_id, chunk_index)
                    pid = self.generate_point_id(document_id, chunk_index)
                    chunks.append(
                        LogicalChunk(
                            chunk_id=cid,
                            chunk_index=chunk_index,
                            element_ids=el_ids,
                            content_type=ContentType.FINANCIAL_TABLE.value,
                            text=t_text,
                            source=el.source,
                            qdrant_point_id=pid,
                        )
                    )
                    chunk_index += 1
                continue

            text_to_add = el.text
            if pending_heading:
                text_to_add = f"### {pending_heading.text}\n\n{text_to_add}"
                current_element_ids.append(pending_heading.element_id)
                current_sources.append(pending_heading.source)
                pending_heading = None

            el_tokens = TokenEstimator.count_tokens(text_to_add)

            if current_text_parts and (current_tokens + el_tokens > self.max_tokens):
                flush_text_chunk(content_type=el.content_type)

            current_text_parts.append(text_to_add)
            current_element_ids.append(el.element_id)
            current_sources.append(el.source)
            current_tokens += el_tokens

        flush_text_chunk()
        return chunks
