"""Split the telecom PDF guide and store its chunks in Chroma."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retriever import DATA_DIR, GUIDE_COLLECTION, sync_documents


GUIDE_PDF_PATH = DATA_DIR / "telecom_guide.pdf"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """Load PDF pages and verify that extractable text exists."""
    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF guide not found: {pdf_path}")

    pages = PyPDFLoader(str(pdf_path)).load()

    if not pages or not any(
        page.page_content.strip() for page in pages
    ):
        raise ValueError(
            "The PDF contains no extractable text. "
            "Scanned documents may require OCR."
        )

    return pages


def split_guide_pages(pages: list[Document]) -> list[Document]:
    """Create overlapping chunks while preserving page metadata."""
    # With the default length function, sizes are measured in characters.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents(pages)
        if chunk.page_content.strip()
    ]

    for chunk_index, chunk in enumerate(chunks):
        # Keep the original file reference before assigning a source label.
        chunk.metadata["file_name"] = GUIDE_PDF_PATH.name
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = chunk_index

    return chunks


def main() -> None:
    """Load, split, embed, and synchronize the PDF guide."""
    print("Loading the PDF guide...")
    pages = load_pdf_pages(GUIDE_PDF_PATH)
    print(f"Loaded {len(pages)} pages.")

    print(
        f"Splitting text: size={CHUNK_SIZE}, "
        f"overlap={CHUNK_OVERLAP}..."
    )
    chunks = split_guide_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    document_ids = [
        f"guide:{GUIDE_PDF_PATH.name}:{chunk.metadata['chunk_index']}"
        for chunk in chunks
    ]

    print("Embedding and synchronizing the guide collection...")

    stored_count = sync_documents(
        collection_name=GUIDE_COLLECTION,
        documents=chunks,
        document_ids=document_ids,
    )

    print(f"Done. {stored_count} guide vectors stored.")


if __name__ == "__main__":
    main()
