from langchain_core.documents import Document

from models.chat_manager import collate_docs, serialize_doc


def test_serialize_doc_includes_index_text_and_metadata():
    doc = Document(page_content="Audit evidence", metadata={"document": "policy.pdf"})

    serialized = serialize_doc(doc, index=2)

    assert serialized == {
        "index": 2,
        "text": "Audit evidence",
        "metadata": {"document": "policy.pdf"},
    }


def test_collate_docs_formats_context_with_sources():
    docs = [
        Document(page_content="First chunk", metadata={"document": "a.pdf"}),
        Document(page_content="Second chunk", metadata={}),
    ]

    context = collate_docs(docs)

    assert "[1] Source: a.pdf\nFirst chunk" in context
    assert "[2] Source: unknown\nSecond chunk" in context
