This repository contains a document question-answering project built around a RAG workflow. It extracts text chunks from PDF documents, builds Chroma vector indexes with Hugging Face embeddings, and serves a FastAPI chat interface that streams answers with supporting source chunks.

## What It Does

- Parses PDF files into reusable text chunks.
- Builds local Chroma indexes for semantic retrieval.
- Retrieves the most relevant chunks for a user question.
- Streams generated answers through an OpenAI-compatible chat completion API. Include inline referencing to the relevant pieces of text.
- Serves a small browser-based chat UI at the API root.
- Tracks data and pipeline outputs with DVC.

## Project Structure

```text
.
├── chat/
│   ├── api/              # FastAPI app, routes, and chat endpoints
│   ├── prompts/          # Prompt templates used for answer generation and for potential LLM-based classifier/summarization/content mining component
│   └── ui/               # Static chat UI
├── data/                 # DVC-tracked source document data
├── models/               # ChatManager retrieval and generation logic
├── output/               # Generated chunks and Chroma indexes
├── scripts/              # Pipeline scripts for chunking and indexing
├── tests/                # Unit and API tests
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Embedding model and indexing parameters
├── pyproject.toml        # Python package and tool configuration
└── Dockerfile            # Container entrypoint for the API server
```

## Requirements

- Python 3.11 or newer
- `uv` for dependency management
- DVC, including the SSH remote support used by this project
- An OpenAI-compatible API key for answer generation

The embedding model is configured in `params.yaml`. By default, the project uses [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Configure the chat model provider:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-chat-model"
```

Optional provider override:

```bash
export OPENAI_API_BASE="https://api.mistral.ai/v1"
```

`OPENAI_API_BASE` defaults to `https://api.mistral.ai/v1` in the current code.

## Data And Indexing

The dataset was generated using `GPT-5.5` based on publicly available audit and compliance guidelines. For each topic, I generated a synthetic company report containing evidence that may support some of the checklist requirements.

To make the task more realistic, the reports were intentionally designed to contain incomplete coverage of the underlying requirements. Some requirements are fully supported, some are only partially addressed, and others are intentionally missing. This allows the retrieval system to operate in a setting that more closely resembles a real audit process.

Since this project is intended as a small prototype, a single synthetic company report was generated for each topic. To increase retrieval difficulty and introduce additional noise, I also included academic publications related to the food domains. This creates a more realistic retrieval environment where relevant evidence competes with semantically related but operationally irrelevant documents.

Only two "company" dataset were added `co1` and `co2` corresponding to the food and health topic respectively. **Nonetheless only `co1` was considered for the Chat demo**. `co2` is only included as a proof of concept for having multiple indices in the dvc pipelines.

The data and generated artifacts are managed with DVC.

Pull available DVC data:

```bash
dvc pull
```

Rebuild all pipeline stages:

```bash
dvc repro
```

The pipeline currently:

1. Extracts chunks from PDFs under `data/friends/co1`.
2. Extracts chunks from PDFs under `data/friends/co2`.
3. Builds a Chroma index at `output/co1/index`.
4. Builds a Chroma index at `output/co2/index`.

The API currently loads the `co1` index from `output/co1/index` with collection name `c1_index`.

## Run The App

Start the FastAPI server:

```bash
uvicorn chat.api.server:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Useful endpoints:

- `GET /health` checks whether the API is running.
- `POST /retrieve` returns retrieved chunks for a question.
- `POST /chat` streams answer deltas and supporting chunks as server-sent events.

Example retrieval request:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say?"}'
```

## Docker

The image is published to GitHub Container Registry on pushes to `main`:

```bash
docker pull ghcr.io/labadier/intact:latest
```

Run the published API container:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e OPENAI_MODEL="$OPENAI_MODEL" \
  ghcr.io/labadier/intact:latest
```

Images are tagged as `latest` and with the Git commit SHA. To build a local image instead:

```bash
docker build -t intact .
```

## Testing

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Configuration

Important configuration files and environment variables:

- `params.yaml` controls the embedding model, prompt names, and index settings.
- `dvc.yaml` defines the chunk extraction and index creation pipeline.
- `OPENAI_API_KEY` is required for chat generation.
- `OPENAI_MODEL` selects the chat model.
- `OPENAI_API_BASE` selects the OpenAI-compatible API endpoint.

### Next Steps 

  - This prototype represents the first step towards an automated auditing/certification assistant. The data is conceptually divided into two categories: (i) company-specific evidence documents, which contain the information used to answer questions and support audit findings, and (ii) domain guidelines and certification requirements (which is independent of the specific companies), which define the criteria against which the evidence is evaluated. A natural next step would be to move from question answering to requirement assessment, where the system iteratively evaluates each requirement, retrieves supporting evidence, identifies potential gaps, and ultimately generates both detailed and executive-level audit reports.

  - The current implementation treats each user query independently and does not maintain conversational context across interactions. A natural extension would be to introduce context-aware retrieval, including conversation summarization, context pruning, and query reformulation. This would allow follow-up questions to be rewritten into self-contained queries, improving retrieval quality and enabling more meaningful multi-turn interactions.

  - While the current system focuses on retrieval and question answering, audit and certification workflows might require information that has to be computed or synthetized out of the text contained in the company reports, therefore this is a good scenario for introducing, specialization in some sort of agentic behaviour, where scores, scales, information synthesis, completion and comparison can be decoupled from the purely retrieval / generation components.

  - The current retrieval pipeline operates solely on chunk content and does not leverage document-level metadata. This can negatively impact retrieval quality, as some passages only become meaningful when considered within their broader context. Future iterations could enrich the retrieval process with metadata such as document type, section title, publication date (to discard overlapping and contradictory information), audit scope, or document.


- In the same way, all the information contained in a chunk is not relevant and this redundancy can provoke noise as well. Therefore to account for token usage efficency  and noise reduction, I would introduce a token-classifier span extractor. Maybe in a first iteration would just rely on an LLM to extract relevant sentences/spans. This also can be used, next to the inline cross-referencing to support the interpretability on the answers gotten from the LLM.

- The evaluation is not covered in this example, but it would be one of the first areas I would address in as a next step. For the retrieval component, I would start by creating relevance annotations between questions and supporting evidence spans. Since obtaining high-quality labels is expensive, an initial approach could rely on synthetic data generation. Rather than starting from questions, I would generate candidate questions from document chunks and treat the originating chunks as silver-label evidence. This would provide a scalable way to asses retrieval quality, compare retrieval strategies, and create the basis for a potential human annotation round. Ultimately, once an evaluation set is generated we could use the aggregation (or not :) of different LLM families to provide a more robust evaluation when it comes to an evaluation set.

- Beyond the retrieval performance at the level of queries, it would be usefull to check how the coverage intra-business is, in this case it can be more challenging to obtain a really good discriminative representation of the chunks embeddings and therefore, relevant chunks can be dominated by non-relevant ones. Also, a groundth truth is necessary on the facts required by the generated answers.

- Aditionally DVC was introduced to keep track of the inprovements based on the metrics designed from the two previous points.

- I introduced here the `unstructured` library for parsing because it offers capabilities of OCR-ing tables, extracting images and this paves the way for a next iteration where we make better use of the tables in the documents, since at this point they are just considered as regular text in lines. 

- The embeddings take some time to be computed whenever a change in the dataset is made. I would cache them to make the recomputation faster in case they change dinamically. This cache could exist even across clients. I also, kept the indices separated in purpose because I wanted to be able to modify the raw chunks with no cross-company impact as well as modify the embeddings, encoders, prompt used for computing the embeddings from one friend without impact in another clients

- Finally, there were several abbreviations I noticed this approach was sensitive to. I would introduce an abbreviation resolver + a simple disambiguation method, maybe some based in a tiny-embedding model.
