This repository contains a document question-answering prototype built around a RAG workflow. It serves a FastAPI chat interface that retrieves evidence from a local Chroma index and streams answers with inline references to the supporting chunks of text.

## Data Stored In The Project

The project stores two types of data:

- `data/friends`: company-specific evidence documents. In this prototype, each company folder contains synthetic audit reports that simulate the information an auditor would inspect.
- `data/topics`: topic-specific audit knowledge. These folders contain guideline or checklist material grouped by knowledge domain, such as `food` and `health`, and are independent from companies. This data should serve to cover the auditing process of multiple companies (friends).

The chat demo currently uses the company-specific evidence data from `data/friends/co1`. The topic data exists to represent the audit criteria side of the project, but it is not directly served by the current chat application.

## What The Demo Serves

The running application serves the data relative to `co1` only. The `co1` dataset represents a food-domain company demo. It contains a synthetic company report plus additional food-domain publications that introduce retrieval noise to simulate real-life complexity. The second dataset, `co2`, is included in the DVC pipeline as a proof of concept for handling multiple company indexes, but it is not loaded by the chat application.

At runtime, a chat request follows this path:

1. The user submits a question through the browser UI or `POST /chat`.
2. The app retrieves the top matching chunks from `output/co1/index`.
3. The retrieved chunks and question are inserted into the answer prompt.
4. An OpenAI-compatible chat model streams the answer.
5. The UI displays inline references such as `<doc1>` and shows only the chunks cited in the answer.

## What It Does

- Parses PDF files into reusable text chunks.
- Builds local Chroma indexes with Hugging Face embeddings.
- Retrieves the most relevant chunks for a user question.
- Streams generated answers through an OpenAI-compatible chat completion API.
- Adds inline references to relevant source chunks.
- Serves a small browser-based chat UI at the API root.
- Tracks source data and generated artifacts with DVC.

## Example Chat Requests

These prompts can be pasted into the chat UI to quickly test retrieval, answer generation, and inline references:

- What is the name of the company mentioned in the available documents?
- Which departments or responsible teams are mentioned in the company report?
- What does the available evidence say about meat handling practices?
- Summarize the most relevant findings from the retrieved evidence.

> For further questions related to this auditing kind of process please refer to `data/topics/food/safety_checklist.txt`

## Indexing Pipeline

The DVC pipeline extracts PDF text from each company folder and builds one Chroma index per company:

| Dataset | Source PDFs | Chunks | Chroma index | Collection | Served by app |
| --- | --- | --- | --- | --- | --- |
| `co1` | `data/friends/co1` | `output/co1/chunks.json` | `output/co1/index` | `c1_index` | Yes |
| `co2` | `data/friends/co2` | `output/co2/chunks.json` | `output/co2/index` | `c2_index` | No |

The embedding model and retrieval settings are configured in `params.yaml`. By default, the project uses [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), with `top_k` set to `10`.

Pull available DVC data and generated artifacts:

```bash
dvc pull
```

Rebuild all pipeline stages:

```bash
dvc repro
```

## Project Structure

```text
.
├── chat/
│   ├── api/              # FastAPI app, routes, and chat endpoints
│   ├── prompts/          # Prompt templates used for answer generation
│   └── ui/               # Static chat UI
├── data/                 # DVC-tracked source document data
├── models/               # ChatManager retrieval and generation logic
├── output/               # Generated chunks and Chroma indexes
├── scripts/              # Pipeline scripts for chunking and indexing
├── tests/                # Unit and API tests
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Embedding model and indexing parameters
├── pyproject.toml        # Python package and tool configuration
├── uv.lock               # Locked Python dependency graph
└── Dockerfile            # Container entrypoint for the API server
```

## Requirements

- Python 3.11 or newer
- `uv` for dependency management
- DVC, including the SSH remote support used by this project
- An OpenAI-compatible API key for answer generation

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Configure the language model provider:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-llm-name"
```

Optional provider override:

```bash
export OPENAI_API_BASE="https://api.mistral.ai/v1"
```

`OPENAI_API_BASE` defaults to `https://api.mistral.ai/v1` in the current code.

## Run The App

Start the FastAPI server:

```bash
uvicorn chat.api.server:app --reload
```

Useful endpoints:

- `GET /health` checks whether the API is running.
- `POST /retrieve` returns retrieved chunks for a question.
- `POST /chat` streams answer deltas and supporting chunks as server-sent events.


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

Images are tagged as `latest` and with the Git commit SHA. To build a local image instead, make sure `output/co1/index` exists locally first:

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

### Next Steps 

  - This prototype represents the first step towards an automated auditing/certification assistant. The data is conceptually divided into two categories: (i) company-specific evidence documents, which contain the information used to answer questions and support audit findings, and (ii) domain guidelines and certification requirements (which is independent of the specific companies), which define the criteria against which the evidence is evaluated. One next step would be to **move from question answering to requirement assessment**, where the system iteratively evaluates each requirement, retrieves the necessary supporting evidence, identifies potential gaps in the auditing process, and ultimately generates both detailed and executive-level audit reports.

  - The current implementation treats each user query independently and does not maintain conversational context across interactions. An extension should be to introduce context-aware retrieval, including conversation summarization, context pruning, and query reformulation. This would allow follow-up questions to be rewritten into self-contained queries, improving retrieval quality and enabling more meaningful multi-turn interactions.

  - While the current system focuses on retrieval and question answering, audit and certification workflows might require information that has to be computed or synthetized out of the text contained in the company reports, therefore this is a good scenario for introducing, specialization in some sort of agentic behaviour, where intermediate scores, scales, information synthesis, completions and comparisons can be decoupled from the purely retrieval / generation components.

  - The current retrieval pipeline operates solely on chunk content and does not leverage document-level metadata. This can negatively impact retrieval quality, as some passages only become meaningful when considered within their broader context. Future iterations should enrich the retrieval process with metadata such as document type, section title, publication date (to discard overlapping and contradictory information), audit scope, or document.

- In the same way, all the information contained in a chunk is not relevant and this redundancy can provoke noise as well. Therefore to account for token usage efficency  and noise reduction, I would introduce a **token-classifier span extractor**. Maybe in a first iteration would just rely on an LLM to extract relevant sentences/spans. This also can be used, next to the inline cross-referencing to support the interpretability on the answers gotten from the LLM.

- The evaluation is not covered in this example, but it would be one of the first areas I would address in as a next step. For the retrieval component, I would start by creating relevance annotations between questions and supporting evidence spans. Since obtaining high-quality labels is expensive, an initial approach could rely on synthetic data generation. Rather than starting from questions, I would generate candidate questions from document chunks and treat the originating chunks as silver-label evidence. This would provide a scalable way to asses retrieval quality, compare retrieval strategies, and create the basis for a potential human annotation round. Ultimately, once an evaluation set is generated we could use the aggregation (or not :) of different LLM families to provide a more robust evaluation when it comes to an evaluation set.

- Beyond the retrieval performance at the level of queries, it would be usefull to check how the coverage intra-business is, in this case it can be more challenging to obtain a really good discriminative representation of the chunks embeddings and therefore, relevant chunks can be dominated by non-relevant ones. Also, groundth truth is necessary on the facts required by the generated answers to easily estimate a coverage score.

- Aditionally DVC was introduced to keep track of the inprovements based on the metrics designed from the two previous points.

- I introduced here the `unstructured` library for parsing because it offers capabilities of OCR-ing tables, extracting images and this paves the way for a next iteration where we make better use of the tables in the documents, since at this point they are just considered as regular text in lines. 

- The embeddings take some time to be computed whenever a change in the dataset is made. I would cache them to make the recomputation faster in case they change dinamically. This cache could exist even across clients. I also, kept the indices separated in purpose because I wanted to be able to modify the raw chunks with no cross-company impact as well as modify the embeddings, encoders, prompt used for computing the embeddings from one friend without impact in another clients

- Finally, there were several abbreviations I noticed this approach was sensitive to. I would introduce an abbreviation resolver + a simple disambiguation method, maybe some based in a tiny-embedding model.
