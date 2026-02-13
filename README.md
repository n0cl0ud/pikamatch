# PikaMatch — Image Matching Pipeline (JPG vs PDF)

Local AI pipeline that matches JPG images against images inside PDF documents and extracts their associated descriptions. Runs 100% locally on GPU via Docker Compose.

## Use Case

The team receives:
- **JPG files** — reference images (from emails, etc.)
- **PDF files** — catalogues/sheets containing images with descriptive metadata (size, origin, reference, format, etc.) in varying layouts (tables, lists, free text, columns)

The system:
1. Finds which image in the PDF matches the JPG visually
2. Extracts the description/metadata fields associated with that image
3. Handles different image sizes (the JPG might be 2000x1500 while the PDF version is 300x200)
4. Handles varying PDF layouts — descriptions are NOT always in the same format/position

---

## Architecture

Two-phase pipeline, all in Docker Compose:

```
Phase 1: CLIP + pHash (fast, ~50ms)
  -> Visual matching — finds the best image match regardless of size differences

Phase 2: Qwen2.5-VL 7B (smart, ~3-5s)
  -> Renders the PDF page zone around the matched image
  -> Sends the screenshot to the VLM which "reads" it like a human
  -> Extracts structured fields regardless of layout
```

---

## Workflow

```
 INPUT                                                               OUTPUT
 =====                                                               ======
 +---------+  +-------------+                          +------------------+
 |  JPG    |  |  PDF         |                          | Structured JSON  |
 | (photo) |  | (catalogue)  |                          | + scores         |
 +----+----+  +------+------+                          | + description    |
      |              |                                  +--------+---------+
      |              |                                           |
======|==============|===========================================|=========
      |              |         DOCKER COMPOSE                    |
      |              |         (100% local GPU)                  |
      |              |                                           |
+-----v--------------v------------------------------------------+--------+
|                                                                         |
|  clip-api  (port 8002)                    FastAPI + CLIP + pHash        |
|  ================================================================       |
|                                                                         |
|  +---------------------------------------------------------------+      |
|  | PHASE 1 -- Fast visual matching (~50ms)                       |      |
|  |                                                                |      |
|  |  JPG --+                                                       |      |
|  |        +---> CLIP ViT-L/14 ---> 768d embedding --+             |      |
|  |        |    (1.5 GB VRAM)                        |             |      |
|  |        |                                   cosine similarity   |      |
|  |        |                                    (weight: 60%)      |      |
|  |  PDF --+                                         |             |      |
|  |        |    +--------------+                      |             |      |
|  |        +---> |  PyMuPDF     |                     +---> COMBINED|      |
|  |        |    |  (fitz)      |                     |     SCORE   |      |
|  |        |    |              |                     |             |      |
|  |        |    | - extract    |                     |             |      |
|  |        |    |   images     |    pHash 256-bit    |             |      |
|  |        |    | - bboxes     +---> hamming distance-+             |      |
|  |        |    | - pages      |    (weight: 40%)                  |      |
|  |        |    +--------------+                                    |      |
|  |        |                                                       |      |
|  |  Score = 0.60 x CLIP_cosine + 0.40 x pHash_similarity         |      |
|  |                                                                |      |
|  |  +--------------------+-----------+------------------------+   |      |
|  |  | VERDICT            | SCORE     | pHash distance         |   |      |
|  |  +--------------------+-----------+------------------------+   |      |
|  |  | identical          | > 0.95    | 0-2                    |   |      |
|  |  | very_similar       | > 0.90    | 3-8                    |   |      |
|  |  | similar            | > 0.80    | 9-16                   |   |      |
|  |  | related            | > 0.70    | 17-28                  |   |      |
|  |  | different          | < 0.70    | > 28                   |   |      |
|  |  +--------------------+-----------+------------------------+   |      |
|  +--------------------------------+-------------------------------+      |
|                                   |                                      |
|                             top_k matches                                |
|                                   |                                      |
|  +--------------------------------v-------------------------------+      |
|  | PHASE 2 -- Smart VLM extraction (~3-5s)                       |      |
|  |                                                                |      |
|  |  For each match:                                               |      |
|  |                                                                |      |
|  |  +-------------+     +---------------------------+             |      |
|  |  | PDF page    |     |  Rendered zone (200 DPI)  |             |      |
|  |  |             |     |                           |             |      |
|  |  |    +---+    |     |  ^ margin 2x (headers)   |             |      |
|  |  |    |IMG|    | --> |  +---------------------+  |             |      |
|  |  |    +---+    |     |  | image + surrounding |--+--> PNG      |      |
|  |  |  text...    |     |  | context & tables    |  |   (max      |      |
|  |  |  ref: xxx   |     |  +---------------------+  |   1200px)   |      |
|  |  |             |     |  v margin 3x (captions)|  |             |      |
|  |  +-------------+     |  < 1x >           4x > |  |             |      |
|  |                       +---------------------------+             |      |
|  |                                   |                            |      |
|  +-----------------------------------+----------------------------+      |
|                                      |                                   |
|                                 PNG base64                               |
|                                      |                                   |
+--------------------------------------+-----------------------------------+
                                       |
                              HTTP POST (internal)
                         /v1/chat/completions
                                       |
+--------------------------------------v-----------------------------------+
|                                                                          |
|  qwen-vl  (port 8001)              vLLM + Qwen2.5-VL-7B-AWQ            |
|  ===========================================================            |
|                                                                          |
|  +-----------------------------------------------------------+          |
|  |  Qwen2.5-VL-7B-Instruct-AWQ  (5 GB VRAM)                 |          |
|  |                                                            |          |
|  |  "Read this PDF zone like a human would"                   |          |
|  |                                                            |          |
|  |  Prompt --> Extract JSON fields:                           |          |
|  |  - description    - size         - origin                  |          |
|  |  - reference      - format       - color                   |          |
|  |  - category       - date         - author                  |          |
|  |  - status         - version      - other                   |          |
|  |                                                            |          |
|  |  Temperature: 0.05 (near-deterministic)                    |          |
|  +-----------------------------------------------------------+          |
|                                                                          |
+--------------------------------------------------------------------------+
```

---

## Interfaces

```
  +------------------+   +------------------+   +----------------------+
  |  Streamlit UI    |   |  REST API        |   |  CLI                 |
  |  port 8501       |   |  port 8002       |   |  test_clip.py        |
  |                  |   |                  |   |                      |
  |  - Single match  |   |  POST /match     |   |  match jpg pdf       |
  |  - Batch N:1     |   |  POST /match-batch|  |  batch pdf imgs...   |
  |  - PDF preview   |   |  POST /extract   |   |  extract pdf         |
  |                  |   |  GET  /health    |   |  health              |
  +------------------+   +------------------+   |                      |
                                                |  --json --csv        |
                                                |  --markdown --minimal|
                                                +----------------------+
```

---

## GPU Infrastructure

```
  +----------------------------------------------------------------------+
  |  NVIDIA RTX 5080 -- 16 GB VRAM                                      |
  |  +-------------------------------------------------------------+    |
  |  | CLIP 1.5 GB |  Qwen2.5-VL 5.0 GB  |  KV cache  |  free    |    |
  |  |             |                      |   ~3.5 GB  |  ~6 GB   |    |
  |  +-------------------------------------------------------------+    |
  |  PyTorch 2.10.0+cu128  |  CUDA sm_120 (Blackwell)                  |
  +----------------------------------------------------------------------+
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Visual matching | CLIP ViT-L/14 (OpenAI) | Semantic image embeddings + cosine similarity |
| Resize robustness | imagehash (pHash) | Perceptual hashing — robust to resize/compression |
| PDF extraction | PyMuPDF (fitz) | Extract embedded images + their bounding boxes |
| Description reading | Qwen2.5-VL-7B-Instruct-AWQ via vLLM | Vision LLM reads PDF page zones to extract fields |
| API | FastAPI | REST API for matching, extraction, batch |
| UI | Streamlit | Drag-and-drop testing interface |
| Serving | Docker Compose with NVIDIA runtime | Everything containerized |
| VLM inference | vLLM (vllm/vllm-openai image) | Efficient GPU inference with OpenAI-compatible API |

---

## Project Structure

```
pikamatch/
├── README.md              # This file
├── .env                   # HF_TOKEN (gitignored)
├── .env.example           # Template
├── docker-compose.yml     # All 3 services
├── Dockerfile.api         # CLIP + pHash + orchestration API
├── Dockerfile.ui          # Streamlit UI
├── api.py                 # Main API (FastAPI)
├── ui.py                  # Streamlit UI
└── test_clip.py           # CLI test tool
```

---

## Quick Start

```bash
# 1. Set your HuggingFace token (optional for public models)
cp .env.example .env
# edit .env if needed

# 2. Launch everything
docker compose up --build
# First start: ~5-10 min (model download, cached after)
# Subsequent starts: ~2 min

# 3. Access
# UI:      http://localhost:8501
# API:     http://localhost:8002/docs  (Swagger)
# VLM:     http://localhost:8001/health
```

---

## CLI Usage

```bash
# Health check
python test_clip.py health

# Match 1 JPG against 1 PDF
python test_clip.py match photo.jpg catalogue.pdf

# Output formats
python test_clip.py match photo.jpg catalogue.pdf --json
python test_clip.py match photo.jpg catalogue.pdf --csv
python test_clip.py match photo.jpg catalogue.pdf --markdown
python test_clip.py match photo.jpg catalogue.pdf --minimal

# Preview PDF extraction
python test_clip.py extract catalogue.pdf

# Batch: multiple JPGs vs 1 PDF
python test_clip.py batch catalogue.pdf img1.jpg img2.jpg img3.png
```

---

## API Endpoints

### `POST /match` — Core endpoint
Upload 1 JPG + 1 PDF → find match + extract description.

**Parameters:**
- `jpg`: image file (JPG/PNG/WebP/BMP)
- `pdf`: PDF file
- `top_k`: int (default 3) — number of matches to return
- `threshold`: float (default 0.0) — minimum combined score

### `POST /match-batch` — Batch matching
Upload N JPGs + 1 PDF → for each JPG, find its match and extract description.

### `POST /extract` — Preview extraction
Upload 1 PDF → extract all images + run VLM on each to read descriptions.

### `GET /health` — Service health check
Returns CLIP status, VLM status, VRAM usage.

---

## Scoring System

**Combined Score:**
```
score = 0.60 x CLIP_cosine + 0.40 x pHash_similarity
```

- **CLIP** (60%): Semantic similarity — understands visual content. Handles different sizes natively.
- **pHash** (40%): Structural perceptual hash — robust to resize, compression, minor edits.

---

## Requirements

- **GPU**: NVIDIA with 16+ GB VRAM (tested on RTX 5080)
- **Docker**: Docker Desktop with GPU support (NVIDIA Container Toolkit)
- **OS**: Linux, Windows (WSL2), macOS (CPU only)
