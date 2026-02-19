"""
PikaMatch API — Image matching pipeline (JPG vs PDF)
Phase 1: CLIP + pHash (fast visual matching)
Phase 2: Qwen2.5-VL (structured field extraction from PDF zones)
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from typing import Optional

import uuid

import clip
import fitz  # PyMuPDF
import httpx
import imagehash
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pikamatch")

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VLM_URL = os.getenv("VLM_URL", "http://qwen-vl:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = "pdf_images"
CLIP_DIM = 768  # CLIP ViT-L/14 embedding dimension
INDEXED_PDFS_DIR = "/app/indexed_pdfs"
CLIP_WEIGHT = 0.60
PHASH_WEIGHT = 0.40
PHASH_HASH_SIZE = 16
PHASH_MAX_BITS = PHASH_HASH_SIZE * PHASH_HASH_SIZE  # 256
MIN_IMAGE_BYTES = 2048  # skip < 2KB (tracking pixels)
MIN_IMAGE_DIM = 50  # skip < 50px
RENDER_DPI = 200
MAX_VLM_PX = 1200

# --- VLM extraction prompt ---
VLM_PROMPT = """Tu es un assistant spécialisé dans l'extraction de métadonnées d'images à partir de documents PDF.

Regarde attentivement cette capture d'écran d'une zone de page PDF. Elle contient une image et potentiellement sa description associée (dans un tableau, une liste, du texte libre, des colonnes, etc.).

Extrais les champs suivants au format JSON. Mets null pour les champs absents :

{
  "description": "description textuelle de l'image",
  "taille": "dimensions (ex: 300x225 px)",
  "origine": "source/fournisseur",
  "reference": "code ou référence",
  "format": "format fichier (JPEG, PNG, etc.)",
  "couleur": "couleur dominante ou palette",
  "categorie": "catégorie/type d'image",
  "date": "date associée",
  "auteur": "auteur/créateur",
  "statut": "statut (validé, brouillon, etc.)",
  "version": "numéro de version",
  "autres": "autres informations pertinentes"
}

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""

# --- Load CLIP model at startup ---
logger.info(f"Loading CLIP ViT-L/14 on {DEVICE}...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=DEVICE)
clip_model.eval()
logger.info("CLIP model loaded.")

# --- Qdrant setup ---
os.makedirs(INDEXED_PDFS_DIR, exist_ok=True)
qdrant = QdrantClient(url=QDRANT_URL, timeout=300.0)
try:
    qdrant.get_collection(QDRANT_COLLECTION)
    logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists.")
except Exception:
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
    )
    logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")

app = FastAPI(title="PikaMatch API", version="1.0.0")


# ============================================================
# Utility functions
# ============================================================

def embed_image(img: Image.Image) -> np.ndarray:
    """Compute CLIP embedding for a PIL image."""
    tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(tensor)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def phash_similarity(img_a: Image.Image, img_b: Image.Image) -> dict:
    """Compute perceptual hash similarity between two images."""
    hash_a = imagehash.phash(img_a, hash_size=PHASH_HASH_SIZE)
    hash_b = imagehash.phash(img_b, hash_size=PHASH_HASH_SIZE)
    distance = int(hash_a - hash_b)
    similarity = 1.0 - (distance / PHASH_MAX_BITS)
    if distance <= 2:
        verdict = "identical"
    elif distance <= 8:
        verdict = "very_similar"
    elif distance <= 16:
        verdict = "similar"
    elif distance <= 28:
        verdict = "related"
    else:
        verdict = "different"
    return {"similarity": round(similarity, 4), "distance": distance, "verdict": verdict}


def compute_phash_hex(img: Image.Image) -> str:
    """Compute perceptual hash and return as hex string."""
    return str(imagehash.phash(img, hash_size=PHASH_HASH_SIZE))


def phash_similarity_from_hex(hex_a: str, hex_b: str) -> dict:
    """Compute pHash similarity from two hex strings."""
    hash_a = imagehash.hex_to_hash(hex_a)
    hash_b = imagehash.hex_to_hash(hex_b)
    distance = int(hash_a - hash_b)
    similarity = 1.0 - (distance / PHASH_MAX_BITS)
    if distance <= 2:
        verdict = "identical"
    elif distance <= 8:
        verdict = "very_similar"
    elif distance <= 16:
        verdict = "similar"
    elif distance <= 28:
        verdict = "related"
    else:
        verdict = "different"
    return {"similarity": round(similarity, 4), "distance": distance, "verdict": verdict}


def combined_score(clip_score: float, phash_sim: float) -> float:
    return round(CLIP_WEIGHT * clip_score + PHASH_WEIGHT * phash_sim, 4)


def score_verdict(score: float) -> str:
    if score > 0.95:
        return "identical"
    elif score > 0.90:
        return "very_similar"
    elif score > 0.80:
        return "similar"
    elif score > 0.70:
        return "related"
    return "different"


def extract_pdf_images(pdf_bytes: bytes) -> list[dict]:
    """Extract embedded images from a PDF with bounding boxes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []
    seen_hashes = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        images = page.get_images(full=True)

        for img_info in images:
            xref = img_info[0]
            try:
                extracted = doc.extract_image(xref)
            except Exception:
                continue

            img_bytes = extracted["image"]
            if len(img_bytes) < MIN_IMAGE_BYTES:
                continue

            md5 = hashlib.md5(img_bytes).hexdigest()
            if md5 in seen_hashes:
                continue
            seen_hashes.add(md5)

            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue

            if pil_img.width < MIN_IMAGE_DIM or pil_img.height < MIN_IMAGE_DIM:
                continue

            rects = page.get_image_rects(xref)
            bbox = rects[0] if rects else None

            results.append({
                "page": page_idx,
                "xref": xref,
                "image": pil_img,
                "image_bytes": img_bytes,
                "bbox": bbox,
                "width": pil_img.width,
                "height": pil_img.height,
            })

    # Fallback for scanned PDFs: render pages as images
    if not results:
        logger.info("No embedded images found — falling back to page renders")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            results.append({
                "page": page_idx,
                "xref": None,
                "image": pil_img,
                "image_bytes": img_bytes,
                "bbox": page.rect,
                "width": pil_img.width,
                "height": pil_img.height,
            })

    doc.close()
    return results


def render_page_zone(pdf_bytes: bytes, page_idx: int, bbox) -> Image.Image:
    """Render the zone around an image bbox on a PDF page with generous margins."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_idx]
    page_rect = page.rect

    # Ensure bbox is a valid Rect
    if not isinstance(bbox, fitz.Rect):
        bbox = fitz.Rect(bbox)

    # Fallback if bbox is empty/invalid — render full page
    if bbox.is_empty or bbox.is_infinite or bbox.width < 1 or bbox.height < 1:
        pix = page.get_pixmap(dpi=RENDER_DPI)
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        doc.close()
        if max(pil_img.width, pil_img.height) > MAX_VLM_PX:
            ratio = MAX_VLM_PX / max(pil_img.width, pil_img.height)
            new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
        return pil_img

    bw = bbox.width
    bh = bbox.height
    margin = max(bw, bh) * 0.5

    clip_rect = fitz.Rect(
        max(page_rect.x0, bbox.x0 - margin * 1),       # left: 1x
        max(page_rect.y0, bbox.y0 - margin * 2),       # above: 2x
        min(page_rect.x1, bbox.x1 + margin * 4),       # right: 4x
        min(page_rect.y1, bbox.y1 + margin * 3),       # below: 3x
    )

    # Ensure clip_rect is valid
    clip_rect = clip_rect & page_rect  # intersect with page bounds
    if clip_rect.is_empty or clip_rect.width < 1 or clip_rect.height < 1:
        clip_rect = page_rect

    pix = page.get_pixmap(dpi=RENDER_DPI, clip=clip_rect)
    img_bytes = pix.tobytes("png")
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Resize to max dimension for VLM efficiency
    if max(pil_img.width, pil_img.height) > MAX_VLM_PX:
        ratio = MAX_VLM_PX / max(pil_img.width, pil_img.height)
        new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

    doc.close()
    return pil_img


def image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def parse_vlm_json(text: str) -> dict:
    """Parse JSON from VLM response, stripping markdown code blocks if present."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_text": text}


async def call_vlm(zone_img: Image.Image) -> dict:
    """Send a page zone screenshot to Qwen2.5-VL and extract description fields."""
    b64 = image_to_b64(zone_img)
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{VLM_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    return parse_vlm_json(content)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health")
async def health():
    clip_ok = clip_model is not None
    vlm_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{VLM_URL}/health")
            vlm_ok = r.status_code == 200
    except Exception:
        pass

    qdrant_ok = False
    qdrant_vectors = 0
    try:
        collection = qdrant.get_collection(QDRANT_COLLECTION)
        qdrant_ok = True
        qdrant_vectors = collection.points_count
    except Exception:
        pass

    vram_info = {}
    if torch.cuda.is_available():
        vram_info = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
        }

    return {
        "clip": "ok" if clip_ok else "error",
        "vlm": "ok" if vlm_ok else "error",
        "qdrant": "ok" if qdrant_ok else "error",
        "qdrant_vectors": qdrant_vectors,
        "device": DEVICE,
        "vram": vram_info,
    }


@app.post("/match")
async def match(
    jpg: UploadFile = File(...),
    pdf: UploadFile = File(...),
    top_k: int = Form(3),
    threshold: float = Form(0.0),
):
    t_start = time.time()

    # Read inputs
    jpg_bytes = await jpg.read()
    pdf_bytes = await pdf.read()
    ref_img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    ref_embedding = embed_image(ref_img)
    ref_phash_hex = compute_phash_hex(ref_img)
    pdf_filename = pdf.filename or "unknown.pdf"

    # Check if this PDF is already indexed in Qdrant
    pdf_indexed = False
    try:
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        pdf_indexed = len(scroll_result[0]) > 0
    except Exception:
        pass

    if pdf_indexed:
        # === FAST PATH: use Qdrant ===
        logger.info(f"PDF '{pdf_filename}' found in Qdrant — using indexed data")
        t_search_start = time.time()
        search_results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=ref_embedding.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
            ),
            limit=top_k * 3,
            with_payload=True,
        ).points
        t_search_end = time.time()

        matches = []
        for result in search_results:
            clip_sim = result.score
            payload = result.payload
            phash_result = phash_similarity_from_hex(ref_phash_hex, payload["phash_hex"])
            combo = combined_score(clip_sim, phash_result["similarity"])

            if combo >= threshold:
                matches.append({
                    "combined_score": combo,
                    "clip_score": round(clip_sim, 4),
                    "phash": phash_result,
                    "verdict": score_verdict(combo),
                    "page": payload["page"],
                    "size_vs_ref": {
                        "ref": f"{ref_img.width}x{ref_img.height}",
                        "pdf": f"{payload['width']}x{payload['height']}",
                        "scale_factor": round(
                            (payload["width"] * payload["height"])
                            / max(ref_img.width * ref_img.height, 1),
                            4,
                        ),
                    },
                    "description": payload.get("description") or {"error": "no description indexed"},
                    "zone_b64": payload.get("zone_b64"),
                })

        matches.sort(key=lambda x: x["combined_score"], reverse=True)
        matches = matches[:top_k]

        # Count total images for this PDF
        count_result = qdrant.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
            ),
        )
        images_found = count_result.count

        best = matches[0] if matches else None
        return {
            "reference": {"filename": jpg.filename, "size": f"{ref_img.width}x{ref_img.height}"},
            "pdf": {"filename": pdf_filename, "images_found": images_found},
            "best_match": best,
            "all_matches": matches,
            "timing": {
                "clip_matching_ms": round((t_search_end - t_search_start) * 1000),
                "total_ms": round((time.time() - t_start) * 1000),
            },
        }

    # === SLOW PATH: PDF not indexed, process from scratch ===
    logger.info(f"PDF '{pdf_filename}' not in Qdrant — processing from scratch")
    t_clip_start = time.time()
    pdf_images = extract_pdf_images(pdf_bytes)

    matches = []
    for pdf_img_info in pdf_images:
        pdf_img = pdf_img_info["image"]
        pdf_embedding = embed_image(pdf_img)
        clip_sim = cosine_similarity(ref_embedding, pdf_embedding)
        phash_result = phash_similarity(ref_img, pdf_img)
        combo = combined_score(clip_sim, phash_result["similarity"])

        if combo >= threshold:
            matches.append({
                "clip_score": round(clip_sim, 4),
                "phash": phash_result,
                "combined_score": combo,
                "verdict": score_verdict(combo),
                "page": pdf_img_info["page"],
                "bbox": pdf_img_info["bbox"],
                "pdf_image": pdf_img,
                "size_vs_ref": {
                    "ref": f"{ref_img.width}x{ref_img.height}",
                    "pdf": f"{pdf_img_info['width']}x{pdf_img_info['height']}",
                    "scale_factor": round(
                        (pdf_img_info["width"] * pdf_img_info["height"])
                        / max(ref_img.width * ref_img.height, 1),
                        4,
                    ),
                },
            })

    matches.sort(key=lambda x: x["combined_score"], reverse=True)
    matches = matches[:top_k]
    t_clip_end = time.time()

    # VLM extraction
    t_vlm_start = time.time()
    for m in matches:
        if m["bbox"] is not None:
            zone_img = render_page_zone(pdf_bytes, m["page"], m["bbox"])
            m["zone_b64"] = image_to_b64(zone_img)
            try:
                m["description"] = await call_vlm(zone_img)
            except Exception as e:
                logger.error(f"VLM call failed: {e}")
                m["description"] = {"error": str(e)}
        else:
            m["description"] = {"error": "no bounding box available"}
            m["zone_b64"] = None
    t_vlm_end = time.time()

    # Build response
    all_matches = []
    for m in matches:
        all_matches.append({
            "combined_score": m["combined_score"],
            "clip_score": m["clip_score"],
            "phash": m["phash"],
            "verdict": m["verdict"],
            "page": m["page"],
            "size_vs_ref": m["size_vs_ref"],
            "description": m["description"],
            "zone_b64": m.get("zone_b64"),
        })

    best = all_matches[0] if all_matches else None

    return {
        "reference": {"filename": jpg.filename, "size": f"{ref_img.width}x{ref_img.height}"},
        "pdf": {"filename": pdf.filename, "images_found": len(pdf_images)},
        "best_match": best,
        "all_matches": all_matches,
        "timing": {
            "clip_matching_ms": round((t_clip_end - t_clip_start) * 1000),
            "vlm_extraction_ms": round((t_vlm_end - t_vlm_start) * 1000),
            "total_ms": round((t_vlm_end - t_start) * 1000),
        },
    }


@app.post("/match-batch")
async def match_batch(
    pdf: UploadFile = File(...),
    images: list[UploadFile] = File(...),
    threshold: float = Form(0.70),
):
    t_start = time.time()
    pdf_bytes = await pdf.read()
    pdf_filename = pdf.filename or "unknown.pdf"

    # Check if this PDF is already indexed in Qdrant
    pdf_indexed = False
    try:
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        pdf_indexed = len(scroll_result[0]) > 0
    except Exception:
        pass

    if pdf_indexed:
        # === FAST PATH: use Qdrant ===
        logger.info(f"Batch: PDF '{pdf_filename}' found in Qdrant")
        count_result = qdrant.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
            ),
        )
        images_found = count_result.count

        results = []
        for img_file in images:
            img_bytes = await img_file.read()
            ref_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ref_embedding = embed_image(ref_img)
            ref_phash_hex = compute_phash_hex(ref_img)

            search_results = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=ref_embedding.tolist(),
                query_filter=Filter(
                    must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
                ),
                limit=3,
                with_payload=True,
            ).points

            best_match = None
            best_score = -1.0
            for result in search_results:
                clip_sim = result.score
                payload = result.payload
                phash_result = phash_similarity_from_hex(ref_phash_hex, payload["phash_hex"])
                combo = combined_score(clip_sim, phash_result["similarity"])
                if combo > best_score:
                    best_score = combo
                    best_match = {
                        "clip_score": round(clip_sim, 4),
                        "phash": phash_result,
                        "combined_score": combo,
                        "verdict": score_verdict(combo),
                        "page": payload["page"],
                        "size_vs_ref": {
                            "ref": f"{ref_img.width}x{ref_img.height}",
                            "pdf": f"{payload['width']}x{payload['height']}",
                        },
                        "description": payload.get("description") or {"error": "no description indexed"},
                        "zone_b64": payload.get("zone_b64"),
                    }

            entry = {"filename": img_file.filename, "matched": False, "match": None}
            if best_match and best_match["combined_score"] >= threshold:
                entry["matched"] = True
                entry["match"] = best_match
            results.append(entry)

        return {
            "pdf": {"filename": pdf_filename, "images_found": images_found},
            "results": results,
            "timing": {"total_ms": round((time.time() - t_start) * 1000)},
        }

    # === SLOW PATH: PDF not indexed ===
    logger.info(f"Batch: PDF '{pdf_filename}' not in Qdrant — processing from scratch")
    pdf_images = extract_pdf_images(pdf_bytes)
    pdf_embeddings = []
    for pdf_img_info in pdf_images:
        emb = embed_image(pdf_img_info["image"])
        pdf_embeddings.append(emb)

    results = []
    for img_file in images:
        img_bytes = await img_file.read()
        ref_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ref_embedding = embed_image(ref_img)

        best_match = None
        best_score = -1.0

        for i, pdf_img_info in enumerate(pdf_images):
            clip_sim = cosine_similarity(ref_embedding, pdf_embeddings[i])
            phash_result = phash_similarity(ref_img, pdf_img_info["image"])
            combo = combined_score(clip_sim, phash_result["similarity"])

            if combo > best_score:
                best_score = combo
                best_match = {
                    "clip_score": round(clip_sim, 4),
                    "phash": phash_result,
                    "combined_score": combo,
                    "verdict": score_verdict(combo),
                    "page": pdf_img_info["page"],
                    "bbox": pdf_img_info["bbox"],
                    "size_vs_ref": {
                        "ref": f"{ref_img.width}x{ref_img.height}",
                        "pdf": f"{pdf_img_info['width']}x{pdf_img_info['height']}",
                    },
                }

        entry = {"filename": img_file.filename, "matched": False, "match": None}

        if best_match and best_match["combined_score"] >= threshold:
            entry["matched"] = True
            if best_match["bbox"] is not None:
                zone_img = render_page_zone(pdf_bytes, best_match["page"], best_match["bbox"])
                best_match["zone_b64"] = image_to_b64(zone_img)
                try:
                    best_match["description"] = await call_vlm(zone_img)
                except Exception as e:
                    best_match["description"] = {"error": str(e)}
            else:
                best_match["description"] = {"error": "no bounding box"}
                best_match["zone_b64"] = None
            best_match.pop("bbox", None)
            entry["match"] = best_match

        results.append(entry)

    return {
        "pdf": {"filename": pdf.filename, "images_found": len(pdf_images)},
        "results": results,
        "timing": {"total_ms": round((time.time() - t_start) * 1000)},
    }


@app.post("/extract")
async def extract(pdf: UploadFile = File(...)):
    t_start = time.time()
    pdf_bytes = await pdf.read()
    pdf_images = extract_pdf_images(pdf_bytes)

    extractions = []
    for pdf_img_info in pdf_images:
        entry = {
            "page": pdf_img_info["page"],
            "size": f"{pdf_img_info['width']}x{pdf_img_info['height']}",
            "thumbnail_b64": image_to_b64(
                pdf_img_info["image"].copy().resize(
                    (min(200, pdf_img_info["image"].width), min(200, pdf_img_info["image"].height)),
                    Image.LANCZOS,
                )
            ),
        }

        if pdf_img_info["bbox"] is not None:
            zone_img = render_page_zone(pdf_bytes, pdf_img_info["page"], pdf_img_info["bbox"])
            entry["zone_b64"] = image_to_b64(zone_img)
            try:
                entry["description"] = await call_vlm(zone_img)
            except Exception as e:
                entry["description"] = {"error": str(e)}
        else:
            entry["zone_b64"] = None
            entry["description"] = {"error": "no bounding box"}

        extractions.append(entry)

    return {
        "pdf": {"filename": pdf.filename, "images_found": len(pdf_images)},
        "extractions": extractions,
        "timing": {"total_ms": round((time.time() - t_start) * 1000)},
    }


# ============================================================
# Qdrant Index / Scan Endpoints
# ============================================================

@app.post("/index")
async def index_pdfs(pdfs: list[UploadFile] = File(...), force: bool = Form(False)):
    """Index one or more PDFs into Qdrant for later searching."""
    t_start = time.time()
    indexed = []
    skipped = []

    for pdf_file in pdfs:
        pdf_bytes = await pdf_file.read()
        pdf_filename = pdf_file.filename or "unknown.pdf"

        # Check if already indexed — skip unless force
        if not force:
            try:
                scroll_result = qdrant.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if len(scroll_result[0]) > 0:
                    logger.info(f"Skipping '{pdf_filename}' — already indexed")
                    skipped.append(pdf_filename)
                    continue
            except Exception:
                pass

        # Skip empty/corrupted PDFs
        if not pdf_bytes or len(pdf_bytes) < 100:
            logger.warning(f"Skipping '{pdf_filename}' — empty or too small ({len(pdf_bytes)} bytes)")
            skipped.append(pdf_filename)
            continue

        try:
            # Save PDF to disk for later VLM rendering
            safe_name = pdf_filename.replace("/", "_").replace("\\", "_")
            pdf_path = os.path.join(INDEXED_PDFS_DIR, safe_name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            # Extract images
            pdf_images = extract_pdf_images(pdf_bytes)
        except Exception as e:
            logger.error(f"Failed to process '{pdf_filename}': {e}")
            skipped.append(pdf_filename)
            continue

        points = []

        for img_idx, img_info in enumerate(pdf_images):
            pil_img = img_info["image"]
            embedding = embed_image(pil_img)
            phash_hex = compute_phash_hex(pil_img)

            # Create thumbnail (max 150px) as base64 for preview
            thumb = pil_img.copy()
            thumb.thumbnail((150, 150), Image.LANCZOS)
            thumb_b64 = image_to_b64(thumb)

            # Serialize bbox
            bbox = img_info["bbox"]
            bbox_list = None
            if bbox is not None:
                if isinstance(bbox, fitz.Rect):
                    bbox_list = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
                else:
                    bbox_list = list(bbox)[:4]

            # VLM description extraction at index time
            description = None
            zone_b64 = None
            if bbox is not None:
                try:
                    zone_img = render_page_zone(pdf_bytes, img_info["page"], bbox)
                    zone_b64 = image_to_b64(zone_img)
                    description = await call_vlm(zone_img)
                    logger.info(f"  VLM extracted description for image {img_idx + 1}/{len(pdf_images)} in {pdf_filename}")
                except Exception as e:
                    logger.error(f"  VLM failed for image {img_idx + 1} in {pdf_filename}: {e}")
                    description = {"error": str(e)}

            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "pdf_filename": pdf_filename,
                    "page": img_info["page"],
                    "bbox": bbox_list,
                    "phash_hex": phash_hex,
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "thumbnail_b64": thumb_b64,
                    "description": description,
                    "zone_b64": zone_b64,
                },
            ))

        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i:i + 100],
            )

        indexed.append({"pdf": pdf_filename, "images": len(points)})
        logger.info(f"Indexed {len(points)} images from {pdf_filename}")

    total_vectors = qdrant.get_collection(QDRANT_COLLECTION).points_count

    return {
        "indexed": indexed,
        "skipped": skipped,
        "total_vectors": total_vectors,
        "timing_ms": round((time.time() - t_start) * 1000),
    }


@app.post("/scan")
async def scan(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: float = Form(0.70),
):
    """Search indexed PDFs for matches to the uploaded image."""
    t_start = time.time()

    # Embed query image
    img_bytes = await image.read()
    ref_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ref_embedding = embed_image(ref_img)
    ref_phash_hex = compute_phash_hex(ref_img)

    # Vector search in Qdrant
    t_search_start = time.time()
    search_results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=ref_embedding.tolist(),
        limit=top_k * 3,  # fetch extra to re-rank after pHash
        with_payload=True,
    ).points
    t_search_end = time.time()

    # Re-rank with combined score (CLIP + pHash)
    candidates = []
    for result in search_results:
        clip_sim = result.score
        payload = result.payload
        phash_result = phash_similarity_from_hex(ref_phash_hex, payload["phash_hex"])
        combo = combined_score(clip_sim, phash_result["similarity"])

        if combo >= threshold:
            candidates.append({
                "clip_score": round(clip_sim, 4),
                "phash": phash_result,
                "combined_score": combo,
                "verdict": score_verdict(combo),
                "pdf": payload["pdf_filename"],
                "page": payload["page"],
                "width": payload["width"],
                "height": payload["height"],
                "thumbnail_b64": payload.get("thumbnail_b64"),
                "description": payload.get("description"),
                "zone_b64": payload.get("zone_b64"),
            })

    # Sort by combined score and take top_k
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    candidates = candidates[:top_k]

    # Build results — descriptions already in Qdrant, no VLM call needed
    results = []
    for match in candidates:
        results.append({
            "pdf": match["pdf"],
            "page": match["page"],
            "combined_score": match["combined_score"],
            "clip_score": match["clip_score"],
            "phash": match["phash"],
            "verdict": match["verdict"],
            "size_vs_ref": {
                "ref": f"{ref_img.width}x{ref_img.height}",
                "pdf": f"{match['width']}x{match['height']}",
                "scale_factor": round(
                    (match["width"] * match["height"])
                    / max(ref_img.width * ref_img.height, 1),
                    4,
                ),
            },
            "thumbnail_b64": match.get("thumbnail_b64"),
            "description": match.get("description") or {"error": "no description indexed"},
            "zone_b64": match.get("zone_b64"),
        })

    return {
        "reference": {"filename": image.filename, "size": f"{ref_img.width}x{ref_img.height}"},
        "total_matches": len(results),
        "results": results,
        "timing": {
            "vector_search_ms": round((t_search_end - t_search_start) * 1000),
            "total_ms": round((time.time() - t_start) * 1000),
        },
    }


@app.get("/index/status")
async def index_status():
    """Return status of indexed PDFs."""
    collection = qdrant.get_collection(QDRANT_COLLECTION)
    total_vectors = collection.points_count

    # Get unique PDF filenames by scrolling points
    pdf_counts: dict[str, int] = {}
    offset = None
    while True:
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=["pdf_filename"],
            with_vectors=False,
        )
        points, next_offset = scroll_result
        for point in points:
            pdf_name = point.payload.get("pdf_filename", "unknown")
            pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
        if next_offset is None:
            break
        offset = next_offset

    pdfs = [{"pdf": name, "images": count} for name, count in sorted(pdf_counts.items())]

    return {
        "total_vectors": total_vectors,
        "total_pdfs": len(pdfs),
        "pdfs": pdfs,
    }


@app.delete("/index/{pdf_filename}")
async def delete_pdf_index(pdf_filename: str):
    """Remove all vectors for a specific PDF from the index."""
    # Delete points matching the pdf_filename
    qdrant.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_filename))]
        ),
    )

    # Remove stored PDF file
    safe_name = pdf_filename.replace("/", "_").replace("\\", "_")
    pdf_path = os.path.join(INDEXED_PDFS_DIR, safe_name)
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    total_vectors = qdrant.get_collection(QDRANT_COLLECTION).points_count

    return {
        "deleted": pdf_filename,
        "remaining_vectors": total_vectors,
    }


@app.delete("/index")
async def clear_index():
    """Clear the entire Qdrant index and stored PDFs."""
    # Recreate collection (fastest way to clear)
    qdrant.delete_collection(QDRANT_COLLECTION)
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
    )

    # Remove all stored PDFs
    for f in os.listdir(INDEXED_PDFS_DIR):
        fpath = os.path.join(INDEXED_PDFS_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)

    return {"status": "cleared", "total_vectors": 0}
