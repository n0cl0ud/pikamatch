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
import cv2
import fitz  # PyMuPDF
import httpx
import imagehash
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams

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
# When color mismatch (one B&W, one color), trust CLIP more
CLIP_WEIGHT_MISMATCH = 0.85
PHASH_WEIGHT_MISMATCH = 0.15
PHASH_HASH_SIZE = 16
PHASH_MAX_BITS = PHASH_HASH_SIZE * PHASH_HASH_SIZE  # 256
MIN_IMAGE_BYTES = 2048  # skip < 2KB (tracking pixels)
MIN_IMAGE_DIM = 50  # skip < 50px
RENDER_DPI = 200
MAX_VLM_PX = 1200
VLM_TIMEOUT = 120.0  # seconds — prevent infinite hangs on VLM calls
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB per file
MAX_CONCURRENT_REQUESTS = 4  # prevent GPU overload

# --- Security ---
API_KEY = os.getenv("API_KEY", "")
_active_requests = 0

# --- Sub-image segmentation constants ---
PAGE_COVERAGE_THRESHOLD = 0.60    # Image covering >60% of page triggers segmentation
MIN_SUBIMAGE_AREA_RATIO = 0.02    # Sub-image must be >2% of page area
MAX_SUBIMAGE_AREA_RATIO = 0.85    # Must be <85% (not the full page itself)
MIN_SUBIMAGE_DIM = 80             # Min 80px dimension

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


def _ensure_payload_indexes():
    """Create keyword indexes on frequently filtered fields (idempotent)."""
    for field in ("pdf_filename", "tag"):
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=False,
            )
            logger.info(f"Requested payload index on '{field}'.")
        except Exception:
            pass  # Already exists


_ensure_payload_indexes()

app = FastAPI(title="PikaMatch API", version="1.0.0")


# --- Auth dependency ---
async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key if one is configured. Skip auth if API_KEY is empty."""
    if not API_KEY:
        return  # No key configured — open access (dev mode)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Concurrency guard middleware ---
@app.middleware("http")
async def concurrency_guard(request: Request, call_next):
    global _active_requests
    # Skip guard for health checks
    if request.url.path == "/health":
        return await call_next(request)
    if _active_requests >= MAX_CONCURRENT_REQUESTS:
        return JSONResponse(status_code=503, content={"detail": "Server busy, try again later"})
    _active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_requests -= 1


# --- Upload size validation helper ---
async def validate_upload_size(file: UploadFile, label: str = "file"):
    """Read file bytes and reject if over MAX_UPLOAD_BYTES."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} too large ({len(data)} bytes, max {MAX_UPLOAD_BYTES})",
        )
    return data


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


def is_grayscale(img: Image.Image) -> bool:
    """Detect if an image is grayscale (B&W) by comparing RGB channels."""
    if img.mode == "L":
        return True
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Sample pixels — compare R, G, B channels
    small = img.resize((50, 50), Image.LANCZOS)
    arr = np.array(small, dtype=np.float32)
    # If R ~= G ~= B for most pixels, it's grayscale
    diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean()
    diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2]).mean()
    diff_gb = np.abs(arr[:, :, 1] - arr[:, :, 2]).mean()
    avg_diff = (diff_rg + diff_rb + diff_gb) / 3.0
    return bool(avg_diff < 10.0)  # threshold: less than 10 avg difference = grayscale


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


def combined_score(clip_score: float, phash_sim: float, color_mismatch: bool = False) -> float:
    if color_mismatch:
        return round(CLIP_WEIGHT_MISMATCH * clip_score + PHASH_WEIGHT_MISMATCH * phash_sim, 4)
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


def is_full_page_image(bbox, page_rect) -> bool:
    """Return True if bbox covers >60% of the page area (likely a full-page scan)."""
    if bbox is None or page_rect is None:
        return False
    if not isinstance(bbox, fitz.Rect):
        bbox = fitz.Rect(bbox)
    if not isinstance(page_rect, fitz.Rect):
        page_rect = fitz.Rect(page_rect)
    page_area = page_rect.width * page_rect.height
    if page_area <= 0:
        return False
    bbox_area = bbox.width * bbox.height
    return (bbox_area / page_area) > PAGE_COVERAGE_THRESHOLD


def segment_page_image(page_pil: Image.Image, page_rect, render_dpi: int = RENDER_DPI) -> list[dict]:
    """
    Use OpenCV contour detection to find individual artwork regions within a
    full-page scan image. Returns a list of dicts with the same shape as
    extract_pdf_images() entries: {image, bbox, width, height, page}.
    The 'page' field must be filled in by the caller.
    """
    img_w, img_h = page_pil.size
    total_area = img_w * img_h

    # Convert PIL → OpenCV (BGR)
    cv_img = cv2.cvtColor(np.array(page_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Pass 1: Adaptive thresholding (handles varying backgrounds)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    # Pass 2: Canny edge detection (catches sharp borders)
    canny = cv2.Canny(blurred, 30, 100)

    # Combine both passes
    combined = cv2.bitwise_or(adaptive, canny)

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding rectangles and filter
    raw_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / total_area

        # Filter by area ratio
        if area_ratio < MIN_SUBIMAGE_AREA_RATIO or area_ratio > MAX_SUBIMAGE_AREA_RATIO:
            continue
        # Filter by minimum dimension
        if w < MIN_SUBIMAGE_DIM or h < MIN_SUBIMAGE_DIM:
            continue
        # Filter extreme aspect ratios (>5:1)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 5.0:
            continue
        # Skip if it's basically the page border (within 5px of all edges)
        if x <= 5 and y <= 5 and (x + w) >= (img_w - 5) and (y + h) >= (img_h - 5):
            continue

        raw_rects.append((x, y, w, h))

    # Merge overlapping detections
    merged = _merge_overlapping_rects(raw_rects)

    # Convert pixel coords → PDF coords and crop sub-images
    scale = 72.0 / render_dpi  # pixel coords → PDF points
    if isinstance(page_rect, fitz.Rect):
        pdf_x0, pdf_y0 = page_rect.x0, page_rect.y0
    else:
        pdf_x0, pdf_y0 = 0.0, 0.0

    sub_images = []
    for (x, y, w, h) in merged:
        # Crop from original PIL image
        crop = page_pil.crop((x, y, x + w, y + h))

        # Convert pixel bbox to PDF coordinate bbox
        pdf_bbox = fitz.Rect(
            pdf_x0 + x * scale,
            pdf_y0 + y * scale,
            pdf_x0 + (x + w) * scale,
            pdf_y0 + (y + h) * scale,
        )

        sub_images.append({
            "image": crop,
            "bbox": pdf_bbox,
            "width": w,
            "height": h,
        })

    logger.info(f"Segmentation: found {len(sub_images)} sub-images from {len(contours)} contours")
    return sub_images


def _merge_overlapping_rects(rects: list[tuple]) -> list[tuple]:
    """Merge rectangles that overlap significantly (IoU >0.50 or containment >0.80)."""
    if not rects:
        return []

    # Sort by area descending
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    merged = []
    used = [False] * len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        x1, y1, w1, h1 = rects[i]
        # Accumulate overlapping rects into this one
        rx0, ry0, rx1, ry1 = x1, y1, x1 + w1, y1 + h1

        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
            x2, y2, w2, h2 = rects[j]
            bx0, by0, bx1, by1 = x2, y2, x2 + w2, y2 + h2

            # Compute intersection
            ix0 = max(rx0, bx0)
            iy0 = max(ry0, by0)
            ix1 = min(rx1, bx1)
            iy1 = min(ry1, by1)

            if ix0 >= ix1 or iy0 >= iy1:
                continue  # no intersection

            inter_area = (ix1 - ix0) * (iy1 - iy0)
            area_a = (rx1 - rx0) * (ry1 - ry0)
            area_b = (bx1 - bx0) * (by1 - by0)
            union_area = area_a + area_b - inter_area

            iou = inter_area / max(union_area, 1)
            containment = inter_area / max(min(area_a, area_b), 1)

            if iou > 0.50 or containment > 0.80:
                # Merge: expand the bounding box
                rx0 = min(rx0, bx0)
                ry0 = min(ry0, by0)
                rx1 = max(rx1, bx1)
                ry1 = max(ry1, by1)
                used[j] = True

        merged.append((rx0, ry0, rx1 - rx0, ry1 - ry0))

    return merged


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

            md5 = hashlib.sha256(img_bytes).hexdigest()
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

            # Check if this image covers the full page (scanned catalog)
            if is_full_page_image(bbox, page.rect):
                logger.info(f"Page {page_idx}: full-page image detected ({pil_img.width}x{pil_img.height}), segmenting...")
                sub_images = segment_page_image(pil_img, page.rect, RENDER_DPI)
                if sub_images:
                    for sub in sub_images:
                        sub_bytes = io.BytesIO()
                        sub["image"].save(sub_bytes, format="PNG")
                        results.append({
                            "page": page_idx,
                            "xref": xref,
                            "image": sub["image"],
                            "image_bytes": sub_bytes.getvalue(),
                            "bbox": sub["bbox"],
                            "width": sub["width"],
                            "height": sub["height"],
                            "segmented": True,
                        })
                else:
                    # Segmentation found nothing — keep original full-page image
                    logger.info(f"Page {page_idx}: segmentation found no sub-images, keeping full page")
                    results.append({
                        "page": page_idx,
                        "xref": xref,
                        "image": pil_img,
                        "image_bytes": img_bytes,
                        "bbox": bbox,
                        "width": pil_img.width,
                        "height": pil_img.height,
                        "segmented": False,
                    })
            else:
                results.append({
                    "page": page_idx,
                    "xref": xref,
                    "image": pil_img,
                    "image_bytes": img_bytes,
                    "bbox": bbox,
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "segmented": False,
                })

    # Fallback for scanned PDFs: render pages as images
    if not results:
        logger.info("No embedded images found — falling back to page renders")
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img_bytes = pix.tobytes("png")
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Try segmenting the rendered page
            sub_images = segment_page_image(pil_img, page.rect, RENDER_DPI)
            if sub_images:
                logger.info(f"Page {page_idx}: segmented into {len(sub_images)} sub-images")
                for sub in sub_images:
                    sub_bytes = io.BytesIO()
                    sub["image"].save(sub_bytes, format="PNG")
                    results.append({
                        "page": page_idx,
                        "xref": None,
                        "image": sub["image"],
                        "image_bytes": sub_bytes.getvalue(),
                        "bbox": sub["bbox"],
                        "width": sub["width"],
                        "height": sub["height"],
                        "segmented": True,
                    })
            else:
                # No sub-images found — keep full page as before
                results.append({
                    "page": page_idx,
                    "xref": None,
                    "image": pil_img,
                    "image_bytes": img_bytes,
                    "bbox": page.rect,
                    "width": pil_img.width,
                    "height": pil_img.height,
                    "segmented": False,
                })

    doc.close()

    # Mark images on dense pages (3+ images) as needing tight VLM zones
    from collections import Counter
    page_counts = Counter(r["page"] for r in results)
    for r in results:
        if page_counts[r["page"]] >= 3:
            r["segmented"] = True

    return results


def render_page_zone(pdf_bytes: bytes, page_idx: int, bbox, tight: bool = False) -> Image.Image:
    """Render the zone around an image bbox on a PDF page.

    Args:
        tight: Use reduced margins for segmented sub-images (avoids capturing
               neighbouring artworks on dense catalog pages).
    """
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

    if tight:
        # Tight margins for dense catalog pages: full row width, minimal vertical
        v_margin = max(bw, bh) * 0.3
        clip_rect = fitz.Rect(
            max(0, bbox.x0 - bw * 0.2),             # left: minimal
            bbox.y0 - v_margin,                       # above: tight
            page_rect.width * 0.95,                   # right: full row width
            bbox.y1 + v_margin,                       # below: tight
        )
    else:
        # Generous margins for single-artwork pages
        margin = max(bw, bh) * 0.5
        clip_rect = fitz.Rect(
            max(0, bbox.x0 - margin * 1),            # left: 1x
            bbox.y0 - margin * 2,                     # above: 2x
            bbox.x1 + margin * 4,                     # right: 4x
            bbox.y1 + margin * 3,                     # below: 3x
        )

    # If bbox is beyond the page rect, search other pages for where this image
    # is actually visible (some PDFs repeat the same images across pages at
    # different offsets — only one page has the image within viewable bounds).
    if bbox.y0 >= page_rect.y1 or bbox.y1 <= page_rect.y0 or bbox.x0 >= page_rect.x1:
        found = False
        for alt_idx in range(len(doc)):
            if alt_idx == page_idx:
                continue
            alt_page = doc[alt_idx]
            for img_info in alt_page.get_images(full=True):
                alt_bbox = alt_page.get_image_bbox(img_info)
                # Same xref or same dimensions at a valid position
                if (alt_bbox.width > 1 and alt_bbox.height > 1
                        and alt_bbox.y0 >= 0 and alt_bbox.y1 <= alt_page.rect.y1
                        and abs(alt_bbox.width - bbox.width) < 2
                        and abs(alt_bbox.height - bbox.height) < 2):
                    # Found the visible instance — switch to this page
                    page = alt_page
                    page_rect = alt_page.rect
                    page_idx = alt_idx
                    bbox = alt_bbox
                    bw = bbox.width
                    bh = bbox.height
                    # Recompute clip_rect with tight margins (multi-page catalogs are dense)
                    v_margin = max(bw, bh) * 0.3
                    clip_rect = fitz.Rect(
                        max(0, bbox.x0 - bw * 0.2),
                        bbox.y0 - v_margin,
                        page_rect.width * 0.95,
                        bbox.y1 + v_margin,
                    )
                    found = True
                    logger.debug(f"Off-page image relocated to page {alt_idx} bbox={bbox}")
                    break
            if found:
                break
        if not found:
            doc.close()
            return None

    # Clamp coordinates to page bounds
    clip_rect = fitz.Rect(
        max(page_rect.x0, clip_rect.x0),
        max(page_rect.y0, clip_rect.y0),
        min(page_rect.x1, clip_rect.x1),
        min(page_rect.y1, clip_rect.y1),
    )

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

    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        resp = await client.post(f"{VLM_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]
    return parse_vlm_json(content)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health")
async def health(x_api_key: str = Header(None)):
    """Health check — basic status for healthchecks, detailed info requires API key."""
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

    # Basic response for unauthenticated healthchecks (Docker, load balancers)
    response = {
        "clip": "ok" if clip_ok else "error",
        "vlm": "ok" if vlm_ok else "error",
        "qdrant": "ok" if qdrant_ok else "error",
    }

    # Extended info only with valid API key (or if no key configured)
    authenticated = (not API_KEY) or (x_api_key == API_KEY)
    if authenticated:
        response["qdrant_vectors"] = qdrant_vectors
        response["device"] = DEVICE
        if torch.cuda.is_available():
            response["vram"] = {
                "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
            }

    return response


@app.post("/match", dependencies=[Depends(verify_api_key)])
async def match(
    jpg: UploadFile = File(...),
    pdf: UploadFile = File(...),
    top_k: int = Form(3),
    threshold: float = Form(0.0),
):
    t_start = time.time()

    # Read and validate inputs
    jpg_bytes = await validate_upload_size(jpg, "image")
    pdf_bytes = await validate_upload_size(pdf, "PDF")
    ref_img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    ref_embedding = embed_image(ref_img)
    ref_phash_hex = compute_phash_hex(ref_img)
    ref_grayscale = is_grayscale(ref_img)
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
            pdf_grayscale = payload.get("is_grayscale", False)
            color_mismatch = ref_grayscale != pdf_grayscale
            combo = combined_score(clip_sim, phash_result["similarity"], color_mismatch=color_mismatch)

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
        color_mismatch = is_grayscale(ref_img) != is_grayscale(pdf_img)
        combo = combined_score(clip_sim, phash_result["similarity"], color_mismatch=color_mismatch)

        if combo >= threshold:
            matches.append({
                "clip_score": round(clip_sim, 4),
                "phash": phash_result,
                "combined_score": combo,
                "verdict": score_verdict(combo),
                "page": pdf_img_info["page"],
                "bbox": pdf_img_info["bbox"],
                "segmented": pdf_img_info.get("segmented", False),
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
            zone_img = render_page_zone(pdf_bytes, m["page"], m["bbox"], tight=m.get("segmented", False))
            if zone_img is None:
                zone_img = m.get("pdf_image")  # Fallback for off-page images
            if zone_img is not None:
                m["zone_b64"] = image_to_b64(zone_img)
                try:
                    m["description"] = await call_vlm(zone_img)
                except Exception as e:
                    logger.error(f"VLM call failed: {e}")
                    m["description"] = {"error": str(e)}
            else:
                m["zone_b64"] = None
                m["description"] = {"error": "no zone available"}
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


@app.post("/match-batch", dependencies=[Depends(verify_api_key)])
async def match_batch(
    pdf: UploadFile = File(...),
    images: list[UploadFile] = File(...),
    threshold: float = Form(0.70),
):
    t_start = time.time()
    pdf_bytes = await validate_upload_size(pdf, "PDF")
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
            ref_grayscale = is_grayscale(ref_img)

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
                pdf_grayscale = payload.get("is_grayscale", False)
                color_mismatch = ref_grayscale != pdf_grayscale
                combo = combined_score(clip_sim, phash_result["similarity"], color_mismatch=color_mismatch)
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
            color_mismatch = is_grayscale(ref_img) != is_grayscale(pdf_img_info["image"])
            combo = combined_score(clip_sim, phash_result["similarity"], color_mismatch=color_mismatch)

            if combo > best_score:
                best_score = combo
                best_match = {
                    "clip_score": round(clip_sim, 4),
                    "phash": phash_result,
                    "combined_score": combo,
                    "verdict": score_verdict(combo),
                    "page": pdf_img_info["page"],
                    "bbox": pdf_img_info["bbox"],
                    "segmented": pdf_img_info.get("segmented", False),
                    "size_vs_ref": {
                        "ref": f"{ref_img.width}x{ref_img.height}",
                        "pdf": f"{pdf_img_info['width']}x{pdf_img_info['height']}",
                    },
                }

        entry = {"filename": img_file.filename, "matched": False, "match": None}

        if best_match and best_match["combined_score"] >= threshold:
            entry["matched"] = True
            if best_match["bbox"] is not None:
                zone_img = render_page_zone(pdf_bytes, best_match["page"], best_match["bbox"], tight=best_match.get("segmented", False))
                if zone_img is None:
                    # Off-page image: find artwork for VLM fallback
                    for pi in pdf_images:
                        if pi["page"] == best_match["page"] and pi["bbox"] == best_match["bbox"]:
                            zone_img = pi["image"]
                            break
                if zone_img is not None:
                    best_match["zone_b64"] = image_to_b64(zone_img)
                    try:
                        best_match["description"] = await call_vlm(zone_img)
                    except Exception as e:
                        best_match["description"] = {"error": str(e)}
                else:
                    best_match["description"] = {"error": "no zone available"}
                    best_match["zone_b64"] = None
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


@app.post("/extract", dependencies=[Depends(verify_api_key)])
async def extract(pdf: UploadFile = File(...)):
    t_start = time.time()
    pdf_bytes = await validate_upload_size(pdf, "PDF")
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
            zone_img = render_page_zone(pdf_bytes, pdf_img_info["page"], pdf_img_info["bbox"], tight=pdf_img_info.get("segmented", False))
            if zone_img is None:
                zone_img = pdf_img_info["image"]  # Fallback for off-page images
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

@app.post("/index", dependencies=[Depends(verify_api_key)])
async def index_pdfs(pdfs: list[UploadFile] = File(...), force: bool = Form(False), tag: str = Form("")):
    """Index one or more PDFs into Qdrant for later searching."""
    t_start = time.time()
    indexed = []
    skipped = []

    for pdf_file in pdfs:
        pdf_bytes = await validate_upload_size(pdf_file, f"PDF '{pdf_file.filename}'")
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
            grayscale = is_grayscale(pil_img)

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
                    zone_img = render_page_zone(pdf_bytes, img_info["page"], bbox, tight=img_info.get("segmented", False))
                    if zone_img is None:
                        zone_img = pil_img  # Fallback: use artwork itself for off-page images
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
                    "is_grayscale": grayscale,
                    "tag": tag,
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


@app.post("/scan", dependencies=[Depends(verify_api_key)])
async def scan(
    image: UploadFile = File(...),
    top_k: int = Form(10),
    threshold: float = Form(0.70),
    tag: str = Form(""),
):
    """Search indexed PDFs for matches to the uploaded image."""
    t_start = time.time()

    # Embed query image
    img_bytes = await validate_upload_size(image, "image")
    ref_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    ref_embedding = embed_image(ref_img)
    ref_phash_hex = compute_phash_hex(ref_img)
    ref_grayscale = is_grayscale(ref_img)

    # Vector search in Qdrant
    t_search_start = time.time()
    query_filter = None
    if tag:
        query_filter = Filter(
            must=[FieldCondition(key="tag", match=MatchValue(value=tag))]
        )
    search_results = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=ref_embedding.tolist(),
        query_filter=query_filter,
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
        pdf_grayscale = payload.get("is_grayscale", False)
        color_mismatch = ref_grayscale != pdf_grayscale
        combo = combined_score(clip_sim, phash_result["similarity"], color_mismatch=color_mismatch)

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


@app.get("/index/status", dependencies=[Depends(verify_api_key)])
async def index_status(limit: int = 25, offset: int = 0):
    """Return status of indexed PDFs with pagination (default 25 per page).

    Uses filesystem (indexed_pdfs dir) for PDF list and Qdrant count() per PDF
    instead of scrolling all vectors (which timeouts on large collections).
    """
    collection = qdrant.get_collection(QDRANT_COLLECTION)
    total_vectors = collection.points_count

    # List PDFs from filesystem (instant, no Qdrant scroll needed)
    all_pdf_names = sorted(
        f for f in os.listdir(INDEXED_PDFS_DIR)
        if os.path.isfile(os.path.join(INDEXED_PDFS_DIR, f))
    )
    total_pdfs = len(all_pdf_names)
    page_names = all_pdf_names[offset:offset + limit]

    # Get image count per PDF using Qdrant count() with filter (fast, no scroll)
    pdfs = []
    for pdf_name in page_names:
        try:
            count_result = qdrant.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=Filter(
                    must=[FieldCondition(key="pdf_filename", match=MatchValue(value=pdf_name))]
                ),
                exact=False,
            )
            pdfs.append({"pdf": pdf_name, "images": count_result.count})
        except Exception:
            pdfs.append({"pdf": pdf_name, "images": "?"})

    return {
        "total_vectors": total_vectors,
        "total_pdfs": total_pdfs,
        "pdfs": pdfs,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_pdfs,
    }


@app.delete("/index/{pdf_filename}", dependencies=[Depends(verify_api_key)])
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


@app.delete("/index", dependencies=[Depends(verify_api_key)])
async def clear_index():
    """Clear the entire Qdrant index and stored PDFs."""
    # Recreate collection (fastest way to clear)
    qdrant.delete_collection(QDRANT_COLLECTION)
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
    )
    _ensure_payload_indexes()

    # Remove all stored PDFs
    for f in os.listdir(INDEXED_PDFS_DIR):
        fpath = os.path.join(INDEXED_PDFS_DIR, f)
        if os.path.isfile(fpath):
            os.remove(fpath)

    return {"status": "cleared", "total_vectors": 0}
