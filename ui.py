"""
PikaMatch UI — Streamlit interface for image matching pipeline
Three tabs: single match, batch, PDF preview
"""

import base64
import io
import json
import os

import httpx
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8002")
API_KEY = os.getenv("API_KEY", "")
TIMEOUT = 120.0

st.set_page_config(page_title="PikaMatch", page_icon="⚡", layout="wide")
st.title("⚡ PikaMatch — Image Matching Pipeline")


def api_headers() -> dict:
    """Return headers with API key if configured."""
    if API_KEY:
        return {"X-API-Key": API_KEY}
    return {}


@st.cache_resource
def get_api_client() -> httpx.Client:
    """Persistent HTTP client with connection pooling (survives Streamlit reruns)."""
    transport = httpx.HTTPTransport(retries=2)
    return httpx.Client(
        base_url=API_URL,
        headers=api_headers(),
        timeout=60.0,
        transport=transport,
    )


def _reset_client():
    """Reset the HTTP client (e.g. after container restart)."""
    get_api_client.clear()


def api_get(path: str, **kwargs) -> httpx.Response:
    """GET request with auto-reconnect on stale connections."""
    try:
        return get_api_client().get(path, **kwargs)
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException):
        _reset_client()
        return get_api_client().get(path, **kwargs)


def api_post(path: str, **kwargs) -> httpx.Response:
    """POST request with auto-reconnect on stale connections."""
    try:
        return get_api_client().post(path, **kwargs)
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException):
        _reset_client()
        return get_api_client().post(path, **kwargs)


def api_delete(path: str, **kwargs) -> httpx.Response:
    """DELETE request with auto-reconnect on stale connections."""
    try:
        return get_api_client().delete(path, **kwargs)
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.TimeoutException):
        _reset_client()
        return get_api_client().delete(path, **kwargs)


# ============================================================
# Cached API calls (avoid blocking on every Streamlit rerun)
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_health() -> dict:
    """Fetch health status, cached for 30s. Errors are NOT cached (exceptions bypass cache)."""
    r = api_get("/health")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_index_status(limit: int, offset: int) -> dict:
    """Fetch index status with pagination, cached for 10s. Errors are NOT cached."""
    r = api_get("/index/status", params={"limit": limit, "offset": offset})
    r.raise_for_status()
    return r.json()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Status")
    try:
        health = fetch_health()
        clip_status = health.get("clip", "unknown")
        vlm_status = health.get("vlm", "unknown")
        qdrant_status = health.get("qdrant", "unknown")
        qdrant_vectors = health.get("qdrant_vectors")
        st.metric("CLIP", clip_status)
        st.metric("VLM (Qwen3-VL)", vlm_status)
        qdrant_label = f"{qdrant_status} ({qdrant_vectors} vectors)" if qdrant_vectors is not None else qdrant_status
        st.metric("Qdrant", qdrant_label)
        if health.get("vram"):
            vram = health["vram"]
            st.metric("VRAM", f"{vram.get('allocated_mb', '?')} / {vram.get('total_mb', '?')} MB")
    except Exception as e:
        st.warning(f"API: {type(e).__name__}")
        st.caption(f"{API_URL} — {e}")

    st.divider()
    st.header("Score Guide")
    st.markdown("""
| Score | Verdict |
|-------|---------|
| > 0.95 | Identical |
| > 0.90 | Very similar |
| > 0.80 | Similar |
| > 0.70 | Related |
| < 0.70 | Different |
""")

    st.divider()
    st.header("Pipeline")
    st.markdown("""
**Phase 1** — Triple-Signal (~50ms)
- CLIP color + CLIP gray + pHash
- Same: 55% / 30% / 15%
- Mismatch: 45% / 45% / 10%

**Phase 2** — Qwen3-VL (~3-5s)
- Lecture de la zone PDF autour de l'image
- Extraction des champs structurés
""")


# ============================================================
# Helpers
# ============================================================

def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def display_score(match_data: dict):
    """Display score breakdown for a match."""
    cols = st.columns(5)
    cols[0].metric("Combined", f"{match_data['combined_score']:.4f}")
    cols[1].metric("CLIP", f"{match_data['clip_score']:.4f}")
    cols[2].metric("CLIP Gray", f"{match_data.get('clip_gray_score', 0):.4f}")
    phash = match_data.get("phash", {})
    cols[3].metric("pHash", f"{phash.get('similarity', 0):.4f}")
    cols[4].metric("Verdict", match_data.get("verdict", "?"))


def display_description(desc: dict):
    """Display extracted description fields."""
    if "error" in desc:
        st.error(f"Extraction error: {desc['error']}")
        return
    if "raw_text" in desc:
        st.warning("VLM returned non-JSON response:")
        st.text(desc["raw_text"])
        return

    fields = [
        ("Description", "description"),
        ("Titre", "titre"),
        ("Taille", "taille"),
        ("Origine", "origine"),
        ("Référence", "reference"),
        ("Medium", "medium"),
        ("Couleur", "couleur"),
        ("Catégorie", "categorie"),
        ("Date", "date"),
        ("Auteur", "auteur"),
        ("Statut", "statut"),
        ("Version", "version"),
        ("Autres", "autres"),
    ]
    for label, key in fields:
        val = desc.get(key)
        if val is not None:
            st.markdown(f"**{label}:** {val}")


# ============================================================
# Tab 1: Single Match
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 JPG → PDF", "📦 Batch", "📄 Preview PDF", "🔎 Search Indexed PDFs", "📂 Scan PDF"])

with tab1:
    st.header("Single Match — JPG vs PDF")
    col_jpg, col_pdf = st.columns(2)

    with col_jpg:
        jpg_file = st.file_uploader("Upload JPG/PNG image", type=["jpg", "jpeg", "png", "webp", "bmp"], key="match_jpg")
    with col_pdf:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="match_pdf")

    col_opts = st.columns(2)
    top_k = col_opts[0].number_input("Top K matches", min_value=1, max_value=20, value=3, key="match_topk")
    threshold = col_opts[1].number_input("Score threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="match_threshold")

    if jpg_file and pdf_file:
        if st.button("🔍 Match", key="btn_match", type="primary"):
            with st.spinner("Matching en cours..."):
                files = {
                    "jpg": (jpg_file.name, jpg_file.getvalue(), jpg_file.type or "image/jpeg"),
                    "pdf": (pdf_file.name, pdf_file.getvalue(), "application/pdf"),
                }
                data = {"top_k": str(top_k), "threshold": str(threshold)}

                try:
                    r = api_post("/match", files=files, data=data, timeout=TIMEOUT)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # Timing
            timing = result.get("timing", {})
            tc = st.columns(3)
            tc[0].metric("CLIP matching", f"{timing.get('clip_matching_ms', '?')} ms")
            tc[1].metric("VLM extraction", f"{timing.get('vlm_extraction_ms', '?')} ms")
            tc[2].metric("Total", f"{timing.get('total_ms', '?')} ms")

            st.divider()

            # Reference info
            ref = result.get("reference", {})
            pdf_info = result.get("pdf", {})
            st.markdown(f"**Référence:** {ref.get('filename')} ({ref.get('size')}) | **PDF:** {pdf_info.get('filename')} ({pdf_info.get('images_found')} images trouvées)")

            # Best match
            best = result.get("best_match")
            if best:
                st.subheader("🏆 Best Match")
                display_score(best)

                size_info = best.get("size_vs_ref", {})
                st.markdown(f"**Page:** {best.get('page', '?')} | **Taille ref:** {size_info.get('ref')} → **PDF:** {size_info.get('pdf')} (scale: {size_info.get('scale_factor')})")

                st.subheader("📋 Description extraite")
                display_description(best.get("description", {}))

                if best.get("zone_b64"):
                    with st.expander("🔎 Zone envoyée au VLM"):
                        st.image(b64_to_image(best["zone_b64"]), caption="Zone PDF autour de l'image matchée")
            else:
                st.warning("Aucun match trouvé.")

            # All matches
            all_matches = result.get("all_matches", [])
            if len(all_matches) > 1:
                st.divider()
                st.subheader(f"Tous les matches ({len(all_matches)})")
                for i, m in enumerate(all_matches):
                    with st.expander(f"Match #{i+1} — Score: {m['combined_score']:.4f} ({m['verdict']})"):
                        display_score(m)
                        display_description(m.get("description", {}))
                        if m.get("zone_b64"):
                            st.image(b64_to_image(m["zone_b64"]), caption=f"Zone PDF — page {m.get('page', '?')}")


# ============================================================
# Tab 2: Batch
# ============================================================

with tab2:
    st.header("Batch Match — Multiple JPGs vs 1 PDF")

    pdf_batch = st.file_uploader("Upload PDF", type=["pdf"], key="batch_pdf")
    imgs_batch = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True, key="batch_imgs",
    )
    batch_threshold = st.number_input("Score threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.05, key="batch_threshold")

    if pdf_batch and imgs_batch:
        if st.button("📦 Batch Match", key="btn_batch", type="primary"):
            with st.spinner(f"Matching {len(imgs_batch)} images..."):
                files = [("pdf", (pdf_batch.name, pdf_batch.getvalue(), "application/pdf"))]
                for img in imgs_batch:
                    files.append(("images", (img.name, img.getvalue(), img.type or "image/jpeg")))
                data = {"threshold": str(batch_threshold)}

                try:
                    r = api_post("/match-batch", files=files, data=data, timeout=TIMEOUT * 2)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            st.metric("Total time", f"{result.get('timing', {}).get('total_ms', '?')} ms")
            st.markdown(f"**PDF:** {result.get('pdf', {}).get('filename')} ({result.get('pdf', {}).get('images_found')} images)")

            st.divider()

            for entry in result.get("results", []):
                if entry["matched"]:
                    icon = "✅"
                    match_info = entry["match"]
                    score_str = f"Score: {match_info['combined_score']:.4f}"
                else:
                    icon = "❌"
                    score_str = "No match"

                with st.expander(f"{icon} {entry['filename']} — {score_str}"):
                    if entry["matched"] and entry["match"]:
                        m = entry["match"]
                        display_score(m)
                        display_description(m.get("description", {}))
                        if m.get("zone_b64"):
                            st.image(b64_to_image(m["zone_b64"]), caption="Zone PDF")
                    else:
                        st.info("Aucun match au-dessus du seuil.")


# ============================================================
# Tab 3: Preview PDF
# ============================================================

with tab3:
    st.header("Preview PDF — Extract all images + descriptions")

    pdf_preview = st.file_uploader("Upload PDF", type=["pdf"], key="preview_pdf")

    if pdf_preview:
        if st.button("📄 Extract", key="btn_extract", type="primary"):
            with st.spinner("Extraction en cours..."):
                files = {"pdf": (pdf_preview.name, pdf_preview.getvalue(), "application/pdf")}

                try:
                    r = api_post("/extract", files=files, timeout=TIMEOUT * 3)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            st.metric("Total time", f"{result.get('timing', {}).get('total_ms', '?')} ms")
            pdf_info = result.get("pdf", {})
            st.markdown(f"**{pdf_info.get('filename')}** — {pdf_info.get('images_found')} images trouvées")

            st.divider()

            for i, ext in enumerate(result.get("extractions", [])):
                with st.expander(f"Image #{i+1} — Page {ext.get('page', '?')} ({ext.get('size', '?')})"):
                    cols = st.columns(2)

                    with cols[0]:
                        st.markdown("**Thumbnail**")
                        if ext.get("thumbnail_b64"):
                            st.image(b64_to_image(ext["thumbnail_b64"]))

                    with cols[1]:
                        st.markdown("**Zone PDF**")
                        if ext.get("zone_b64"):
                            st.image(b64_to_image(ext["zone_b64"]))

                    st.markdown("**Description extraite:**")
                    display_description(ext.get("description", {}))


# ============================================================
# Tab 4: Search Indexed PDFs
# ============================================================

with tab4:
    st.header("Search Indexed PDFs")

    # --- Section 1: Index Management ---
    st.subheader("📁 Index Management")

    # Upload PDFs to index
    pdfs_to_index = st.file_uploader(
        "Upload PDFs to index", type=["pdf"],
        accept_multiple_files=True, key="index_pdfs",
    )

    if pdfs_to_index:
        if st.button("📥 Index these PDFs", key="btn_index", type="primary"):
            with st.spinner(f"Indexing {len(pdfs_to_index)} PDF(s)..."):
                files = []
                for pdf in pdfs_to_index:
                    files.append(("pdfs", (pdf.name, pdf.getvalue(), "application/pdf")))

                try:
                    r = api_post("/index", files=files, timeout=TIMEOUT * 5)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            st.success(f"Indexed! Total vectors: {result.get('total_vectors', '?')} ({result.get('timing_ms', '?')}ms)")
            for item in result.get("indexed", []):
                st.markdown(f"- **{item['pdf']}**: {item['images']} images")
            # Invalidate index status cache after indexing
            fetch_index_status.clear()

    # Show currently indexed PDFs
    st.markdown("---")
    st.markdown("**Currently indexed PDFs:**")

    PAGE_SIZE = 25
    if "pdf_page" not in st.session_state:
        st.session_state.pdf_page = 0

    current_offset = st.session_state.pdf_page * PAGE_SIZE
    try:
        status = fetch_index_status(PAGE_SIZE, current_offset)
    except Exception as e:
        status = None
        st.warning(f"Could not fetch index status: {e}")

    if status is None:
        st.info("Loading... the API may be busy. Refresh in a few seconds.")
    elif status.get("total_vectors", 0) == 0:
        st.info("No PDFs indexed yet. Upload PDFs above to get started.")
    else:
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total vectors", status.get("total_vectors", 0))
        col_m2.metric("Total PDFs", status.get("total_pdfs", 0))

        for item in status.get("pdfs", []):
            col_name, col_count, col_del = st.columns([4, 1, 1])
            col_name.write(item["pdf"])
            col_count.write(f"{item['images']} imgs")
            if col_del.button("🗑️", key=f"del_{item['pdf']}"):
                try:
                    rd = api_delete(f"/index/{item['pdf']}")
                    rd.raise_for_status()
                    st.success(f"Deleted {item['pdf']}")
                    fetch_index_status.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete error: {e}")

        # Pagination controls
        total_pdfs = status.get("total_pdfs", 0)
        total_pages = max(1, (total_pdfs + PAGE_SIZE - 1) // PAGE_SIZE)
        if total_pages > 1:
            st.markdown("---")
            col_prev, col_info, col_next = st.columns([1, 2, 1])
            col_info.markdown(f"**Page {st.session_state.pdf_page + 1} / {total_pages}** ({total_pdfs} PDFs)")
            if col_prev.button("⬅️ Previous", key="btn_prev_page", disabled=(st.session_state.pdf_page == 0)):
                st.session_state.pdf_page -= 1
                st.rerun()
            if col_next.button("Next ➡️", key="btn_next_page", disabled=(not status.get("has_more", False))):
                st.session_state.pdf_page += 1
                st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear entire index", key="btn_clear_index"):
            try:
                rd = api_delete("/index")
                rd.raise_for_status()
                st.session_state.pdf_page = 0
                st.success("Index cleared!")
                fetch_index_status.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Clear error: {e}")

    # --- Section 2: Search ---
    st.divider()
    st.subheader("🔍 Search")

    scan_image = st.file_uploader(
        "Upload image to search", type=["jpg", "jpeg", "png", "webp", "bmp"], key="scan_img",
    )

    col_scan_opts = st.columns(3)
    scan_top_k = col_scan_opts[0].number_input("Top K results", min_value=1, max_value=100, value=10, key="scan_topk")
    scan_threshold = col_scan_opts[1].number_input("Score threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.05, key="scan_threshold")
    scan_tag = col_scan_opts[2].text_input("Tag filter", value="", key="scan_tag")

    if scan_image:
        if st.button("🔍 Search", key="btn_scan", type="primary"):
            with st.spinner("Searching indexed PDFs..."):
                files = {"image": (scan_image.name, scan_image.getvalue(), scan_image.type or "image/jpeg")}
                data = {"top_k": str(scan_top_k), "threshold": str(scan_threshold)}
                if scan_tag:
                    data["tag"] = scan_tag

                try:
                    r = api_post("/scan", files=files, data=data, timeout=TIMEOUT * 3)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # Timing
            timing = result.get("timing", {})
            tc = st.columns(3)
            tc[0].metric("Vector search", f"{timing.get('vector_search_ms', '?')} ms")
            tc[1].metric("VLM extraction", f"{timing.get('vlm_extraction_ms', '?')} ms")
            tc[2].metric("Total", f"{timing.get('total_ms', '?')} ms")

            st.divider()

            ref = result.get("reference", {})
            st.markdown(f"**Référence:** {ref.get('filename')} ({ref.get('size')}) | **Matches:** {result.get('total_matches', 0)}")

            results_list = result.get("results", [])
            if not results_list:
                st.warning("No matches found above threshold.")
            else:
                for i, m in enumerate(results_list):
                    label = f"{'🏆 ' if i == 0 else ''}Match #{i+1} — {m.get('pdf', '?')} (page {m.get('page', '?')}) — Score: {m['combined_score']:.4f} ({m['verdict']})"
                    with st.expander(label, expanded=(i == 0)):
                        display_score(m)

                        size_info = m.get("size_vs_ref", {})
                        st.markdown(f"**PDF:** {m.get('pdf')} | **Page:** {m.get('page', '?')} | **Taille ref:** {size_info.get('ref')} → **PDF:** {size_info.get('pdf')} (scale: {size_info.get('scale_factor')})")

                        st.markdown("**📋 Description extraite:**")
                        display_description(m.get("description", {}))

                        if m.get("zone_b64"):
                            st.image(b64_to_image(m["zone_b64"]), caption=f"Zone PDF — {m.get('pdf')} page {m.get('page', '?')}")


# ============================================================
# Tab 5: Scan PDF
# ============================================================

with tab5:
    st.header("Scan PDF — Search all images from a PDF against the index")

    pdf_scan = st.file_uploader("Upload PDF to scan", type=["pdf"], key="scan_pdf_file")

    col_sp_opts = st.columns(4)
    sp_top_k = col_sp_opts[0].number_input("Top K per image", min_value=1, max_value=100, value=5, key="sp_topk")
    sp_threshold = col_sp_opts[1].number_input("Score threshold", min_value=0.0, max_value=1.0, value=0.60, step=0.05, key="sp_threshold")
    sp_tag = col_sp_opts[2].text_input("Tag filter", value="", key="sp_tag")
    sp_exclude_self = col_sp_opts[3].checkbox("Skip source PDF", value=True, key="sp_exclude")

    if pdf_scan:
        if st.button("📂 Scan PDF", key="btn_scan_pdf", type="primary"):
            with st.spinner(f"Scanning {pdf_scan.name} against index..."):
                files = {"pdf": (pdf_scan.name, pdf_scan.getvalue(), "application/pdf")}
                data = {
                    "top_k": str(sp_top_k),
                    "threshold": str(sp_threshold),
                    "exclude_self": str(sp_exclude_self).lower(),
                }
                if sp_tag:
                    data["tag"] = sp_tag

                try:
                    r = api_post("/scan-pdf", files=files, data=data, timeout=TIMEOUT * 5)
                    r.raise_for_status()
                    result = r.json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # Timing
            timing = result.get("timing", {})
            tc = st.columns(3)
            tc[0].metric("Extraction", f"{timing.get('extraction_ms', '?')} ms")
            tc[1].metric("Search", f"{timing.get('search_ms', '?')} ms")
            tc[2].metric("Total", f"{timing.get('total_ms', '?')} ms")

            pdf_info = result.get("pdf", {})
            st.markdown(f"**{pdf_info.get('filename')}** — {pdf_info.get('images_extracted', 0)} images extracted")

            st.divider()

            results_list = result.get("results", [])
            if not results_list:
                st.warning("No images extracted from PDF.")
            else:
                for entry in results_list:
                    src = entry.get("source", {})
                    matches = entry.get("matches", [])
                    count = entry.get("total_matches", len(matches))
                    match_summary = f"{count} match(es)" if count > 0 else "no match"
                    label = f"Image #{src.get('index', 0) + 1} — Page {src.get('page', '?')} ({src.get('size', '?')}) — {match_summary}"

                    with st.expander(label, expanded=(count > 0)):
                        cols = st.columns([1, 3])
                        with cols[0]:
                            if src.get("thumbnail_b64"):
                                st.image(b64_to_image(src["thumbnail_b64"]), caption="Source")

                        with cols[1]:
                            if not matches:
                                st.info("No matches above threshold.")
                            else:
                                for i, m in enumerate(matches):
                                    if i > 0:
                                        st.markdown("---")
                                    icon = "🏆" if i == 0 else f"#{i+1}"
                                    st.markdown(f"**{icon} {m.get('pdf', '?')} p{m.get('page', '?')} — {m.get('combined_score', 0):.4f} ({m.get('verdict', '?')})**")
                                    display_score(m)
                                    display_description(m.get("description", {}))
                                    if m.get("thumbnail_b64"):
                                        st.image(b64_to_image(m["thumbnail_b64"]), caption=f"Match — {m.get('pdf')} page {m.get('page', '?')}")
