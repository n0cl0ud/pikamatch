"""
PikaMatch CLI — Test tool for the matching pipeline.

Usage:
    python test_clip.py health [FORMAT]
    python test_clip.py match photo.jpg catalogue.pdf [FORMAT]
    python test_clip.py extract catalogue.pdf [FORMAT]
    python test_clip.py batch catalogue.pdf img1.jpg img2.jpg [FORMAT]

Formats:
    (default)    Human-readable with score bars
    --json       Raw JSON
    --csv        CSV (semicolon-separated, for Excel)
    --markdown   Markdown table
    --minimal    One-line summary per match
"""

import csv
import io
import json
import os
import sys

import httpx

API_URL = "http://localhost:8002"
TIMEOUT = 120.0

DESC_FIELDS = [
    ("description", "Description"),
    ("auteur", "Auteur"),
    ("date", "Date"),
    ("taille", "Taille"),
    ("origine", "Origine"),
    ("reference", "Reference"),
    ("format", "Format"),
    ("couleur", "Couleur"),
    ("categorie", "Categorie"),
    ("statut", "Statut"),
    ("version", "Version"),
    ("autres", "Autres"),
]


def get_format(argv: list[str]) -> str:
    for flag in ["--json", "--csv", "--markdown", "--minimal"]:
        if flag in argv:
            return flag.lstrip("-")
    return "pretty"


# ============================================================
# Pretty output
# ============================================================

def score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {score:.4f}"


def verdict_icon(verdict: str) -> str:
    icons = {
        "identical": "=== IDENTICAL",
        "very_similar": "~~~ VERY SIMILAR",
        "similar": " ~  SIMILAR",
        "related": " ?  RELATED",
        "different": " X  DIFFERENT",
    }
    return icons.get(verdict, verdict)


def print_header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_desc(desc: dict):
    if not desc or "error" in desc:
        print(f"\n  Description: ERROR - {desc.get('error', '?')}")
        return
    if "raw_text" in desc:
        print(f"\n  Description (raw): {desc['raw_text'][:200]}")
        return
    print(f"\n  Description:")
    for key, label in DESC_FIELDS:
        val = desc.get(key)
        if val is not None:
            print(f"    {label:12s}: {val}")


def pretty_match(match: dict, index: int = 0):
    phash = match.get("phash", {})
    size = match.get("size_vs_ref", {})
    label = "BEST MATCH" if index == 0 else f"Match #{index + 1}"
    print(f"\n  --- {label} (page {match.get('page', '?')}) ---")
    print(f"  Verdict:  {verdict_icon(match.get('verdict', '?'))}")
    print(f"  Combined: {score_bar(match.get('combined_score', 0))}")
    print(f"  CLIP:     {score_bar(match.get('clip_score', 0))}")
    print(f"  pHash:    {score_bar(phash.get('similarity', 0))}  (distance: {phash.get('distance', 0)})")
    if size:
        print(f"  Size:     {size.get('ref', '?')} -> {size.get('pdf', '?')} (scale: {size.get('scale_factor', '?')})")
    print_desc(match.get("description", {}))


def pretty_match_result(result: dict):
    ref = result.get("reference", {})
    pdf = result.get("pdf", {})
    timing = result.get("timing", {})
    print_header("MATCH RESULT")
    print(f"  Image:    {ref.get('filename')} ({ref.get('size')})")
    print(f"  PDF:      {pdf.get('filename')} ({pdf.get('images_found')} images found)")
    print(f"  Timing:   CLIP {timing.get('clip_matching_ms', '?')}ms | VLM {timing.get('vlm_extraction_ms', '?')}ms | Total {timing.get('total_ms', '?')}ms")
    for i, m in enumerate(result.get("all_matches", [])):
        pretty_match(m, i)
    print()


def pretty_extract_result(result: dict):
    pdf = result.get("pdf", {})
    timing = result.get("timing", {})
    print_header("EXTRACT RESULT")
    print(f"  PDF:      {pdf.get('filename')} ({pdf.get('images_found')} images found)")
    print(f"  Timing:   {timing.get('total_ms', '?')}ms")
    for i, ext in enumerate(result.get("extractions", [])):
        print(f"\n  --- Image #{i + 1} (page {ext.get('page', '?')}, {ext.get('size', '?')}) ---")
        print_desc(ext.get("description", {}))
    print()


def pretty_batch_result(result: dict):
    pdf = result.get("pdf", {})
    timing = result.get("timing", {})
    print_header("BATCH RESULT")
    print(f"  PDF:      {pdf.get('filename')} ({pdf.get('images_found')} images found)")
    print(f"  Timing:   {timing.get('total_ms', '?')}ms")
    for entry in result.get("results", []):
        matched = entry.get("matched", False)
        icon = "[OK]" if matched else "[--]"
        name = entry.get("filename", "?")
        if matched and entry.get("match"):
            m = entry["match"]
            print(f"\n  {icon} {name} -> {m.get('combined_score', 0):.4f} ({m.get('verdict', '?')})")
            desc = m.get("description", {})
            if desc and "error" not in desc:
                for key in ["description", "auteur", "reference", "taille"]:
                    val = desc.get(key)
                    if val:
                        print(f"       {key}: {val}")
        else:
            print(f"\n  {icon} {name} -> no match")
    print()


# ============================================================
# CSV output
# ============================================================

def match_to_row(match: dict, ref_file: str = "", pdf_file: str = "") -> dict:
    """Flatten a match into a single dict row."""
    phash = match.get("phash", {})
    size = match.get("size_vs_ref", {})
    desc = match.get("description", {})
    if "error" in desc or "raw_text" in desc:
        desc = {}
    row = {
        "image": ref_file,
        "pdf": pdf_file,
        "page": match.get("page", ""),
        "combined_score": match.get("combined_score", ""),
        "clip_score": match.get("clip_score", ""),
        "phash_similarity": phash.get("similarity", ""),
        "phash_distance": phash.get("distance", ""),
        "verdict": match.get("verdict", ""),
        "ref_size": size.get("ref", ""),
        "pdf_size": size.get("pdf", ""),
        "scale_factor": size.get("scale_factor", ""),
    }
    for key, _ in DESC_FIELDS:
        row[key] = desc.get(key) or ""
    return row


CSV_COLUMNS = [
    "image", "pdf", "page", "combined_score", "clip_score",
    "phash_similarity", "phash_distance", "verdict",
    "ref_size", "pdf_size", "scale_factor",
] + [k for k, _ in DESC_FIELDS]


def csv_match_result(result: dict):
    ref = result.get("reference", {}).get("filename", "")
    pdf = result.get("pdf", {}).get("filename", "")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, delimiter=";")
    writer.writeheader()
    for m in result.get("all_matches", []):
        writer.writerow(match_to_row(m, ref, pdf))
    print(buf.getvalue().strip())


def csv_extract_result(result: dict):
    pdf = result.get("pdf", {}).get("filename", "")
    cols = ["pdf", "image_index", "page", "size"] + [k for k, _ in DESC_FIELDS]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, delimiter=";")
    writer.writeheader()
    for i, ext in enumerate(result.get("extractions", [])):
        desc = ext.get("description", {})
        if "error" in desc:
            desc = {}
        row = {"pdf": pdf, "image_index": i + 1, "page": ext.get("page", ""), "size": ext.get("size", "")}
        for key, _ in DESC_FIELDS:
            row[key] = desc.get(key) or ""
        writer.writerow(row)
    print(buf.getvalue().strip())


def csv_batch_result(result: dict):
    pdf = result.get("pdf", {}).get("filename", "")
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["matched"] + CSV_COLUMNS, delimiter=";")
    writer.writeheader()
    for entry in result.get("results", []):
        if entry.get("matched") and entry.get("match"):
            row = match_to_row(entry["match"], entry.get("filename", ""), pdf)
            row["matched"] = "yes"
        else:
            row = {c: "" for c in CSV_COLUMNS}
            row["image"] = entry.get("filename", "")
            row["pdf"] = pdf
            row["matched"] = "no"
        writer.writerow(row)
    print(buf.getvalue().strip())


# ============================================================
# Markdown output
# ============================================================

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def md_match_result(result: dict):
    ref = result.get("reference", {})
    pdf = result.get("pdf", {})
    timing = result.get("timing", {})
    print(f"## Match: {ref.get('filename')} vs {pdf.get('filename')}")
    print(f"*{pdf.get('images_found')} images found | CLIP {timing.get('clip_matching_ms', '?')}ms | VLM {timing.get('vlm_extraction_ms', '?')}ms | Total {timing.get('total_ms', '?')}ms*\n")

    # Scores table
    headers = ["#", "Page", "Combined", "CLIP", "pHash", "Distance", "Verdict"]
    rows = []
    for i, m in enumerate(result.get("all_matches", [])):
        ph = m.get("phash", {})
        rows.append([
            i + 1, m.get("page", "?"),
            f"{m.get('combined_score', 0):.4f}",
            f"{m.get('clip_score', 0):.4f}",
            f"{ph.get('similarity', 0):.4f}",
            ph.get("distance", "?"),
            m.get("verdict", "?"),
        ])
    print(md_table(headers, rows))

    # Description table for each match
    for i, m in enumerate(result.get("all_matches", [])):
        desc = m.get("description", {})
        if not desc or "error" in desc:
            continue
        print(f"\n### Match #{i + 1} — Description")
        headers = ["Field", "Value"]
        rows = []
        for key, label in DESC_FIELDS:
            val = desc.get(key)
            if val is not None:
                rows.append([label, val])
        if rows:
            print(md_table(headers, rows))
    print()


def md_extract_result(result: dict):
    pdf = result.get("pdf", {})
    print(f"## Extract: {pdf.get('filename')} ({pdf.get('images_found')} images)\n")
    for i, ext in enumerate(result.get("extractions", [])):
        desc = ext.get("description", {})
        if not desc or "error" in desc:
            continue
        print(f"### Image #{i + 1} (page {ext.get('page', '?')}, {ext.get('size', '?')})")
        headers = ["Field", "Value"]
        rows = [[label, desc.get(key, "")] for key, label in DESC_FIELDS if desc.get(key) is not None]
        if rows:
            print(md_table(headers, rows))
        print()


def md_batch_result(result: dict):
    pdf = result.get("pdf", {})
    print(f"## Batch: {pdf.get('filename')} ({pdf.get('images_found')} images)\n")
    headers = ["Image", "Matched", "Score", "Verdict", "Description"]
    rows = []
    for entry in result.get("results", []):
        name = entry.get("filename", "?")
        if entry.get("matched") and entry.get("match"):
            m = entry["match"]
            desc_text = m.get("description", {}).get("description", "") or ""
            rows.append([name, "yes", f"{m.get('combined_score', 0):.4f}", m.get("verdict", "?"), desc_text[:60]])
        else:
            rows.append([name, "no", "-", "-", "-"])
    print(md_table(headers, rows))
    print()


# ============================================================
# Minimal output
# ============================================================

def minimal_match_result(result: dict):
    ref = result.get("reference", {}).get("filename", "?")
    timing = result.get("timing", {})
    for i, m in enumerate(result.get("all_matches", [])):
        desc = m.get("description", {})
        desc_short = desc.get("description", "") or ""
        if len(desc_short) > 50:
            desc_short = desc_short[:50] + "..."
        tag = "*" if i == 0 else " "
        print(f"{tag} {ref} -> p{m.get('page', '?')} | {m.get('combined_score', 0):.4f} {m.get('verdict', '?'):14s} | {desc_short}")
    total = timing.get("total_ms", "?")
    print(f"  ({total}ms)")


def minimal_extract_result(result: dict):
    pdf = result.get("pdf", {}).get("filename", "?")
    for i, ext in enumerate(result.get("extractions", [])):
        desc = ext.get("description", {})
        desc_short = desc.get("description", "") or ""
        if len(desc_short) > 60:
            desc_short = desc_short[:60] + "..."
        print(f"  img{i + 1} p{ext.get('page', '?')} {ext.get('size', '?'):>10s} | {desc_short}")


def minimal_batch_result(result: dict):
    for entry in result.get("results", []):
        name = entry.get("filename", "?")
        if entry.get("matched") and entry.get("match"):
            m = entry["match"]
            print(f"  [OK] {name:20s} -> {m.get('combined_score', 0):.4f} ({m.get('verdict', '?')})")
        else:
            print(f"  [--] {name:20s} -> no match")


# ============================================================
# Output dispatcher
# ============================================================

def output(result: dict, fmt: str, kind: str):
    """Dispatch output based on format and result kind (match/extract/batch)."""
    if fmt == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif fmt == "csv":
        {"match": csv_match_result, "extract": csv_extract_result, "batch": csv_batch_result}[kind](result)
    elif fmt == "markdown":
        {"match": md_match_result, "extract": md_extract_result, "batch": md_batch_result}[kind](result)
    elif fmt == "minimal":
        {"match": minimal_match_result, "extract": minimal_extract_result, "batch": minimal_batch_result}[kind](result)
    else:
        {"match": pretty_match_result, "extract": pretty_extract_result, "batch": pretty_batch_result}[kind](result)


# ============================================================
# Commands
# ============================================================

def strip_b64(result: dict, kind: str):
    """Remove base64 data from results."""
    if kind == "match":
        for m in result.get("all_matches", []):
            m.pop("zone_b64", None)
        if result.get("best_match"):
            result["best_match"].pop("zone_b64", None)
    elif kind == "extract":
        for ext in result.get("extractions", []):
            ext.pop("zone_b64", None)
            ext.pop("thumbnail_b64", None)
    elif kind == "batch":
        for entry in result.get("results", []):
            if entry.get("match"):
                entry["match"].pop("zone_b64", None)


def cmd_health(fmt: str):
    r = httpx.get(f"{API_URL}/health", timeout=10.0)
    r.raise_for_status()
    data = r.json()
    if fmt == "json":
        print(json.dumps(data, indent=2))
    else:
        clip_s = data.get("clip", "?")
        vlm_s = data.get("vlm", "?")
        device = data.get("device", "?")
        vram = data.get("vram", {})
        print_header("HEALTH")
        print(f"  CLIP:   {clip_s}")
        print(f"  VLM:    {vlm_s}")
        print(f"  Device: {device}")
        if vram:
            print(f"  VRAM:   {vram.get('allocated_mb', 0):.0f} / {vram.get('total_mb', 0):.0f} MB")
        print()


def cmd_match(jpg_path: str, pdf_path: str, fmt: str):
    with open(jpg_path, "rb") as f_jpg, open(pdf_path, "rb") as f_pdf:
        files = {
            "jpg": (os.path.basename(jpg_path), f_jpg, "image/jpeg"),
            "pdf": (os.path.basename(pdf_path), f_pdf, "application/pdf"),
        }
        r = httpx.post(f"{API_URL}/match", files=files, timeout=TIMEOUT)
    r.raise_for_status()
    result = r.json()
    strip_b64(result, "match")
    output(result, fmt, "match")


def cmd_extract(pdf_path: str, fmt: str):
    with open(pdf_path, "rb") as f_pdf:
        files = {"pdf": (os.path.basename(pdf_path), f_pdf, "application/pdf")}
        r = httpx.post(f"{API_URL}/extract", files=files, timeout=TIMEOUT * 2)
    r.raise_for_status()
    result = r.json()
    strip_b64(result, "extract")
    output(result, fmt, "extract")


def cmd_batch(pdf_path: str, img_paths: list[str], fmt: str):
    with open(pdf_path, "rb") as f_pdf:
        files = [("pdf", (os.path.basename(pdf_path), f_pdf, "application/pdf"))]
        img_handles = []
        for p in img_paths:
            fh = open(p, "rb")
            img_handles.append(fh)
            files.append(("images", (os.path.basename(p), fh, "image/jpeg")))

        r = httpx.post(f"{API_URL}/match-batch", files=files, timeout=TIMEOUT * 3)

        for fh in img_handles:
            fh.close()

    r.raise_for_status()
    result = r.json()
    strip_b64(result, "batch")
    output(result, fmt, "batch")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fmt = get_format(sys.argv)

    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]

    if cmd == "health":
        cmd_health(fmt)
    elif cmd == "match":
        if len(args) != 3:
            print("Usage: python test_clip.py match <jpg> <pdf> [--json|--csv|--markdown|--minimal]")
            sys.exit(1)
        cmd_match(args[1], args[2], fmt)
    elif cmd == "extract":
        if len(args) != 2:
            print("Usage: python test_clip.py extract <pdf> [--json|--csv|--markdown|--minimal]")
            sys.exit(1)
        cmd_extract(args[1], fmt)
    elif cmd == "batch":
        if len(args) < 3:
            print("Usage: python test_clip.py batch <pdf> <img1> [img2] ... [--json|--csv|--markdown|--minimal]")
            sys.exit(1)
        cmd_batch(args[1], args[2:], fmt)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
