from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import time
import logging
import math
from pathlib import Path

from FileReaderv2.app.core.config import get_settings

logger = logging.getLogger(__name__)

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError("PaddleOCR not installed.")

    # For air-gapped environments: use pre-downloaded models from local directory.
    # __file__ = <project_root>/app/services/paddle_ocr.py
    # .parent       = app/services/
    # .parent.parent = app/
    # .parent.parent.parent = <project_root>/
    model_base = Path(__file__).resolve().parent.parent.parent / "models"
    det_dir = model_base / "en_PP-OCRv3_det_infer"
    rec_dir = model_base / "en_PP-OCRv3_rec_infer"
    cls_dir = model_base / "ch_ppocr_mobile_v2.0_cls_infer"

    local_kwargs = {}
    if det_dir.exists() and rec_dir.exists():
        logger.info("Using local models from %s", model_base)
        local_kwargs = {
            "det_model_dir": str(det_dir),
            "rec_model_dir": str(rec_dir),
            "cls_model_dir": str(cls_dir) if cls_dir.exists() else None,
        }
        local_kwargs = {k: v for k, v in local_kwargs.items() if v is not None}

    logger.info("Loading PP-OCRv5 model (CPU)...")
    _ocr_instance = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_angle_cls=True,
        use_textline_orientation=False,
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.8,
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
        enable_mkldnn=False,
        **local_kwargs,
    )
    logger.info("Model loaded.")
    return _ocr_instance


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    """Render every page of a PDF to a numpy array using pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError("pypdfium2 not installed. Run: pip install pypdfium2")

    import numpy as np

    doc = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72  # pypdfium2 default is 72 DPI
    images = []

    for page_num, page in enumerate(doc, start=1):
        bitmap = page.render(scale=scale, rotation=0)
        img_array = bitmap.to_numpy()

        # Drop alpha channel if present (RGBA -> RGB)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        images.append((page_num, img_array))
        logger.debug("Page %d rendered at %d DPI (%dx%d px)",
                     page_num, dpi, img_array.shape[1], img_array.shape[0])

    doc.close()
    return images

def _auto_deskew_image(img_array, max_skew_degrees: float = 10.0):
    """
    Automatically correct small page skew before OCR.

    This is intended for scanned PDFs where the page is tilted slightly left/right.
    It only rotates when the detected angle is realistic and safe.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not installed; skipping deskew.")
        return img_array

    if img_array is None:
        return img_array

    original = img_array

    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Invert threshold so text becomes white on black background.
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )[1]

        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            logger.debug("Deskew skipped: no foreground pixels found.")
            return original

        angle = cv2.minAreaRect(coords)[-1]

        # OpenCV returns angles in a slightly awkward range.
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if not math.isfinite(angle):
            logger.debug("Deskew skipped: invalid angle.")
            return original

        # Guardrails: only correct small scan skew.
        if abs(angle) < 0.3:
            logger.debug("Deskew skipped: angle %.2f is too small.", angle)
            return original

        if abs(angle) > max_skew_degrees:
            logger.warning(
                "Deskew skipped: detected angle %.2f exceeds safe limit %.2f.",
                angle,
                max_skew_degrees,
            )
            return original

        height, width = img_array.shape[:2]
        center = (width // 2, height // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        deskewed = cv2.warpAffine(
            img_array,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        logger.info("Deskew applied: %.2f degrees", angle)
        return deskewed

    except Exception as exc:
        logger.warning("Deskew failed; using original image: %s", exc)
        return original


def _extract_bbox_info(bbox_coords: list) -> tuple[float, float, float, float]:
    """Extract bounding box coordinates from PaddleOCR format.
    
    PaddleOCR returns bbox as: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    Returns: (min_x, min_y, max_x, max_y) for easier row/column grouping.
    """
    if not bbox_coords or len(bbox_coords) < 2:
        return 0, 0, 0, 0
    
    xs = [coord[0] for coord in bbox_coords]
    ys = [coord[1] for coord in bbox_coords]
    return min(xs), min(ys), max(xs), max(ys)


def _collect_text_elements(page_results: list) -> list[tuple[float, float, float, float, str]]:
    """Return (min_y, max_y, min_x, max_x, text) for every detected text box."""
    text_elements: list[tuple[float, float, float, float, str]] = []

    for line in page_results:
        if line is None or len(line) < 2:
            continue

        bbox = line[0]
        text_info = line[1]

        if bbox is None or text_info is None:
            continue

        min_x, min_y, max_x, max_y = _extract_bbox_info(bbox)
        text = str(text_info[0]).strip()

        if text:
            text_elements.append((min_y, max_y, min_x, max_x, text))

    return text_elements


def _is_same_row_legacy(
    min_y: float,
    max_y: float,
    row_min_y: float,
    row_max_y: float,
    y_threshold: float,
) -> bool:
    """Cells in one visual row overlap vertically even when their box heights differ."""
    overlap = min(max_y, row_max_y) - max(min_y, row_min_y)
    element_height = max(max_y - min_y, 1.0)
    row_height = max(row_max_y - row_min_y, 1.0)
    min_height = min(element_height, row_height)

    if overlap >= 0.35 * min_height:
        return True

    element_mid = (min_y + max_y) / 2
    row_mid = (row_min_y + row_max_y) / 2
    return abs(element_mid - row_mid) <= max(y_threshold, 0.6 * min_height)


def _group_text_by_rows_legacy(page_results: list, y_threshold: float = 15) -> list[list[tuple[float, float, str]]]:
    """Group detected text elements into rows using vertical bounding-box overlap.

    Args:
        page_results: PaddleOCR results for one page
        y_threshold: Fallback max Y-midpoint difference when box heights are unusable

    Returns:
        List of rows, where each row is a list of (min_x, max_x, text) tuples sorted left-to-right.
    """
    text_elements = _collect_text_elements(page_results)
    if not text_elements:
        return []

    # Sort by top edge so rows are built top-to-bottom.
    text_elements.sort(key=lambda t: (t[0], t[2]))

    rows: list[list[tuple[float, float, str]]] = []
    current_row: list[tuple[float, float, str]] = []
    row_min_y = 0.0
    row_max_y = 0.0

    for min_y, max_y, min_x, max_x, text in text_elements:
        if current_row and _is_same_row_legacy(min_y, max_y, row_min_y, row_max_y, y_threshold):
            current_row.append((min_x, max_x, text))
            row_min_y = min(row_min_y, min_y)
            row_max_y = max(row_max_y, max_y)
            continue

        if current_row:
            rows.append(current_row)
        current_row = [(min_x, max_x, text)]
        row_min_y, row_max_y = min_y, max_y

    if current_row:
        rows.append(current_row)

    # Sort each row left-to-right by X coordinate
    for row in rows:
        row.sort(key=lambda t: t[0])

    return rows


def _row_band(row: list[tuple[float, float, float, float, str]]) -> tuple[float, float]:
    """Median centre +/- half median height, so the band never grows as elements join."""
    centres = sorted((e[0] + e[1]) / 2 for e in row)
    heights = sorted(e[1] - e[0] for e in row)
    centre = centres[len(centres) // 2]
    half = max(heights[len(heights) // 2], 1.0) / 2
    return centre - half, centre + half


def _is_same_row_median(min_y: float, max_y: float, band_min: float, band_max: float) -> bool:
    centre = (min_y + max_y) / 2
    band_centre = (band_min + band_max) / 2
    # Second condition keeps tall/spanning cells with their row.
    return band_min <= centre <= band_max or min_y <= band_centre <= max_y


def _group_text_by_rows_median(page_results: list) -> list[list[tuple[float, float, str]]]:
    """Group text into rows by centre containment in a fixed-height band.

    Unlike the legacy grouping, the band does not expand to the union of its
    members, so padded boxes on consecutive lines cannot chain into one row.
    """
    text_elements = _collect_text_elements(page_results)
    if not text_elements:
        return []

    text_elements.sort(key=lambda t: ((t[0] + t[1]) / 2, t[2]))

    rows: list[list[tuple[float, float, float, float, str]]] = []
    current: list[tuple[float, float, float, float, str]] = []

    for element in text_elements:
        if current and _is_same_row_median(element[0], element[1], *_row_band(current)):
            current.append(element)
            continue
        if current:
            rows.append(current)
        current = [element]
    if current:
        rows.append(current)

    return [
        sorted(((e[2], e[3], e[4]) for e in row), key=lambda c: c[0])
        for row in rows
    ]


def _group_text_by_rows(page_results: list) -> list[list[tuple[float, float, str]]]:
    if get_settings().ocr_row_grouping == "median":
        return _group_text_by_rows_median(page_results)
    return _group_text_by_rows_legacy(page_results)


def _column_gap_threshold(cells: list[tuple[float, float, str]]) -> float:
    """Roughly 2.5 characters wide: column padding exceeds this, normal word spacing does not."""
    total_width = sum(max_x - min_x for min_x, max_x, _ in cells)
    total_chars = sum(len(text) for _, _, text in cells)
    if total_width <= 0 or total_chars <= 0:
        return float("inf")
    return 2.5 * (total_width / total_chars)


def _has_column_gap(cells: list[tuple[float, float, str]], threshold: float) -> bool:
    return any(
        cells[i + 1][0] - cells[i][1] >= threshold
        for i in range(len(cells) - 1)
    )


def _leading_label_size(texts: list[str]) -> int:
    """Number of leading cells that form a "Label :" caption, else 0."""
    if len(texts) >= 3 and texts[1] == ":":
        return 2
    if texts[0].endswith(":"):
        return 1
    return 0


def _format_markdown_row(row: list[tuple[float, float, str]]) -> str:
    cells = [
        (min_x, max_x, text.replace("|", "\\|").replace("\n", " ").strip())
        for min_x, max_x, text in row
    ]
    cells = [cell for cell in cells if cell[2]]
    if not cells:
        return ""
    if len(cells) == 1:
        return cells[0][2]

    label_size = _leading_label_size([cell[2] for cell in cells])
    data_cells = cells[label_size:]

    # Split off the caption only when the remaining cells are separated like real columns.
    if label_size and len(data_cells) >= 2 and _has_column_gap(data_cells, _column_gap_threshold(cells)):
        label = " ".join(cell[2] for cell in cells[:label_size])
        return f"{label} | {' | '.join(cell[2] for cell in data_cells)} |"

    return f"| {' | '.join(cell[2] for cell in cells)} |"


def _format_page_as_markdown(results: list, page_num: int) -> str:
    page_lines: list[str] = []
    for page_result in results:
        if page_result is None:
            continue
        for row in _group_text_by_rows(page_result):
            markdown_row = _format_markdown_row(row)
            if markdown_row:
                page_lines.append(markdown_row)

    if not page_lines:
        return ""
    return f"# Page {page_num}\n\n" + "\n".join(page_lines)


def process_pdf(pdf_path: str, dpi: int = 300) -> str:
    pdf_path = str(pdf_path)
    logger.info("Processing: %s at %d DPI", pdf_path, dpi)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    ocr = _get_ocr()

    markdown_pages: list[str] = []
    t0 = time.time()

    for page_num, img_array in images:
        logger.debug("OCR on page %d...", page_num)

        img_array =  _auto_deskew_image(img_array)

        try:
            results = ocr.ocr(img_array, cls=True)
        except Exception as exc:
            logger.warning("Page %d OCR error: %s", page_num, exc)
            continue

        if not results:
            logger.debug("Page %d: no text detected.", page_num)
            continue

        markdown_page = _format_page_as_markdown(results, page_num)
        if markdown_page:
            markdown_pages.append(markdown_page)

    elapsed = time.time() - t0
    line_count = sum(page.count("\n") + 1 for page in markdown_pages)
    logger.info("Done in %.1fs — %d Markdown lines extracted", elapsed, line_count)
    return "\n\n".join(markdown_pages)
