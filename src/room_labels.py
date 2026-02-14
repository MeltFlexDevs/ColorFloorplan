"""Extract room type labels from magenta pixels in the AI-generated floorplan.

The AI places magenta (255, 0, 255) capital letters in each room:
    L=Living room, B=Bedroom, K=Kitchen, T=Bathroom, H=Hallway,
    O=Office, D=Dining room, W=Closet/Storage, R=Laundry, A=Balcony

Pipeline:
    1. Load the magenta binary mask (saved by convert_to_svg)
    2. Morphological dilation to connect anti-aliased edges
    3. Find connected pixel blobs (each letter = one blob)
    4. Match each blob against pre-rendered letter templates (IoU)
    5. Return list of (letter, room_type, centroid_position, confidence)
"""

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from .config import OUTPUT_FOLDER

ROOM_TYPE_MAP = {
    "L": "livingRoom",
    "B": "bedroom",
    "K": "kitchen",
    "T": "bathroom",
    "H": "hallway",
    "O": "office",
    "D": "diningRoom",
    "W": "closet",
    "R": "laundry",
    "A": "balcony",
}

LETTERS = list(ROOM_TYPE_MAP.keys())
TEMPLATE_SIZE = 64


# ---------------------------------------------------------------------------
# Center-of-mass alignment
# ---------------------------------------------------------------------------

def _center_on_mass(binary: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    """Re-center a binary image so its center of mass is at (size//2, size//2)."""
    ys, xs = np.where(binary)
    if len(ys) == 0:
        return np.zeros((size, size), dtype=bool)
    cy = int(round(ys.mean()))
    cx = int(round(xs.mean()))
    shift_y = size // 2 - cy
    shift_x = size // 2 - cx
    result = np.zeros((size, size), dtype=bool)
    new_ys = ys + shift_y
    new_xs = xs + shift_x
    valid = (new_ys >= 0) & (new_ys < size) & (new_xs >= 0) & (new_xs < size)
    result[new_ys[valid], new_xs[valid]] = True
    return result


# ---------------------------------------------------------------------------
# Letter template generation (PIL)
# ---------------------------------------------------------------------------

def _render_letter(letter: str, font_size: int) -> np.ndarray:
    """Render a single letter at a given font size, centered on mass."""
    font = PIL.ImageFont.load_default(size=font_size)
    img = PIL.Image.new("L", (TEMPLATE_SIZE, TEMPLATE_SIZE), 255)
    draw = PIL.ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (TEMPLATE_SIZE - text_w) / 2 - bbox[0]
    y = (TEMPLATE_SIZE - text_h) / 2 - bbox[1]
    draw.text((x, y), letter, fill=0, font=font)
    return _center_on_mass(np.array(img) < 128)


def _create_letter_templates() -> dict[str, list[tuple[np.ndarray, int]]]:
    """Pre-render each letter at multiple font sizes for multi-scale matching.

    Returns dict mapping letter -> list of (binary_array, hole_count) tuples.
    """
    templates: dict[str, list[tuple[np.ndarray, int]]] = {}
    font_sizes = [36, 42, 48, 54]

    for letter in LETTERS:
        variants = []
        for fs in font_sizes:
            rendered = _render_letter(letter, fs)
            holes = _count_holes(rendered)
            variants.append((rendered, holes))
        templates[letter] = variants

    return templates


_templates_cache: dict[str, list[tuple[np.ndarray, int]]] | None = None


def _get_templates() -> dict[str, list[tuple[np.ndarray, int]]]:
    global _templates_cache
    if _templates_cache is None:
        _templates_cache = _create_letter_templates()
    return _templates_cache


# ---------------------------------------------------------------------------
# Morphological dilation (3x3, no scipy dependency)
# ---------------------------------------------------------------------------

def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    """3x3 binary dilation to connect anti-aliased / slightly separated pixels."""
    h, w = mask.shape
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            # Shift the mask by (dy, dx) and OR with output
            src_y = slice(max(0, -dy), min(h, h - dy))
            dst_y = slice(max(0, dy), min(h, h + dy))
            src_x = slice(max(0, -dx), min(w, w - dx))
            dst_x = slice(max(0, dx), min(w, w + dx))
            out[dst_y, dst_x] |= mask[src_y, src_x]
    return out


# ---------------------------------------------------------------------------
# Connected-component labelling (8-connected DFS)
# ---------------------------------------------------------------------------

def _find_blobs(
    mask: np.ndarray,
    min_area: int = 30,
    max_area_fraction: float = 0.08,
) -> tuple[np.ndarray, list[dict]]:
    """Find connected blobs in a binary mask using 8-connected DFS.

    Returns:
        labeled: int32 array, same shape as mask, with blob IDs (0 = background).
        blobs:   list of dicts with id, area, bbox, centroid.
    """
    h, w = mask.shape
    max_area = int(h * w * max_area_fraction)
    labeled = np.zeros((h, w), dtype=np.int32)
    blobs: list[dict] = []
    label_id = 0

    for start_y in range(h):
        for start_x in range(w):
            if not mask[start_y, start_x] or labeled[start_y, start_x] != 0:
                continue

            label_id += 1
            stack = [(start_y, start_x)]
            labeled[start_y, start_x] = label_id
            area = 0
            min_y = max_y = start_y
            min_x = max_x = start_x
            sum_y = 0
            sum_x = 0

            while stack:
                y, x = stack.pop()
                area += 1
                sum_y += y
                sum_x += x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = label_id
                            stack.append((ny, nx))

            if area < min_area or area > max_area:
                labeled[labeled == label_id] = 0
                label_id -= 1
                continue

            blobs.append(
                {
                    "id": label_id,
                    "area": area,
                    "bbox": (min_y, max_y, min_x, max_x),
                    "centroid": (sum_x / area, sum_y / area),
                }
            )

    return labeled, blobs


# ---------------------------------------------------------------------------
# Hu moments (scale/position/rotation invariant shape descriptor)
# ---------------------------------------------------------------------------

def _hu_moments(binary: np.ndarray) -> np.ndarray:
    """Compute the first 4 Hu moment invariants for a binary image."""
    h, w = binary.shape
    if binary.sum() == 0:
        return np.zeros(4)

    Y, X = np.mgrid[0:h, 0:w]
    mask = binary.astype(np.float64)
    total = mask.sum()

    # Centroid
    cx = (X * mask).sum() / total
    cy = (Y * mask).sum() / total

    # Central moments
    Xc = X - cx
    Yc = Y - cy

    def mu(p: int, q: int) -> float:
        return float((Xc**p * Yc**q * mask).sum())

    # Normalized central moments
    m00 = total
    def eta(p: int, q: int) -> float:
        gamma = (p + q) / 2.0 + 1.0
        return mu(p, q) / (m00**gamma) if m00 > 0 else 0.0

    e20 = eta(2, 0); e02 = eta(0, 2); e11 = eta(1, 1)
    e30 = eta(3, 0); e03 = eta(0, 3); e21 = eta(2, 1); e12 = eta(1, 2)

    h1 = e20 + e02
    h2 = (e20 - e02)**2 + 4 * e11**2
    h3 = (e30 - 3*e12)**2 + (3*e21 - e03)**2
    h4 = (e30 + e12)**2 + (e21 + e03)**2
    h5 = ((e30 - 3*e12) * (e30 + e12) * ((e30 + e12)**2 - 3*(e21 + e03)**2)
          + (3*e21 - e03) * (e21 + e03) * (3*(e30 + e12)**2 - (e21 + e03)**2))
    h6 = ((e20 - e02) * ((e30 + e12)**2 - (e21 + e03)**2)
          + 4 * e11 * (e30 + e12) * (e21 + e03))
    h7 = ((3*e21 - e03) * (e30 + e12) * ((e30 + e12)**2 - 3*(e21 + e03)**2)
          - (e30 - 3*e12) * (e21 + e03) * (3*(e30 + e12)**2 - (e21 + e03)**2))

    # Only return the first 4 moments – h5-h7 are extremely sensitive to
    # anti-aliasing and minor pixel-level noise, making them unreliable for
    # matching letters rendered at different sizes/fonts.
    return np.array([h1, h2, h3, h4])


def _log_hu(moments: np.ndarray) -> np.ndarray:
    """Log-transform Hu moments for better comparison scale."""
    return np.sign(moments) * np.log10(np.abs(moments) + 1e-20)


# ---------------------------------------------------------------------------
# Topological features (hole counting)
# ---------------------------------------------------------------------------

def _count_holes(binary: np.ndarray) -> int:
    """Count the number of enclosed holes in a binary shape.

    Pads with background, flood-fills from the outer background,
    then counts remaining background components (= holes).
    Distinguishes e.g. O(1 hole), B(2 holes), W(0 holes).
    """
    h, w = binary.shape
    # Pad with 1px border of background
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = binary

    # Flood fill outer background from (0,0)
    visited = np.zeros_like(padded, dtype=bool)
    ph, pw = padded.shape
    stack = [(0, 0)]
    visited[0, 0] = True

    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < ph and 0 <= nx < pw and not visited[ny, nx] and not padded[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

    # Remaining unvisited background pixels are holes
    hole_mask = (~padded) & (~visited)
    if not hole_mask.any():
        return 0

    # Count connected components of hole pixels
    hole_labeled = np.zeros_like(hole_mask, dtype=np.int32)
    n_holes = 0
    for sy in range(ph):
        for sx in range(pw):
            if hole_mask[sy, sx] and hole_labeled[sy, sx] == 0:
                n_holes += 1
                fill_stack = [(sy, sx)]
                hole_labeled[sy, sx] = n_holes
                while fill_stack:
                    fy, fx = fill_stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = fy + dy, fx + dx
                            if 0 <= ny < ph and 0 <= nx < pw and hole_mask[ny, nx] and hole_labeled[ny, nx] == 0:
                                hole_labeled[ny, nx] = n_holes
                                fill_stack.append((ny, nx))

    return n_holes


# ---------------------------------------------------------------------------
# Letter matching (Hu moments + IoU + topology)
# ---------------------------------------------------------------------------

def _match_letter(
    blob_crop: np.ndarray,
    templates: dict[str, list[tuple[np.ndarray, int]]],
) -> tuple[str, float]:
    """Match a cropped binary blob against all letter templates at multiple scales.

    Uses three complementary features:
      1. Topology (hole count) - hard filter on letter + per-template bonus
      2. Hu moment invariants - scale/position invariant shape descriptor
      3. Center-of-mass aligned IoU - pixel overlap confirmation
    For each letter, tries all template scales and keeps the best match.
    Returns (best_letter, combined_confidence).
    """
    h, w = blob_crop.shape
    if h < 3 or w < 3:
        return "?", 0.0

    # --- Topology: count holes in blob ---
    blob_holes = _count_holes(blob_crop)

    # Expected holes per letter.  Used as a HARD filter: candidates whose
    # hole count doesn't match the blob are excluded from consideration
    # (unless *no* candidate matches, in which case all compete).
    EXPECTED_HOLES: dict[str, set[int]] = {
        "L": {0},
        "B": {1, 2},       # 2 holes typical, but thick fonts can merge to 1
        "K": {0},
        "T": {0},
        "H": {0, 1},       # Can have 1 enclosed region or 0 at small sizes
        "O": {1},
        "D": {0, 1},       # D can have 0 or 1 hole depending on rendering
        "W": {0},
        "R": {0, 1},       # R can have 0 or 1 hole
        "A": {0, 1},       # A can have 0 or 1 hole depending on size
    }

    # Split candidates into topology-matching and non-matching
    topo_match = [l for l in templates if blob_holes in EXPECTED_HOLES.get(l, {0})]
    # Use topology-filtered set if it has candidates, otherwise fall back to all
    candidates = topo_match if topo_match else list(templates.keys())

    # --- Pre-compute blob descriptors (shared across all templates) ---
    blob_hu = _log_hu(_hu_moments(blob_crop))

    # Prepare resized blob for IoU (also shared)
    size = int(max(h, w) * 1.3)
    if size < 4:
        size = max(h, w) + 4
    padded_arr = np.zeros((size, size), dtype=np.uint8)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    padded_arr[pad_y : pad_y + h, pad_x : pad_x + w] = blob_crop.astype(np.uint8) * 255

    img = PIL.Image.fromarray(padded_arr, mode="L")
    img_resized = img.resize((TEMPLATE_SIZE, TEMPLATE_SIZE), PIL.Image.LANCZOS)
    blob_centered = _center_on_mass(np.array(img_resized) > 128)

    # --- Match only against topology-valid candidates, all scales ---
    best_score = -1.0
    best_letter = "?"

    for letter in candidates:
        template_variants = templates[letter]
        letter_best = -1.0

        for template, tmpl_holes in template_variants:
            # Hu moments score
            tmpl_hu = _log_hu(_hu_moments(template))
            dist = float(np.sum(np.abs(blob_hu - tmpl_hu)))
            hu_score = 1.0 / (1.0 + dist)

            # IoU score
            intersection = int(np.sum(blob_centered & template))
            union = int(np.sum(blob_centered | template))
            iou_score = intersection / union if union > 0 else 0.0

            # Topology bonus: prefer templates whose hole count matches blob
            topo_bonus = 0.10 if tmpl_holes == blob_holes else 0.0

            # Combined: Hu 60% + IoU 30% + topology match 10%
            combined = 0.60 * hu_score + 0.30 * iou_score + topo_bonus
            if combined > letter_best:
                letter_best = combined

        if letter_best > best_score:
            best_score = letter_best
            best_letter = letter

    return best_letter, best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_room_labels(name: str, min_confidence: float = 0.10) -> list[dict]:
    """Extract room type labels from the magenta mask saved by convert_to_svg.

    Args:
        name: Base name of the floorplan (matches the SVG/BMP naming).
        min_confidence: Minimum IoU to accept a match (0..1).

    Returns:
        List of dicts with keys:
            letter (str):      Best-matching letter (e.g. "L", "B").
            room_type (str):   Mapped room type (e.g. "livingRoom").
            position (tuple):  Centroid (x, y) in *pixel* coordinates.
            confidence (float): IoU score of the match.
    """
    mask_path = OUTPUT_FOLDER / f"{name}.labels.npy"

    if not mask_path.exists():
        print("[room_labels] No labels mask found – skipping room classification")
        return []

    try:
        mask: np.ndarray = np.load(mask_path)
    finally:
        mask_path.unlink(missing_ok=True)

    if not mask.any():
        print("[room_labels] No magenta pixels detected in image")
        return []

    # Dilate to fix anti-aliasing gaps within letters
    mask = _dilate_mask(mask)

    # Find connected blobs
    labeled, blobs = _find_blobs(mask)

    if not blobs:
        print("[room_labels] No letter blobs found (all too small or too large)")
        return []

    print(f"[room_labels] Found {len(blobs)} potential letter blob(s)")

    templates = _get_templates()
    results: list[dict] = []

    for blob in blobs:
        min_y, max_y, min_x, max_x = blob["bbox"]

        # Aspect ratio sanity (letters are not extremely wide or tall)
        blob_h = max_y - min_y + 1
        blob_w = max_x - min_x + 1
        aspect = blob_w / blob_h if blob_h > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            print(f"  Blob id={blob['id']} skipped: bad aspect ratio {aspect:.2f}")
            continue

        # Crop the blob from the labeled array
        crop = labeled[min_y : max_y + 1, min_x : max_x + 1] == blob["id"]

        letter, confidence = _match_letter(crop, templates)

        print(
            f"  Blob at ({blob['centroid'][0]:.0f}, {blob['centroid'][1]:.0f}): "
            f"'{letter}' (IoU: {confidence:.3f}, area: {blob['area']}px)"
        )

        if confidence >= min_confidence:
            results.append(
                {
                    "letter": letter,
                    "room_type": ROOM_TYPE_MAP.get(letter, "unknown"),
                    "position": blob["centroid"],
                    "confidence": confidence,
                }
            )

    print(f"[room_labels] Recognised {len(results)} room label(s)")
    return results
