"""Extract room type labels from magenta pixels in the AI-generated floorplan.

The AI places magenta (255, 0, 255) capital letters in each room:
    L=Living room, B=Bedroom, K=Kitchen, T=Bathroom, H=Hallway,
    O=Office, D=Dining room, W=Closet/Storage, R=Laundry, A=Balcony

Pipeline:
    1. Load the magenta binary mask (saved by convert_to_svg)
    2. Morphological dilation to connect anti-aliased edges
    3. Find connected pixel blobs (each letter = one blob)
    4. Match each blob using zone-grid density + topology
    5. Return list of (letter, room_type, centroid_position, confidence)
"""

import numpy as np

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


# ---------------------------------------------------------------------------
# Zone-grid density features (font-independent structural matching)
# ---------------------------------------------------------------------------

def _compute_zone_grid(binary: np.ndarray, rows: int = 5, cols: int = 4) -> np.ndarray:
    """Compute fill ratios in a grid of zones over the binary image.

    The blob is padded to a square before grid computation so that
    the zone pattern is aspect-ratio independent.

    Returns a flat array of (rows * cols) fill ratios, each in [0, 1].
    """
    h, w = binary.shape
    size = max(h, w)
    padded = np.zeros((size, size), dtype=bool)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2
    padded[pad_y : pad_y + h, pad_x : pad_x + w] = binary

    grid = np.zeros(rows * cols, dtype=np.float64)
    for r in range(rows):
        y0 = r * size // rows
        y1 = (r + 1) * size // rows
        for c in range(cols):
            x0 = c * size // cols
            x1 = (c + 1) * size // cols
            zone = padded[y0:y1, x0:x1]
            if zone.size > 0:
                grid[r * cols + c] = float(zone.sum()) / zone.size
    return grid


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0..1 for non-negative inputs)."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def _horizontal_symmetry(binary: np.ndarray) -> float:
    """How symmetric the blob is about its vertical center axis (0..1)."""
    h, w = binary.shape
    if w < 2:
        return 1.0
    flipped = binary[:, ::-1]
    return float((binary == flipped).sum()) / max(1, binary.size)


def _vertical_symmetry(binary: np.ndarray) -> float:
    """How symmetric the blob is about its horizontal center axis (0..1)."""
    h, w = binary.shape
    if h < 2:
        return 1.0
    flipped = binary[::-1, :]
    return float((binary == flipped).sum()) / max(1, binary.size)


# Reference zone grids (5 rows x 4 cols = 20 values) per letter.
# Each array captures the universal structural ink distribution of a
# capital letter, independent of specific font face or rendering.
# Multiple variants per letter handle common font-style differences.
ZONE_REFS: dict[str, list[np.ndarray]] = {
    # L: left vertical stroke + bottom horizontal bar
    "L": [np.array([
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.5, 0.5, 0.3,
    ]), np.array([
        0.8, 0.1, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.7, 0.0, 0.0, 0.0,
        0.8, 0.6, 0.6, 0.4,
    ])],
    # B: left bar + two bumps right (typically 2 holes)
    "B": [np.array([
        0.8, 0.6, 0.5, 0.4,
        0.7, 0.1, 0.2, 0.5,
        0.8, 0.5, 0.5, 0.3,
        0.7, 0.1, 0.2, 0.5,
        0.8, 0.6, 0.5, 0.4,
    ]), np.array([
        0.8, 0.7, 0.6, 0.3,
        0.7, 0.0, 0.1, 0.6,
        0.8, 0.6, 0.5, 0.3,
        0.7, 0.0, 0.1, 0.6,
        0.8, 0.7, 0.6, 0.3,
    ])],
    # K: left bar + diagonal strokes meeting at mid-left
    "K": [np.array([
        0.7, 0.1, 0.3, 0.5,
        0.7, 0.3, 0.5, 0.2,
        0.7, 0.5, 0.2, 0.0,
        0.7, 0.3, 0.5, 0.2,
        0.7, 0.1, 0.3, 0.5,
    ]), np.array([
        0.7, 0.0, 0.4, 0.6,
        0.7, 0.2, 0.6, 0.1,
        0.7, 0.6, 0.1, 0.0,
        0.7, 0.2, 0.6, 0.1,
        0.7, 0.0, 0.4, 0.6,
    ])],
    # T: full top bar + center vertical stem
    "T": [np.array([
        0.6, 0.7, 0.7, 0.6,
        0.0, 0.5, 0.5, 0.0,
        0.0, 0.5, 0.5, 0.0,
        0.0, 0.5, 0.5, 0.0,
        0.0, 0.5, 0.5, 0.0,
    ]), np.array([
        0.7, 0.8, 0.8, 0.7,
        0.1, 0.6, 0.6, 0.1,
        0.0, 0.6, 0.6, 0.0,
        0.0, 0.6, 0.6, 0.0,
        0.0, 0.6, 0.6, 0.0,
    ])],
    # H: two vertical bars + middle crossbar
    "H": [np.array([
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.5, 0.5, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.0, 0.0, 0.7,
    ]), np.array([
        0.7, 0.1, 0.1, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.6, 0.6, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.1, 0.1, 0.7,
    ])],
    # O: oval ring with center hole
    "O": [np.array([
        0.3, 0.6, 0.6, 0.3,
        0.6, 0.1, 0.1, 0.6,
        0.7, 0.0, 0.0, 0.7,
        0.6, 0.1, 0.1, 0.6,
        0.3, 0.6, 0.6, 0.3,
    ]), np.array([
        0.2, 0.7, 0.7, 0.2,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.0, 0.0, 0.7,
        0.2, 0.7, 0.7, 0.2,
    ])],
    # D: left bar + right curve with hole
    "D": [np.array([
        0.8, 0.5, 0.4, 0.2,
        0.7, 0.1, 0.1, 0.5,
        0.7, 0.0, 0.0, 0.6,
        0.7, 0.1, 0.1, 0.5,
        0.8, 0.5, 0.4, 0.2,
    ]), np.array([
        0.8, 0.6, 0.3, 0.1,
        0.7, 0.0, 0.2, 0.5,
        0.7, 0.0, 0.1, 0.6,
        0.7, 0.0, 0.2, 0.5,
        0.8, 0.6, 0.3, 0.1,
    ])],
    # W: diagonal strokes - wide at top, converging at bottom center
    "W": [np.array([
        0.5, 0.1, 0.1, 0.5,
        0.4, 0.2, 0.2, 0.4,
        0.3, 0.4, 0.4, 0.3,
        0.2, 0.5, 0.5, 0.2,
        0.1, 0.6, 0.6, 0.1,
    ]), np.array([
        0.6, 0.0, 0.0, 0.6,
        0.5, 0.2, 0.2, 0.5,
        0.4, 0.4, 0.4, 0.4,
        0.2, 0.5, 0.5, 0.2,
        0.1, 0.7, 0.7, 0.1,
    ])],
    # R: left bar + top loop + diagonal leg to lower-right
    "R": [np.array([
        0.8, 0.6, 0.5, 0.3,
        0.7, 0.1, 0.2, 0.5,
        0.8, 0.5, 0.4, 0.2,
        0.7, 0.2, 0.4, 0.1,
        0.7, 0.0, 0.2, 0.5,
    ]), np.array([
        0.8, 0.7, 0.5, 0.2,
        0.7, 0.0, 0.1, 0.5,
        0.8, 0.6, 0.3, 0.1,
        0.7, 0.3, 0.3, 0.0,
        0.7, 0.1, 0.1, 0.5,
    ])],
    # A: triangle peak + crossbar + two legs at base
    "A": [np.array([
        0.0, 0.5, 0.5, 0.0,
        0.3, 0.3, 0.3, 0.3,
        0.6, 0.5, 0.5, 0.6,
        0.7, 0.1, 0.1, 0.7,
        0.7, 0.0, 0.0, 0.7,
    ]), np.array([
        0.1, 0.6, 0.6, 0.1,
        0.4, 0.3, 0.3, 0.4,
        0.6, 0.6, 0.6, 0.6,
        0.7, 0.0, 0.0, 0.7,
        0.7, 0.0, 0.0, 0.7,
    ])],
}

# Expected structural properties for secondary discrimination.
# h_sym = horizontally symmetric, left_heavy = more ink on left half
_LETTER_PROPS: dict[str, dict] = {
    "L": {"h_sym": False, "left_heavy": True,  "top_heavy": False},
    "B": {"h_sym": False, "left_heavy": True,  "top_heavy": None},
    "K": {"h_sym": False, "left_heavy": True,  "top_heavy": None},
    "T": {"h_sym": True,  "left_heavy": False, "top_heavy": True},
    "H": {"h_sym": True,  "left_heavy": False, "top_heavy": None},
    "O": {"h_sym": True,  "left_heavy": False, "top_heavy": None},
    "D": {"h_sym": False, "left_heavy": True,  "top_heavy": None},
    "W": {"h_sym": True,  "left_heavy": False, "top_heavy": False},
    "R": {"h_sym": False, "left_heavy": True,  "top_heavy": True},
    "A": {"h_sym": True,  "left_heavy": False, "top_heavy": True},
}


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
# Topological features (hole counting)
# ---------------------------------------------------------------------------

def _count_holes(binary: np.ndarray) -> int:
    """Count the number of enclosed holes in a binary shape.

    Pads with background, flood-fills from the outer background,
    then counts remaining background components (= holes).
    Distinguishes e.g. O(1 hole), B(2 holes), W(0 holes).
    """
    h, w = binary.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = binary

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

    hole_mask = (~padded) & (~visited)
    if not hole_mask.any():
        return 0

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
# Letter matching (zone-grid density + topology + structural properties)
# ---------------------------------------------------------------------------

# Expected holes per letter.  Used as a hard filter: candidates whose
# hole count doesn't match the blob are excluded from consideration.
EXPECTED_HOLES: dict[str, set[int]] = {
    "L": {0},
    "B": {1, 2},       # 2 holes typical, but thick fonts can merge to 1
    "K": {0},
    "T": {0},
    "H": {0, 1},       # Can have 1 enclosed region at small sizes
    "O": {1},
    "D": {0, 1},       # D can have 0 or 1 hole depending on rendering
    "W": {0},
    "R": {0, 1},       # R can have 0 or 1 hole
    "A": {0, 1},       # A can have 0 or 1 hole
}


def _structural_bonus(binary: np.ndarray, letter: str) -> float:
    """Score how well the blob's structural properties match the letter (0..0.15)."""
    props = _LETTER_PROPS.get(letter)
    if not props:
        return 0.0

    h, w = binary.shape
    bonus = 0.0
    checks = 0

    # Horizontal symmetry check
    h_sym = _horizontal_symmetry(binary)
    expected_sym = props.get("h_sym")
    if expected_sym is not None:
        checks += 1
        if expected_sym and h_sym > 0.80:
            bonus += 1.0
        elif not expected_sym and h_sym < 0.75:
            bonus += 1.0

    # Left-heavy check
    mid_x = w // 2
    if mid_x > 0:
        left_d = float(binary[:, :mid_x].sum()) / max(1, binary[:, :mid_x].size)
        right_d = float(binary[:, mid_x:].sum()) / max(1, binary[:, mid_x:].size)
        expected_left = props.get("left_heavy")
        if expected_left is not None:
            checks += 1
            if expected_left and left_d > right_d * 1.15:
                bonus += 1.0
            elif not expected_left and left_d <= right_d * 1.15:
                bonus += 1.0

    # Top-heavy check
    mid_y = h // 2
    if mid_y > 0:
        top_d = float(binary[:mid_y, :].sum()) / max(1, binary[:mid_y, :].size)
        bot_d = float(binary[mid_y:, :].sum()) / max(1, binary[mid_y:, :].size)
        expected_top = props.get("top_heavy")
        if expected_top is not None:
            checks += 1
            if expected_top and top_d > bot_d * 1.05:
                bonus += 1.0
            elif not expected_top and top_d <= bot_d * 1.05:
                bonus += 1.0

    # Normalize to 0..0.15 range
    if checks == 0:
        return 0.0
    return 0.15 * bonus / checks


def _match_letter(blob_crop: np.ndarray) -> tuple[str, float]:
    """Match a cropped binary blob against zone-grid reference patterns.

    Uses three complementary features:
      1. Topology (hole count) - hard filter on letter candidates
      2. Zone-grid density - font-independent structural comparison via
         cosine similarity between the blob's zone grid and each reference
      3. Structural properties - symmetry and ink distribution checks

    Returns (best_letter, confidence).
    """
    h, w = blob_crop.shape
    if h < 3 or w < 3:
        return "?", 0.0

    # --- Topology: count holes in blob ---
    blob_holes = _count_holes(blob_crop)

    # Filter by topology - if any candidates match, prefer them
    topo_match = [l for l in ZONE_REFS if blob_holes in EXPECTED_HOLES.get(l, {0})]
    candidates = topo_match if topo_match else list(ZONE_REFS.keys())

    # --- Zone-grid density comparison ---
    blob_grid = _compute_zone_grid(blob_crop)

    best_score = -1.0
    best_letter = "?"
    all_scores: list[tuple[str, float]] = []

    for letter in candidates:
        letter_best_sim = -1.0
        for ref in ZONE_REFS[letter]:
            sim = _cosine_sim(blob_grid, ref)
            if sim > letter_best_sim:
                letter_best_sim = sim
        # Structural bonus (0..0.15)
        s_bonus = _structural_bonus(blob_crop, letter)
        combined = letter_best_sim * 0.85 + s_bonus
        all_scores.append((letter, combined))
        if combined > best_score:
            best_score = combined
            best_letter = letter

    # Debug logging
    all_scores.sort(key=lambda x: -x[1])
    top3 = all_scores[:3]
    grid_str = ", ".join(f"{v:.2f}" for v in blob_grid)
    print(f"    [zone-grid] holes={blob_holes} grid=[{grid_str}]")
    print(f"    [zone-grid] top matches: {', '.join(f'{l}={s:.3f}' for l, s in top3)}")

    return best_letter, best_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_room_labels(name: str, min_confidence: float = 0.10) -> tuple[list[dict], tuple[int, int] | None]:
    """Extract room type labels from the magenta mask saved by convert_to_svg.

    Args:
        name: Base name of the floorplan (matches the SVG/BMP naming).
        min_confidence: Minimum score to accept a match (0..1).

    Returns:
        Tuple of (labels, image_shape) where:
            labels: list of dicts with keys letter, room_type, position, confidence.
            image_shape: (height, width) of the source image, or None if no mask found.
    """
    mask_path = OUTPUT_FOLDER / f"{name}.labels.npy"

    if not mask_path.exists():
        print("[room_labels] No labels mask found -- skipping room classification")
        return [], None

    try:
        mask: np.ndarray = np.load(mask_path)
    finally:
        mask_path.unlink(missing_ok=True)

    image_shape: tuple[int, int] = (mask.shape[0], mask.shape[1])

    if not mask.any():
        print("[room_labels] No magenta pixels detected in image")
        return [], image_shape

    # Dilate to fix anti-aliasing gaps within letters
    mask = _dilate_mask(mask)

    # Find connected blobs
    labeled, blobs = _find_blobs(mask)

    if not blobs:
        print("[room_labels] No letter blobs found (all too small or too large)")
        return [], image_shape

    print(f"[room_labels] Found {len(blobs)} potential letter blob(s)")

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

        letter, confidence = _match_letter(crop)

        print(
            f"  Blob at ({blob['centroid'][0]:.0f}, {blob['centroid'][1]:.0f}): "
            f"'{letter}' (score: {confidence:.3f}, area: {blob['area']}px)"
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
    return results, image_shape
