"""
Wall Segmentation Module

Rozdeľuje stenové polygóny na samostatné segmenty podľa zmien uhlu.
Podporuje:
- Rovné steny (horizontálne/vertikálne)
- Šikmé steny
- Oblé steny
- L-tvary, T-tvary, atď.
"""

from dataclasses import dataclass
from typing import Generator
import warnings
import numpy as np
from shapely import LineString, Point, MultiLineString, GeometryCollection
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import linemerge, nearest_points, split, voronoi_diagram
from shapely.affinity import scale
from shapely import union_all, buffer, make_valid

# Potlač Shapely warningy pre degenerate geometrie
warnings.filterwarnings('ignore', message='.*oriented_envelope.*')
warnings.filterwarnings('ignore', message='.*divide by zero.*')

# Konštanty pre segmentáciu
ANGLE_THRESHOLD_DEGREES = 20  # Minimálny uhol pre rozdelenie steny
CURVE_DETECTION_THRESHOLD = 5  # Stupne - ak je uhol menší, považuje sa za krivku
MIN_SEGMENT_LENGTH = 0.1  # Minimálna dĺžka segmentu v metroch


@dataclass
class WallSegment:
    """Reprezentuje jeden segment steny."""
    polygon: Polygon
    segment_type: str  # "straight", "angled", "curved"
    direction: np.ndarray | None  # Smerový vektor pre rovné/šikmé steny
    index: int  # Poradie segmentu


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Vypočíta uhol medzi dvoma vektormi v stupňoch.
    Vracia uhol v rozsahu 0-180.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)

    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def get_direction_type(direction: np.ndarray, tolerance: float = 5.0) -> str:
    """
    Určí typ smeru - horizontálny, vertikálny, alebo šikmý.
    """
    angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180

    if angle < tolerance or angle > 180 - tolerance:
        return "horizontal"
    elif abs(angle - 90) < tolerance:
        return "vertical"
    else:
        return "angled"


def extract_medial_axis(polygon: Polygon, num_samples: int = 100) -> LineString | MultiLineString | None:
    """
    Extrahuje strednú os (medial axis) polygónu steny.

    Používa iteratívny buffer prístup - zmenšuje polygón až kým sa nezredukuje na čiaru.
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    if polygon.is_empty or polygon.area < 1e-10:
        return None

    # Odhadni hrúbku steny
    min_rect = polygon.minimum_rotated_rectangle
    if isinstance(min_rect, Point):
        return None

    coords = list(min_rect.exterior.coords)
    side1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
    side2 = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
    wall_thickness = min(side1, side2)

    # Iteratívne zmenšuj polygón
    shrink_amount = wall_thickness * 0.4
    shrunk = polygon.buffer(-shrink_amount)

    if shrunk.is_empty:
        # Polygón je príliš tenký, použij centroid line
        return _create_centerline_from_rectangle(polygon)

    # Ak sa rozpadol na viac častí, máme komplexnejší tvar
    if isinstance(shrunk, MultiPolygon):
        lines = []
        for geom in shrunk.geoms:
            line = extract_medial_axis(geom)
            if line:
                if isinstance(line, MultiLineString):
                    lines.extend(line.geoms)
                else:
                    lines.append(line)
        return MultiLineString(lines) if lines else None

    # Pokračuj v zmenšovaní
    result = extract_medial_axis(shrunk, num_samples)
    if result:
        return result

    # Ak sme na konci, vytvor čiaru z polygónu
    return _create_centerline_from_rectangle(polygon)


def _create_centerline_from_rectangle(polygon: Polygon) -> LineString | None:
    """
    Vytvorí strednú čiaru z obdĺžnikového polygónu.
    """
    min_rect = polygon.minimum_rotated_rectangle
    if isinstance(min_rect, Point) or min_rect.is_empty:
        return None

    coords = np.array(min_rect.exterior.coords)[:-1]

    # Nájdi dlhšiu os
    side1 = np.linalg.norm(coords[0] - coords[1])
    side2 = np.linalg.norm(coords[1] - coords[2])

    if side1 > side2:
        # Dlhšia strana je 0-1 a 2-3
        mid1 = (coords[0] + coords[3]) / 2
        mid2 = (coords[1] + coords[2]) / 2
    else:
        # Dlhšia strana je 1-2 a 3-0
        mid1 = (coords[0] + coords[1]) / 2
        mid2 = (coords[2] + coords[3]) / 2

    return LineString([mid1, mid2])


def extract_skeleton_via_edges(polygon: Polygon) -> LineString | MultiLineString | None:
    """
    Alternatívna metóda extrakcie skeleton pomocou analýzy hrán.
    Robustnejšia pre typické stenové polygóny.
    """
    coords = np.array(polygon.exterior.coords)[:-1]
    n = len(coords)

    if n < 4:
        return None

    # Vypočítaj dĺžky a smery hrán
    edges = []
    for i in range(n):
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        length = np.linalg.norm(p2 - p1)
        direction = (p2 - p1) / (length + 1e-10)
        edges.append({
            'start': p1,
            'end': p2,
            'length': length,
            'direction': direction,
            'midpoint': (p1 + p2) / 2
        })

    # Nájdi protiľahlé hrany (podobný smer, opačná orientácia)
    centerline_points = []
    used_edges = set()

    for i, edge1 in enumerate(edges):
        if i in used_edges:
            continue

        for j, edge2 in enumerate(edges):
            if j in used_edges or j == i:
                continue

            # Skontroluj či sú hrany protiľahlé (paralelné, opačný smer)
            dot = np.dot(edge1['direction'], edge2['direction'])
            if abs(dot + 1) < 0.3 or abs(dot - 1) < 0.3:  # Paralelné (rovnaký alebo opačný smer)
                # Pridaj stredové body
                mid1 = edge1['midpoint']
                mid2 = edge2['midpoint']
                center = (mid1 + mid2) / 2
                centerline_points.append(center)
                used_edges.add(i)
                used_edges.add(j)
                break

    if len(centerline_points) < 2:
        return _create_centerline_from_rectangle(polygon)

    # Usporiadaj body a vytvor čiaru
    centerline_points = np.array(centerline_points)
    # Jednoduché usporiadanie podľa vzdialenosti
    ordered = [centerline_points[0]]
    remaining = list(range(1, len(centerline_points)))

    while remaining:
        last = ordered[-1]
        distances = [np.linalg.norm(centerline_points[i] - last) for i in remaining]
        nearest_idx = remaining[np.argmin(distances)]
        ordered.append(centerline_points[nearest_idx])
        remaining.remove(nearest_idx)

    return LineString(ordered)


def analyze_skeleton_angles(skeleton: LineString | MultiLineString) -> list[tuple[int, float, np.ndarray]]:
    """
    Analyzuje uhly pozdĺž skeleton a nájde body rozdelenia.

    Returns:
        List tuples (index bodu, uhol zmeny, pozícia)
    """
    if isinstance(skeleton, MultiLineString):
        # Spracuj každú čiaru samostatne
        results = []
        offset = 0
        for line in skeleton.geoms:
            line_results = analyze_skeleton_angles(line)
            results.extend([(idx + offset, angle, pos) for idx, angle, pos in line_results])
            offset += len(line.coords)
        return results

    coords = np.array(skeleton.coords)
    n = len(coords)

    if n < 3:
        return []

    break_points = []

    for i in range(1, n - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]

        angle = angle_between_vectors(v1, v2)

        # Ak je uhol výrazný, označ ako bod rozdelenia
        if angle > ANGLE_THRESHOLD_DEGREES:
            break_points.append((i, angle, coords[i]))

    return break_points


def segment_skeleton(skeleton: LineString, break_points: list[tuple[int, float, np.ndarray]]) -> list[LineString]:
    """
    Rozdelí skeleton na segmenty podľa break points.
    """
    if not break_points:
        return [skeleton]

    coords = list(skeleton.coords)
    segments = []

    # Pridaj začiatok
    break_indices = [0] + [bp[0] for bp in break_points] + [len(coords) - 1]

    for i in range(len(break_indices) - 1):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1]

        if end_idx - start_idx < 1:
            continue

        segment_coords = coords[start_idx:end_idx + 1]
        if len(segment_coords) >= 2:
            segments.append(LineString(segment_coords))

    return segments


def reconstruct_wall_segment(skeleton_segment: LineString, original_polygon: Polygon, wall_thickness: float) -> Polygon | None:
    """
    Rekonštruuje stenový segment z jeho skeleton.
    Používa buffer a priesečník s pôvodným polygónom.
    """
    # Vytvor buffer okolo skeleton
    buffered = skeleton_segment.buffer(wall_thickness * 0.6, cap_style='flat')

    # Orez na priesečník s pôvodnou stenou (rozšírenou o trochu)
    expanded_original = original_polygon.buffer(wall_thickness * 0.1)
    result = buffered.intersection(expanded_original)

    if result.is_empty:
        return None

    if isinstance(result, Polygon):
        return result
    elif isinstance(result, MultiPolygon):
        # Vráť najväčší polygón
        return max(result.geoms, key=lambda p: p.area)

    return None


def classify_segment(skeleton_segment: LineString, polygon: Polygon = None) -> str:
    """
    Klasifikuje segment ako rovný, šikmý alebo zakrivený.

    Args:
        skeleton_segment: LineString reprezentujúci skeleton segmentu
        polygon: Voliteľný polygón pre lepšiu klasifikáciu
    """
    coords = np.array(skeleton_segment.coords)

    if len(coords) < 2:
        return "straight"

    # Celkový smer segmentu
    overall_direction = coords[-1] - coords[0]
    overall_length = np.linalg.norm(overall_direction)

    if overall_length < 1e-10:
        return "straight"

    # Pre 2 body - jednoduchá klasifikácia
    if len(coords) == 2:
        dir_type = get_direction_type(overall_direction)
        if dir_type == "angled":
            return "angled"
        return "straight"

    # Pre viac bodov - analyzuj krivosť
    # Vypočítaj celkovú zmenu uhlu a priemernú zmenu
    angle_changes = []
    for i in range(1, len(coords) - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        angle = angle_between_vectors(v1, v2)
        angle_changes.append(angle)

    if not angle_changes:
        dir_type = get_direction_type(overall_direction)
        return "angled" if dir_type == "angled" else "straight"

    total_angle_change = sum(angle_changes)
    avg_angle_change = total_angle_change / len(angle_changes)

    # Ak má veľa malých zmien uhlu (konzistentná krivka)
    if len(angle_changes) >= 3 and avg_angle_change > 2 and avg_angle_change < 20:
        return "curved"

    # Ak má veľkú celkovú zmenu uhlu
    if total_angle_change > 45:
        return "curved"

    # Skontroluj či je polygón výrazne zakrivený pomocou pomeru obvod/dĺžka
    if polygon is not None:
        # Pomer dĺžky skeleton k obvodu polygónu
        perimeter = polygon.exterior.length
        if perimeter > 0 and overall_length > 0:
            # Pre rovnú stenu je pomer ~0.5 (skeleton je polovica obvodu)
            # Pre zakrivenú stenu je pomer menší
            ratio = overall_length / perimeter
            if ratio < 0.3:
                return "curved"

    # Skontroluj celkový smer
    dir_type = get_direction_type(overall_direction)

    if dir_type == "angled":
        return "angled"

    return "straight"


def get_segment_direction(skeleton_segment: LineString) -> np.ndarray | None:
    """
    Získa hlavný smer segmentu.
    """
    coords = np.array(skeleton_segment.coords)
    if len(coords) < 2:
        return None

    direction = coords[-1] - coords[0]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None

    return direction / norm


def segment_wall_polygon(polygon: Polygon, debug_output=None) -> list[WallSegment]:
    """
    Hlavná funkcia - rozdelí stenový polygón na segmenty.

    Args:
        polygon: Shapely Polygon reprezentujúci stenu
        debug_output: Voliteľný debug output objekt

    Returns:
        List WallSegment objektov
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    if polygon.is_empty or polygon.area < 1e-10:
        return []

    # Odhadni hrúbku steny
    min_rect = polygon.minimum_rotated_rectangle
    if isinstance(min_rect, Point) or min_rect.is_empty:
        return [WallSegment(polygon, "straight", None, 0)]

    coords = list(min_rect.exterior.coords)
    side1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[1]))
    side2 = np.linalg.norm(np.array(coords[1]) - np.array(coords[2]))
    wall_thickness = min(side1, side2)

    # Extrahuj skeleton
    skeleton = extract_skeleton_via_edges(polygon)

    if skeleton is None:
        return [WallSegment(polygon, "straight", None, 0)]

    # Analyzuj uhly
    if isinstance(skeleton, MultiLineString):
        # Spracuj každú časť samostatne
        all_segments = []
        segment_idx = 0
        for line in skeleton.geoms:
            break_points = analyze_skeleton_angles(line)
            skeleton_segments = segment_skeleton(line, break_points)

            for skel_seg in skeleton_segments:
                reconstructed = reconstruct_wall_segment(skel_seg, polygon, wall_thickness)
                if reconstructed and reconstructed.area > wall_thickness * wall_thickness * 0.1:
                    seg_type = classify_segment(skel_seg, reconstructed)
                    direction = get_segment_direction(skel_seg)
                    all_segments.append(WallSegment(reconstructed, seg_type, direction, segment_idx))
                    segment_idx += 1

        return all_segments if all_segments else [WallSegment(polygon, "straight", None, 0)]

    # Pre jednoduchý LineString
    break_points = analyze_skeleton_angles(skeleton)

    if debug_output:
        debug_output.print(skeleton, "purple", label="skeleton")
        for bp in break_points:
            debug_output.print(Point(bp[2]), "red", label=f"{bp[1]:.1f}°")

    # Ak nie sú body rozdelenia, vráť celú stenu
    if not break_points:
        seg_type = classify_segment(skeleton, polygon)
        direction = get_segment_direction(skeleton)
        return [WallSegment(polygon, seg_type, direction, 0)]

    # Rozdeľ skeleton
    skeleton_segments = segment_skeleton(skeleton, break_points)

    # Rekonštruuj segmenty
    wall_segments = []
    for idx, skel_seg in enumerate(skeleton_segments):
        reconstructed = reconstruct_wall_segment(skel_seg, polygon, wall_thickness)

        if reconstructed and reconstructed.area > wall_thickness * wall_thickness * 0.1:
            seg_type = classify_segment(skel_seg, reconstructed)
            direction = get_segment_direction(skel_seg)
            wall_segments.append(WallSegment(reconstructed, seg_type, direction, idx))

            if debug_output:
                debug_output.print(reconstructed, "green", label=f"seg_{idx}")

    # Ak rekonštrukcia zlyhala, vráť pôvodnú stenu
    if not wall_segments:
        return [WallSegment(polygon, "straight", None, 0)]

    return wall_segments


def segment_all_walls(wall_polygons: list[Polygon], debug_output=None) -> list[WallSegment]:
    """
    Rozdelí všetky stenové polygóny na segmenty.
    """
    all_segments = []
    global_idx = 0

    for polygon in wall_polygons:
        segments = segment_wall_polygon(polygon, debug_output)
        for seg in segments:
            seg.index = global_idx
            all_segments.append(seg)
            global_idx += 1

    return all_segments


# === ALTERNATÍVNY PRÍSTUP - Priama analýza hrán polygónu ===

def segment_wall_by_edge_analysis(polygon: Polygon, angle_threshold: float = ANGLE_THRESHOLD_DEGREES) -> list[WallSegment]:
    """
    Rozdelí stenu analýzou hrán vonkajšieho obrysu.
    Tento prístup je robustnejší pre typické steny.
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    coords = np.array(polygon.exterior.coords)[:-1]
    n = len(coords)

    if n < 4:
        return [WallSegment(polygon, "straight", None, 0)]

    # Vypočítaj uhol v každom bode
    angles_at_vertices = []
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        v1 = coords[i] - coords[prev_idx]
        v2 = coords[next_idx] - coords[i]

        angle = angle_between_vectors(v1, v2)
        angles_at_vertices.append(angle)

    # Nájdi rohové body (významná zmena uhlu)
    corner_indices = []
    for i, angle in enumerate(angles_at_vertices):
        if angle > angle_threshold:
            corner_indices.append(i)

    # Ak nie sú rohy, vráť celú stenu
    if len(corner_indices) < 2:
        return [WallSegment(polygon, "straight", None, 0)]

    # Rozdeľ polygón v rohových bodoch
    # Toto je komplexné - potrebujeme nájsť páry protiľahlých rohov

    # Pre zjednodušenie: ak máme 4 rohy a sú to pravé uhly, je to obdĺžnik
    if len(corner_indices) == 4 and all(abs(angles_at_vertices[i] - 90) < 15 for i in corner_indices):
        return [WallSegment(polygon, "straight", None, 0)]

    # Pre L-tvar a podobné - nájdi vnútorné rohy
    segments = _split_at_inner_corners(polygon, coords, corner_indices, angles_at_vertices)

    return segments if segments else [WallSegment(polygon, "straight", None, 0)]


def _split_at_inner_corners(polygon: Polygon, coords: np.ndarray, corner_indices: list[int], angles: list[float]) -> list[WallSegment]:
    """
    Rozdelí polygón v miestach vnútorných rohov (kde uhol > 180°, t.j. konkávne).
    """
    n = len(coords)

    # Nájdi vnútorné rohy (konkávne body)
    inner_corners = []
    for i in corner_indices:
        # Skontroluj orientáciu - je to vnútorný alebo vonkajší roh?
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        v1 = coords[i] - coords[prev_idx]
        v2 = coords[next_idx] - coords[i]

        # Cross product pre určenie orientácie
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        # Pre clockwise polygon, záporný cross = konkávny bod
        # Toto závisí od orientácie polygónu
        if polygon.exterior.is_ccw:
            is_inner = cross < 0
        else:
            is_inner = cross > 0

        if is_inner and angles[i] > 45:  # Významný vnútorný roh
            inner_corners.append(i)

    if not inner_corners:
        return []

    # Pre každý pár vnútorných rohov skús rozdeliť
    # Toto je zjednodušená verzia - pre plnú funkčnosť by bolo treba komplexnejší algoritmus

    segments = []
    segment_idx = 0

    # Ak máme presne 2 vnútorné rohy oproti sebe, môžeme rozdeliť
    if len(inner_corners) == 2:
        i1, i2 = inner_corners

        # Vytvor reznú čiaru
        cut_line = LineString([coords[i1], coords[i2]])

        # Rozdeľ polygón
        try:
            from shapely.ops import split
            result = split(polygon, cut_line)

            for geom in result.geoms:
                if isinstance(geom, Polygon) and geom.area > 0.01:
                    seg_type = "straight"  # Zjednodušené
                    segments.append(WallSegment(geom, seg_type, None, segment_idx))
                    segment_idx += 1
        except Exception:
            pass

    return segments


def advanced_wall_segmentation(polygon: Polygon, debug_output=None) -> list[WallSegment]:
    """
    Pokročilá segmentácia stien kombinujúca viacero prístupov.
    """
    # Prvý pokus - skeleton-based
    skeleton_segments = segment_wall_polygon(polygon, debug_output)

    if len(skeleton_segments) > 1:
        return skeleton_segments

    # Druhý pokus - edge analysis
    edge_segments = segment_wall_by_edge_analysis(polygon)

    if len(edge_segments) > 1:
        return edge_segments

    # Fallback - vráť pôvodnú stenu
    return [WallSegment(polygon, "straight", None, 0)]
