"""
Wall Segmentation Module

Rozdeľuje stenové polygóny na samostatné meshe podľa zmien uhlu hrán.
Používa pôvodnú geometriu polygónu - nepretvára ju.
"""

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon
from shapely import make_valid

# Konštanty
ANGLE_THRESHOLD_DEGREES = 15.0


@dataclass
class WallSegment:
    """Segment steny - skupina hrán s podobnou orientáciou."""
    edges: list[tuple[tuple, tuple]]  # Zoznam hrán [(p1, p2), ...]
    segment_type: str  # "straight", "angled", "curved"
    index: int


@dataclass
class WallEdgeGroup:
    """Skupina hrán pre jeden segment."""
    edges: list[dict]
    segment_type: str


def get_edge_angle(p1: tuple, p2: tuple) -> float:
    """Vypočíta uhol hrany v radiánoch."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)


def angle_difference(a1: float, a2: float) -> float:
    """Vypočíta absolútnu zmenu uhlu, normalizovanú pre 180° symetriu."""
    diff = abs(a1 - a2)
    diff = min(diff, 2 * np.pi - diff)
    if diff > np.pi / 2:
        diff = np.pi - diff
    return diff


def get_direction_type(angle: float, tolerance_deg: float = 5.0) -> str:
    """Určí typ smeru - straight alebo angled."""
    angle_deg = np.degrees(angle) % 180
    if angle_deg < tolerance_deg or angle_deg > 180 - tolerance_deg:
        return "straight"
    elif abs(angle_deg - 90) < tolerance_deg:
        return "straight"
    else:
        return "angled"


def classify_edge_group(edges: list[dict]) -> str:
    """Klasifikuje skupinu hrán."""
    if not edges:
        return "straight"

    if len(edges) == 1:
        return get_direction_type(edges[0]['angle'])

    # Pre viac hrán - skontroluj krivku
    total_angle_change = 0
    for i in range(1, len(edges)):
        diff = angle_difference(edges[i]['angle'], edges[i-1]['angle'])
        total_angle_change += diff

    avg_change = total_angle_change / (len(edges) - 1) if len(edges) > 1 else 0

    if len(edges) >= 3 and avg_change > np.radians(1) and avg_change < np.radians(15):
        return "curved"

    return get_direction_type(edges[0]['angle'])


def group_polygon_edges(polygon: Polygon, angle_threshold_degrees: float = ANGLE_THRESHOLD_DEGREES) -> list[WallEdgeGroup]:
    """
    Zoskupí hrany polygónu podľa uhlu.
    Vracia skupiny hrán, kde každá skupina bude jeden segment.
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    coords = list(polygon.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        return []

    threshold = np.radians(angle_threshold_degrees)

    # Vytvor hrany
    edges = []
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

        if length < 0.001:
            continue

        angle = get_edge_angle(p1, p2)
        edges.append({
            'start': p1,
            'end': p2,
            'angle': angle,
            'length': length
        })

    if not edges:
        return []

    # Zoskup hrany
    groups = []
    current_group = [edges[0]]

    for i in range(1, len(edges)):
        current = edges[i]
        prev = current_group[-1]
        diff = angle_difference(current['angle'], prev['angle'])

        if diff < threshold:
            current_group.append(current)
        else:
            groups.append(current_group)
            current_group = [current]

    groups.append(current_group)

    # Spoj prvú a poslednú skupinu ak majú podobný uhol
    if len(groups) > 1:
        first_angle = groups[0][0]['angle']
        last_angle = groups[-1][-1]['angle']
        if angle_difference(first_angle, last_angle) < threshold:
            groups[0] = groups[-1] + groups[0]
            groups.pop()

    # Vytvor WallEdgeGroup objekty
    result = []
    for group in groups:
        seg_type = classify_edge_group(group)
        result.append(WallEdgeGroup(edges=group, segment_type=seg_type))

    return result


def get_wall_thickness(polygon: Polygon) -> float:
    """Odhadne hrúbku steny."""
    area = polygon.area
    perimeter = polygon.exterior.length
    if perimeter == 0:
        return 0.2
    estimated = 2 * area / perimeter
    return max(0.05, min(0.5, estimated))


def get_inward_normal(p1: tuple, p2: tuple) -> np.ndarray:
    """Získa dovnútra smerujúci normálový vektor."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0:
        return np.array([0.0, 0.0])
    return np.array([dy / length, -dx / length])


def build_wall_segments_geometry(
    polygon: Polygon,
    builder,  # MeshBuilder
    height: float = 2.6,
    angle_threshold: float = ANGLE_THRESHOLD_DEGREES
):
    """
    Vytvorí 3D geometriu pre stenový polygón, rozdelenú na segmenty podľa uhlu hrán.

    Každý segment je kompletná 3D geometria s:
    - Vonkajšou vertikálnou stenou
    - Vnútornou vertikálnou stenou
    - Vrchnou plochou
    - Spodnou plochou
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    thickness = get_wall_thickness(polygon)

    # Zoskup hrany vonkajšieho obrysu
    edge_groups = group_polygon_edges(polygon, angle_threshold)

    if not edge_groups:
        return []

    segments_created = []

    for group in edge_groups:
        for edge in group.edges:
            p1 = edge['start']
            p2 = edge['end']

            # Vypočítaj vnútorné body
            normal = get_inward_normal(p1, p2)
            p1_inner = (p1[0] + normal[0] * thickness, p1[1] + normal[1] * thickness)
            p2_inner = (p2[0] + normal[0] * thickness, p2[1] + normal[1] * thickness)

            # Vonkajšia vertikálna stena
            builder.add_quad(
                [p2[0], 0, p2[1]],
                [p1[0], 0, p1[1]],
                [p1[0], height, p1[1]],
                [p2[0], height, p2[1]],
            )

            # Vnútorná vertikálna stena
            builder.add_quad(
                [p1_inner[0], 0, p1_inner[1]],
                [p2_inner[0], 0, p2_inner[1]],
                [p2_inner[0], height, p2_inner[1]],
                [p1_inner[0], height, p1_inner[1]],
            )

            # Spodná plocha
            builder.add_quad(
                [p1[0], 0, p1[1]],
                [p2[0], 0, p2[1]],
                [p2_inner[0], 0, p2_inner[1]],
                [p1_inner[0], 0, p1_inner[1]],
            )

            # Vrchná plocha
            builder.add_quad(
                [p2[0], height, p2[1]],
                [p1[0], height, p1[1]],
                [p1_inner[0], height, p1_inner[1]],
                [p2_inner[0], height, p2_inner[1]],
            )

        # Vytvor mesh pre tento segment
        segment_type_prefix = {
            "straight": "Wall",
            "angled": "AngledWall",
            "curved": "CurvedWall"
        }.get(group.segment_type, "Wall")

        builder.create_mesh(f"{segment_type_prefix}_", None, None)
        segments_created.append(group.segment_type)

    # Spracuj vnútorné hranice (diery)
    for hole in polygon.interiors:
        hole_coords = list(hole.coords)
        if len(hole_coords) > 1 and hole_coords[0] == hole_coords[-1]:
            hole_coords = hole_coords[:-1]

        hole_polygon = Polygon(hole_coords)
        hole_groups = group_polygon_edges(hole_polygon, angle_threshold)

        for group in hole_groups:
            for edge in group.edges:
                p1 = edge['start']
                p2 = edge['end']

                # Pre diery je normála opačná
                normal = get_inward_normal(p1, p2)
                p1_outer = (p1[0] - normal[0] * thickness, p1[1] - normal[1] * thickness)
                p2_outer = (p2[0] - normal[0] * thickness, p2[1] - normal[1] * thickness)

                # Vnútorná stena (smerom do miestnosti)
                builder.add_quad(
                    [p1[0], 0, p1[1]],
                    [p2[0], 0, p2[1]],
                    [p2[0], height, p2[1]],
                    [p1[0], height, p1[1]],
                )

                # Vonkajšia stena
                builder.add_quad(
                    [p2_outer[0], 0, p2_outer[1]],
                    [p1_outer[0], 0, p1_outer[1]],
                    [p1_outer[0], height, p1_outer[1]],
                    [p2_outer[0], height, p2_outer[1]],
                )

                # Spodná plocha
                builder.add_quad(
                    [p2[0], 0, p2[1]],
                    [p1[0], 0, p1[1]],
                    [p1_outer[0], 0, p1_outer[1]],
                    [p2_outer[0], 0, p2_outer[1]],
                )

                # Vrchná plocha
                builder.add_quad(
                    [p1[0], height, p1[1]],
                    [p2[0], height, p2[1]],
                    [p2_outer[0], height, p2_outer[1]],
                    [p1_outer[0], height, p1_outer[1]],
                )

            segment_type_prefix = {
                "straight": "Wall",
                "angled": "AngledWall",
                "curved": "CurvedWall"
            }.get(group.segment_type, "Wall")

            builder.create_mesh(f"{segment_type_prefix}_", None, None)
            segments_created.append(group.segment_type)

    return segments_created


def build_all_wall_segments(wall_polygons: list[Polygon], builder, height: float = 2.6):
    """
    Vytvorí 3D geometriu pre všetky stenové polygóny.
    """
    all_segments = []
    for polygon in wall_polygons:
        segments = build_wall_segments_geometry(polygon, builder, height)
        all_segments.extend(segments)
    return all_segments


# Pre spätnú kompatibilitu
@dataclass
class WallSegmentCompat:
    polygon: Polygon
    segment_type: str
    direction: np.ndarray | None
    index: int

WallSegment = WallSegmentCompat


def segment_all_walls(wall_polygons: list[Polygon], debug_output=None) -> list[WallSegmentCompat]:
    """Pre spätnú kompatibilitu - vráti prázdny list, geometria sa vytvára priamo."""
    return []
