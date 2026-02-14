import re
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path
from sys import argv

import numpy as np
from mapbox_earcut import triangulate_float32
from math import atan2, radians

from shapely import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, STRtree, box, buffer, difference, oriented_envelope, union_all, make_valid
from shapely.affinity import affine_transform, scale
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, split
from svgpathtools import svg2paths

from .config import DEBUG_ALL_SHAPES, DEBUG_DOOR_FIX, DEBUG_EXTENDING_OBJECTS, DEBUG_OUTPUT, DELETE_SVG, OUTPUT_FOLDER
from .MeshBuilder import MeshBuilder
from .room_labels import extract_room_labels


def _parse_potrace_svg_transform(svg_path: Path) -> tuple[float, float, float] | None:
    """Parse potrace SVG group transform to get (scale_x, scale_y, translate_y).

    Potrace generates SVGs like:
        <g transform="translate(0.000000,579.000000) scale(0.100000,-0.100000)">

    svgpathtools does NOT apply this transform, so path coordinates are in
    potrace's internal system (10x scale, y-up from bottom).  We need these
    values to convert pixel coordinates into the same space.

    Returns (scale_x, scale_y, translate_y) or None if parsing fails.
    """
    try:
        with open(svg_path) as f:
            # Only read the first 2KB — the transform is always near the top
            header = f.read(2048)

        m = re.search(
            r'transform\s*=\s*"'
            r"translate\(\s*[\d.+-]+\s*,\s*([\d.eE+-]+)\s*\)"
            r"\s*"
            r"scale\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)",
            header,
        )
        if m:
            ty = float(m.group(1))
            sx = float(m.group(2))
            sy = float(m.group(3))
            print(f"[convert_to_gltf] Parsed SVG transform: translate_y={ty}, scale=({sx}, {sy})")
            return (sx, sy, ty)
    except Exception as e:
        print(f"[convert_to_gltf] Warning: could not parse SVG transform: {e}")

    return None


def path_to_points(path, distance_step=1.0):
    points = []
    for segment in path:
        # If it's a Line, we only need the start and end
        if type(segment).__name__ == "Line":
            points.append((segment.start.real, segment.start.imag))
        else:
            # For curves, sample based on length to maintain density
            seg_len = segment.length()
            num_steps = max(int(seg_len / distance_step), 2)
            for i in range(num_steps):
                p = segment.point(i / num_steps)
                points.append((p.real, p.imag))

    # Add the final end point
    last_p = path[-1].end
    points.append((last_p.real, last_p.imag))
    return points


def load_shapes(file_path):
    print(f"load_shapes({file_path})")
    paths = svg2paths(file_path)[0]

    for super_path in paths:
        outline = []
        holes = []

        subpaths = super_path.continuous_subpaths()
        for i, path in enumerate(subpaths):
            if not path.isclosed():
                continue  # We can only triangulate closed shapes

            # 1. Discretize path into points
            points = path_to_points(path)

            # 2. Create a Shapely polygon for this specific path
            if len(points) >= 3:
                if i == 0:
                    outline.append(Polygon(points))
                else:
                    holes.append(Polygon(points))

        # 3. Merge overlapping paths to resolve holes automatically
        # This logic identifies which paths are inside others
        merged = make_valid(difference(make_valid(union_all(outline)), make_valid(union_all(holes))))
        yield get_shapes_in_geometry(merged)


def get_shapes_in_geometry(geometry: BaseGeometry):
    # Handle both single Polygons and MultiPolygons (multiple separate objects)
    if isinstance(geometry, Polygon):
        return [geometry]
    elif isinstance(geometry, MultiPolygon):
        return geometry.geoms
    else:
        # Unreachable
        return []


def svg_to_triangles_with_holes(file_path, name: str, builder: MeshBuilder):
    paths = svg2paths(file_path)[0]

    counter = 0

    for super_path in paths:
        outline = []
        holes = []

        subpaths = super_path.continuous_subpaths()
        for i, path in enumerate(subpaths):
            if not path.isclosed():
                continue  # We can only triangulate closed shapes

            # 1. Discretize path into points
            points = path_to_points(path)

            # 2. Create a Shapely polygon for this specific path
            if len(points) >= 3:
                if i == 0:
                    outline.append(Polygon(points))
                else:
                    holes.append(Polygon(points))

        # 3. Merge overlapping paths to resolve holes automatically
        # This logic identifies which paths are inside others
        merged = make_valid(difference(make_valid(union_all(outline)), make_valid(union_all(holes))))

        # Handle both single Polygons and MultiPolygons (multiple separate objects)
        if isinstance(merged, Polygon):
            merged = [merged]
        elif isinstance(merged, MultiPolygon):
            merged = merged.geoms
        else:
            # Unreachable
            return []

        counter += 1


debug_shape_counter = 0


@dataclass
class ConcaveCorner:
    index: int
    vertex: np.ndarray
    e_in: np.ndarray
    e_out: np.ndarray
    turn_angle: float


def find_concave_corners(coords: list[tuple[float, ...]], angle_threshold_deg: float = 30.0) -> list[ConcaveCorner]:
    """Walk exterior ring vertices (CCW) and identify concave corners above angle threshold."""
    n = len(coords)
    if n < 4:
        return []

    threshold_rad = radians(angle_threshold_deg)
    corners: list[ConcaveCorner] = []

    for i in range(n):
        prev_pt = np.array(coords[(i - 1) % n])
        curr_pt = np.array(coords[i])
        next_pt = np.array(coords[(i + 1) % n])

        e_in_raw = curr_pt - prev_pt
        e_out_raw = next_pt - curr_pt

        len_in = np.linalg.norm(e_in_raw)
        len_out = np.linalg.norm(e_out_raw)
        if len_in < 1e-10 or len_out < 1e-10:
            continue

        e_in = e_in_raw / len_in
        e_out = e_out_raw / len_out

        cross = e_in[0] * e_out[1] - e_in[1] * e_out[0]

        # For CCW polygon, cross < 0 means concave (right turn)
        if cross >= 0:
            continue

        dot_val = float(np.clip(np.dot(e_in, e_out), -1.0, 1.0))
        turn_angle = abs(atan2(abs(cross), dot_val))

        if turn_angle < threshold_rad:
            continue

        corners.append(ConcaveCorner(i, curr_pt, e_in, e_out, turn_angle))

    return corners


def compute_cut_line(polygon: Polygon, corner: ConcaveCorner, epsilon: float = 1e-4) -> LineString | None:
    """Compute a cutting line from concave vertex across the wall width.

    Tries extending in both the incoming edge direction and reversed outgoing
    edge direction, picking the shorter cut (which crosses the wall thickness).
    """
    vertex = corner.vertex
    boundary = polygon.exterior
    max_len = boundary.length

    candidates: list[tuple[float, LineString]] = []

    for direction in [corner.e_in, -corner.e_out]:
        far_point = vertex + direction * max_len
        ray = LineString([vertex - direction * epsilon, far_point])

        intersection = ray.intersection(boundary)
        if intersection.is_empty:
            continue

        # Extract all intersection points
        hit_points: list[Point] = []
        if isinstance(intersection, MultiPoint):
            hit_points = list(intersection.geoms)
        elif isinstance(intersection, Point):
            hit_points = [intersection]
        elif isinstance(intersection, GeometryCollection):
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    hit_points.append(geom)
                elif isinstance(geom, LineString):
                    for c in geom.coords:
                        hit_points.append(Point(c))
        elif isinstance(intersection, (LineString, MultiLineString)):
            for c in intersection.coords if isinstance(intersection, LineString) else [c for ls in intersection.geoms for c in ls.coords]:
                hit_points.append(Point(c))

        # Filter out points too close to origin
        vertex_point = Point(vertex)
        hits = [p for p in hit_points if vertex_point.distance(p) > epsilon * 10]

        if not hits:
            continue

        nearest = min(hits, key=lambda p: vertex_point.distance(p))
        dist = vertex_point.distance(nearest)

        # Create cut line, slightly extended for robust splitting
        dir_norm = direction / np.linalg.norm(direction)
        start = vertex - dir_norm * epsilon
        end = np.array(nearest.coords[0]) + dir_norm * epsilon
        candidates.append((dist, LineString([start, end])))

    if not candidates:
        return None

    # Pick shorter cut (crosses wall width, not length)
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def compute_corner_cut_line(polygon: Polygon, corner: ConcaveCorner, epsilon: float = 1e-4) -> LineString | None:
    """Compute an L-shaped cut line through a concave vertex.

    The cut goes: far_boundary_1 → vertex → far_boundary_2, crossing the polygon
    boundary at two points and passing through the concave vertex. This produces
    a clean 3-way split (two straight wall segments + a corner square piece)
    in a single Shapely split() call, avoiding epsilon-overlap issues.

    Falls back to a single straight cut if only one direction yields a hit.
    """
    vertex = corner.vertex
    boundary = polygon.exterior
    max_len = boundary.length
    vertex_point = Point(vertex)

    far_ends: list[tuple[np.ndarray, np.ndarray]] = []  # (extended_far_point, direction)

    for direction in [corner.e_in, -corner.e_out]:
        far_point = vertex + direction * max_len
        ray = LineString([vertex, far_point])

        intersection = ray.intersection(boundary)
        if intersection.is_empty:
            continue

        hit_points: list[Point] = []
        if isinstance(intersection, MultiPoint):
            hit_points = list(intersection.geoms)
        elif isinstance(intersection, Point):
            hit_points = [intersection]
        elif isinstance(intersection, GeometryCollection):
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    hit_points.append(geom)
                elif isinstance(geom, LineString):
                    for c in geom.coords:
                        hit_points.append(Point(c))
        elif isinstance(intersection, (LineString, MultiLineString)):
            for c in (intersection.coords if isinstance(intersection, LineString)
                      else [c for ls in intersection.geoms for c in ls.coords]):
                hit_points.append(Point(c))

        # Filter out hits at/near the vertex itself
        hits = [p for p in hit_points if vertex_point.distance(p) > epsilon * 10]
        if not hits:
            continue

        nearest = min(hits, key=lambda p: vertex_point.distance(p))
        dir_norm = direction / np.linalg.norm(direction)
        # Extend slightly past boundary for robust splitting
        end = np.array(nearest.coords[0]) + dir_norm * epsilon
        far_ends.append((end, dir_norm))

    if len(far_ends) == 2:
        # L-shaped cut: far_end_0 → vertex → far_end_1
        return LineString([far_ends[0][0], vertex, far_ends[1][0]])
    elif len(far_ends) == 1:
        # Fallback: single straight cut (same as original compute_cut_line)
        start = vertex - far_ends[0][1] * epsilon
        return LineString([start, far_ends[0][0]])

    return None


def split_wall_polygon(
    polygon: Polygon,
    angle_threshold_deg: float = 30.0,
    min_area: float = 0.001,
    _depth: int = 0,
) -> list[Polygon]:
    """Split a wall polygon at concave corners into individual straight segments.

    At each concave corner, an L-shaped cut through the vertex isolates a corner
    square piece between the two straight wall segments. Recurses on resulting
    pieces to handle T-shapes, U-shapes, etc.
    """
    if _depth > 20:
        return [polygon]

    polygon = orient(polygon, sign=1.0)
    coords = list(polygon.exterior.coords)[:-1]

    if len(coords) < 5:
        return [polygon]

    corners = find_concave_corners(coords, angle_threshold_deg)
    if not corners:
        return [polygon]

    # Try each corner, starting with the sharpest turn
    corners.sort(key=lambda c: -c.turn_angle)

    for corner in corners:
        cut_line = compute_corner_cut_line(polygon, corner)
        if cut_line is None:
            continue

        try:
            result = split(polygon, cut_line)
        except Exception:
            continue

        pieces = [g for g in result.geoms if isinstance(g, Polygon) and g.area > min_area]

        if len(pieces) < 2:
            continue

        # Successfully split — recurse on each piece
        final: list[Polygon] = []
        for piece in pieces:
            final.extend(split_wall_polygon(piece, angle_threshold_deg, min_area, _depth + 1))
        return final

    # No successful split found
    return [polygon]


def _wall_long_axis(wall_poly: Polygon):
    """Get the long axis direction and endpoints for a wall polygon via its minimum rotated rectangle."""
    obb = wall_poly.minimum_rotated_rectangle
    obb_coords = list(obb.exterior.coords)[:-1]

    edge1_len = np.linalg.norm(np.array(obb_coords[1]) - np.array(obb_coords[0]))
    edge2_len = np.linalg.norm(np.array(obb_coords[2]) - np.array(obb_coords[1]))

    if edge1_len >= edge2_len:
        start = (np.array(obb_coords[0]) + np.array(obb_coords[3])) / 2
        end = (np.array(obb_coords[1]) + np.array(obb_coords[2])) / 2
    else:
        start = (np.array(obb_coords[0]) + np.array(obb_coords[1])) / 2
        end = (np.array(obb_coords[3]) + np.array(obb_coords[2])) / 2

    return start, end


def split_wall_by_rooms(
    wall_poly: Polygon,
    rooms: "list[Polygon]",
    min_area: float = 0.001,
) -> list[Polygon]:
    """Split a wall polygon where it transitions from bordering one room to another.

    Samples points along both sides of the wall just outside its boundary.
    Where the adjacent room changes, a perpendicular cut is made.
    """
    start, end = _wall_long_axis(wall_poly)
    long_vec = end - start
    length = np.linalg.norm(long_vec)

    if length < 0.05:
        return [wall_poly]

    long_dir = long_vec / length
    perp = np.array([-long_dir[1], long_dir[0]])

    # Estimate wall half-thickness from OBB short dimension
    obb = wall_poly.minimum_rotated_rectangle
    obb_coords = list(obb.exterior.coords)[:-1]
    e1 = np.linalg.norm(np.array(obb_coords[1]) - np.array(obb_coords[0]))
    e2 = np.linalg.norm(np.array(obb_coords[2]) - np.array(obb_coords[1]))
    wall_half_thickness = min(e1, e2) / 2

    # Probe distance: just beyond the wall surface
    probe_dist = wall_half_thickness + 0.05

    # Prepare rooms for fast lookup via STRtree
    room_tree = STRtree(rooms)

    def find_room_id(point: Point) -> int:
        hits = room_tree.query(point)
        for idx in hits:
            if rooms[idx].contains(point):
                return idx
        return -1

    # Sample along wall length
    num_samples = max(int(length / 0.02), 20)

    assignments: list[tuple[float, int, int]] = []
    for i in range(num_samples + 1):
        t = i / num_samples
        center = start + long_vec * t
        side_a_pt = Point(center + perp * probe_dist)
        side_b_pt = Point(center - perp * probe_dist)
        ra = find_room_id(side_a_pt)
        rb = find_room_id(side_b_pt)
        assignments.append((t, ra, rb))

    # Find transition points (where room assignment changes on either side)
    transition_ts: list[float] = []
    for i in range(1, len(assignments)):
        prev_t, prev_a, prev_b = assignments[i - 1]
        curr_t, curr_a, curr_b = assignments[i]
        if prev_a != curr_a or prev_b != curr_b:
            mid_t = (prev_t + curr_t) / 2
            transition_ts.append(mid_t)

    if not transition_ts:
        return [wall_poly]

    # Deduplicate transitions that are very close together
    deduped: list[float] = [transition_ts[0]]
    for t in transition_ts[1:]:
        if t - deduped[-1] > 0.01:
            deduped.append(t)
    transition_ts = deduped

    # Create perpendicular cutting lines at each transition
    result = [wall_poly]
    for t in transition_ts:
        center = start + long_vec * t
        # Extend cut line well beyond the wall to ensure a clean split
        cut_start = center - perp * (length + 1)
        cut_end = center + perp * (length + 1)
        cut_line = LineString([cut_start, cut_end])

        new_result: list[Polygon] = []
        for poly in result:
            try:
                parts = split(poly, cut_line)
                new_result.extend(g for g in parts.geoms if isinstance(g, Polygon) and g.area > min_area)
            except Exception:
                new_result.append(poly)
        result = new_result

    return result if result else [wall_poly]


def triangulate_single_polygon(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a single polygon (with possible holes) into vertices and indices."""
    v_list = [np.array(poly.exterior.coords)[:-1]]
    rings = [len(v_list[0])]

    for hole in poly.interiors:
        v_list.append(np.array(hole.coords)[:-1])
        rings.append(len(v_list[-1]))

    vertices = np.vstack(v_list).astype(np.float32)

    if len(rings) > 1:
        ring_indices = np.cumsum(rings)[:-1].astype(np.uint32)
    else:
        ring_indices = np.array([], dtype=np.uint32)

    indices = triangulate_float32(vertices, np.concat([ring_indices, [np.uint32(len(vertices))]], dtype=np.uint32))
    return vertices, indices


def _classify_room_side(wall_center: np.ndarray, perp: np.ndarray, wall_half_thickness: float, rooms: "list[Polygon]", room_tree: STRtree | None) -> np.ndarray:
    """Determine which perpendicular direction is the 'inner' side (faces a room)."""
    if room_tree is None or len(rooms) == 0:
        return perp

    probe_dist = wall_half_thickness + 0.05

    def has_room(pt_2d):
        pt = Point(pt_2d)
        for idx in room_tree.query(pt):
            if rooms[idx].contains(pt):
                return True
        return False

    side_a = has_room(wall_center + perp * probe_dist)
    side_b = has_room(wall_center - perp * probe_dist)

    if side_b and not side_a:
        return -perp
    return perp  # default: positive perp = inner


def build_element_faces(
    poly: Polygon,
    rooms: "list[Polygon]",
    room_tree: STRtree | None,
    builder: MeshBuilder,
    prefix: str,
    num: int,
    seg_height: float,
    offset: float = 0.0,
):
    """Build a polygon section as separate face meshes (inner, outer, left, right, top, bottom).

    Works for walls, door-top walls, and window above/below walls.
    prefix + face + num forms the mesh name, e.g. WallInner_1, DoorTopOuter_3.
    """
    start, end = _wall_long_axis(poly)
    long_vec = end - start
    length = np.linalg.norm(long_vec)

    if length < 1e-10:
        verts, indices = triangulate_single_polygon(poly)
        builder.extrude_shape(indices, verts, height=seg_height, offset=offset)
        builder.create_mesh(f"{prefix}_{num}", None, None)
        return

    long_dir = long_vec / length
    perp = np.array([-long_dir[1], long_dir[0]])

    # Estimate wall half-thickness
    obb = poly.minimum_rotated_rectangle
    obb_coords = list(obb.exterior.coords)[:-1]
    e1 = np.linalg.norm(np.array(obb_coords[1]) - np.array(obb_coords[0]))
    e2 = np.linalg.norm(np.array(obb_coords[2]) - np.array(obb_coords[1]))
    wall_half_thickness = min(e1, e2) / 2

    wall_center = np.array(poly.centroid.coords[0])
    inner_perp = _classify_room_side(wall_center, perp, wall_half_thickness, rooms, room_tree)

    # Triangulate
    verts, indices = triangulate_single_polygon(poly)
    poly_centroid = np.array([poly.centroid.x, poly.centroid.y])

    # Find boundary edges (edges used by exactly 1 triangle)
    edge_usage: dict[tuple[int, int], int] = {}
    actual_edges: dict[tuple[int, int], tuple[int, int]] = {}

    for i in range(0, len(indices), 3):
        tri = (int(indices[i]), int(indices[i + 1]), int(indices[i + 2]))
        for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            se = tuple(sorted(edge))
            edge_usage[se] = edge_usage.get(se, 0) + 1
            actual_edges[se] = edge

    # Classify each boundary edge into face groups
    inner_quads: list[tuple] = []
    outer_quads: list[tuple] = []
    left_quads: list[tuple] = []
    right_quads: list[tuple] = []

    for se, count in edge_usage.items():
        if count != 1:
            continue
        ae = actual_edges[se]
        a = verts[ae[1]]  # reversed order (same as extrude_shape)
        b = verts[ae[0]]

        mid = (np.array(a) + np.array(b)) / 2
        edge_dir = np.array([b[0] - a[0], b[1] - a[1]])
        elen = np.linalg.norm(edge_dir)
        if elen < 1e-10:
            continue

        # Outward normal: perpendicular direction pointing AWAY from polygon centroid
        n1 = np.array([-edge_dir[1], edge_dir[0]]) / elen
        to_centroid = poly_centroid - mid
        outward_normal = n1 if np.dot(n1, to_centroid) < 0 else -n1

        quad = (
            [a[0], offset, a[1]],
            [b[0], offset, b[1]],
            [b[0], offset + seg_height, b[1]],
            [a[0], offset + seg_height, a[1]],
        )

        # Classify: normal mostly along perp → long side; mostly along long_dir → end cap
        dot_perp = abs(np.dot(outward_normal, perp))
        dot_long = abs(np.dot(outward_normal, long_dir))

        if dot_perp >= dot_long:
            if np.dot(outward_normal, inner_perp) > 0:
                inner_quads.append(quad)
            else:
                outer_quads.append(quad)
        else:
            if np.dot(outward_normal, long_dir) > 0:
                right_quads.append(quad)
            else:
                left_quads.append(quad)

    # Top face
    builder.add_mesh_segment(indices[::-1], np.insert(verts, 1, offset + seg_height, axis=1))
    builder.create_mesh(f"{prefix}Top_{num}", None, None)

    # Bottom face
    builder.add_mesh_segment(indices, np.insert(verts, 1, offset, axis=1))
    builder.create_mesh(f"{prefix}Bottom_{num}", None, None)

    # Inner face
    if inner_quads:
        for q in inner_quads:
            builder.add_quad(*q)
        builder.create_mesh(f"{prefix}Inner_{num}", None, None)

    # Outer face
    if outer_quads:
        for q in outer_quads:
            builder.add_quad(*q)
        builder.create_mesh(f"{prefix}Outer_{num}", None, None)

    # Left cap
    if left_quads:
        for q in left_quads:
            builder.add_quad(*q)
        builder.create_mesh(f"{prefix}Left_{num}", None, None)

    # Right cap
    if right_quads:
        for q in right_quads:
            builder.add_quad(*q)
        builder.create_mesh(f"{prefix}Right_{num}", None, None)


def build_wall_faces(
    wall_poly: Polygon,
    rooms: "list[Polygon]",
    room_tree: STRtree | None,
    builder: MeshBuilder,
    height: float = 2.6,
):
    """Build a single wall polygon as separate face meshes."""
    wall_num = builder.counters.get("Wall_", 1)
    builder.counters["Wall_"] = wall_num + 1
    build_element_faces(wall_poly, rooms, room_tree, builder, "Wall", wall_num, height, 0.0)


def build_walls(wall_polygons: "list[Polygon]", rooms: "list[Polygon]", room_tree: STRtree | None, builder: MeshBuilder):
    """Build wall meshes: split at corners, then by rooms, then decompose into faces."""
    for poly in wall_polygons:
        # Phase 1: Split at geometric corners (L/T/U shapes)
        corner_segments = split_wall_polygon(poly)

        # Phase 2: Split each straight segment by room adjacency
        for segment in corner_segments:
            room_segments = split_wall_by_rooms(segment, rooms)

            # Phase 3: Build each final segment as separate face meshes
            for final_segment in room_segments:
                build_wall_faces(final_segment, rooms, room_tree, builder)


def build_doors(door_polygons: "list[Polygon]", rooms: "list[Polygon]", room_tree: STRtree | None, builder: MeshBuilder, height: float = 2.6):
    """Build door meshes: door opening + wall-above-door decomposed into faces."""
    for poly in door_polygons:
        door_num = builder.counters.get("Door_", 1)
        builder.counters["Door_"] = door_num + 1

        verts, indices = triangulate_single_polygon(poly)

        # Door opening (solid box from 0 to door height)
        builder.extrude_shape(indices, verts, height=height * 10 / 13)
        builder.create_mesh(f"Door_{door_num}", None, None)

        # Wall above door - decomposed into 6 face meshes
        build_element_faces(poly, rooms, room_tree, builder, "DoorTop", door_num, height * 3 / 13, height * 10 / 13)


def build_windows(window_polygons: "list[Polygon]", rooms: "list[Polygon]", room_tree: STRtree | None, builder: MeshBuilder, height: float = 2.6):
    """Build window meshes: glass area + wall-below and wall-above decomposed into faces.

    Window top is explicitly tied to door top so they always match:
      - Door top:   height * 10 / 13  (same formula as build_doors)
      - Parapet:    0.0  to 0.8m      (fixed sill height)
      - Window:     0.8m to DOOR_TOP  (glass area)
      - Lintel:     DOOR_TOP to height (wall above window = wall above door)
    """
    DOOR_TOP  = height * 10 / 13      # 2.0m — MUST match build_doors door opening height
    PARAPET_H = 0.8                    # standard sill height (80cm)
    WINDOW_H  = DOOR_TOP - PARAPET_H  # 1.2m — glass area (parapet to door top)
    LINTEL_H  = height - DOOR_TOP     # 0.6m — wall above window = wall above door

    for poly in window_polygons:
        win_num = builder.counters.get("Window_", 1)
        builder.counters["Window_"] = win_num + 1

        verts, indices = triangulate_single_polygon(poly)

        # Window glass area (from parapet to door-top height)
        builder.extrude_shape(indices, verts, height=WINDOW_H, offset=PARAPET_H)
        builder.create_mesh(f"Window_{win_num}", None, None)

        # Wall below window (parapet) - decomposed into 6 face meshes
        build_element_faces(poly, rooms, room_tree, builder, "WindowBottom", win_num, PARAPET_H, 0.0)

        # Wall above window (lintel) - decomposed into 6 face meshes — matches DoorTop
        build_element_faces(poly, rooms, room_tree, builder, "WindowTop", win_num, LINTEL_H, DOOR_TOP)


def build_shape(name: str, merged: "list[Polygon]", builder: MeshBuilder):
    final_vertices = []
    final_indices = []
    v_offset = 0

    for poly in merged:
        # Prepare vertices: Shell points first, then all hole points
        v_list = [np.array(poly.exterior.coords)[:-1]]  # Remove duplicate last point
        rings = [len(v_list[0])]

        for hole in poly.interiors:
            v_list.append(np.array(hole.coords)[:-1])
            rings.append(len(v_list[-1]))

        vertices = np.vstack(v_list).astype(np.float32)

        # --- FIXED LOGIC START ---
        if len(rings) > 1:
            # We have holes. Define the end-index of each ring except the last one.
            # e.g., if rings are [100, 20], ring_indices should be [100]
            ring_indices = np.cumsum(rings)[:-1].astype(np.uint32)
        else:
            # Single polygon, no holes. Pass an empty array of type int32.
            ring_indices = np.array([], dtype=np.uint32)
        # --- FIXED LOGIC END ---

        indices = triangulate_float32(vertices, np.concat([ring_indices, [np.uint32(len(vertices))]], dtype=np.uint32))

        final_vertices.extend(vertices)
        final_indices.extend(indices + v_offset)
        v_offset += len(vertices)

    final_vertices = np.array(final_vertices)
    height = 2.6

    if DEBUG_OUTPUT and DEBUG_ALL_SHAPES:
        for polygon in merged:
            global debug_shape_counter
            DEBUG_OUTPUT.print(polygon, label=f"{debug_shape_counter}", stroke="#999999", labelColor="#999999")
            debug_shape_counter += 1

    if name == "room":
        builder.create_mesh(["Floor_", "Room_"], final_indices, np.insert(final_vertices, 1, 0, axis=1), invert_normals=True)
        return

    if name == "windows":
        raise RuntimeError("Windows must be built via build_windows(), not build_shape()")

    if name == "balcony":
        builder.extrude_shape(final_indices, final_vertices, height=height / 3)
        builder.create_mesh("Fence_", None, None)
        return

    if name == "doors":
        raise RuntimeError("Doors must be built via build_doors(), not build_shape()")
        return

    if name == "walls":
        raise RuntimeError("Walls must be built via build_walls(), not build_shape()")
        return

    return []


@dataclass
class Shape:
    kind: str
    polygons: list[Polygon]


simplification_tolerance = 0.01


@dataclass(eq=False)
class Rectangle:
    shape: Shape
    center: np.ndarray
    i: np.ndarray
    j: np.ndarray

    merged = False

    @cached_property
    def attachment_points(self):
        return [
            self.center + self.j,
            self.center - self.j,
        ]


def raycast(target: BaseGeometry, origin: np.ndarray, direction: np.ndarray, max_distance: float | np.floating):
    if np.allclose(direction, 0) or max_distance <= 0:
        return None
    ray = LineString([origin, Point(origin + direction * max_distance)])
    hit_point = target.intersection(ray)
    if isinstance(hit_point, MultiPoint):
        return min(hit_point.geoms, key=lambda v: np.linalg.norm(origin - v.coords[0]))
    elif isinstance(hit_point, Point):
        return hit_point

    # There is either not hit or we are inside the target boundary which should also be considered a miss
    return None


def get_rectangle_from_envelope(envelope: Polygon):
    coords = np.array(envelope.exterior.coords)

    i: np.ndarray = (coords[0] - coords[1]) * 0.5
    j: np.ndarray = (coords[1] - coords[2]) * 0.5

    center: np.ndarray = coords[1] + i - j
    return i, j, center


def mirror_by_line[T: BaseGeometry](geom: T, point, direction) -> T:
    x0, y0 = point
    dx, dy = direction

    # Normalize the direction vector
    mag = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / mag, dy / mag

    # Calculate matrix coefficients
    a = dx**2 - dy**2
    b = 2 * dx * dy
    d = 2 * dx * dy
    e = dy**2 - dx**2

    xoff = 2 * dy * (x0 * dy - y0 * dx)
    yoff = 2 * dx * (y0 * dx - x0 * dy)

    return affine_transform(geom, [a, b, d, e, xoff, yoff])


def extend_doors_and_windows_to_touch_walls(shapes: list[Shape]):
    # Doors and windows are expected to be rectangular in shape, so reduce them to their oriented
    # envelope (a rectangle). This will simplify their model, but more importantly, we can easily
    # get their axes. This will allow us to extend them to touch their surrounding walls. Some
    # windows/doors should be touching other windows/doors.

    rects: list[Rectangle] = []

    # Get the combined boundaries of all walls for raycasting
    walls = [shape for shape in shapes if shape.kind == "walls" or shape.kind == "balcony"]
    walls_geometry = union_all(list(chain.from_iterable(wall.polygons for wall in walls)))
    walls_boundary = walls_geometry.boundary
    if DEBUG_OUTPUT and DEBUG_EXTENDING_OBJECTS:
        DEBUG_OUTPUT.print(walls_boundary, "#00000055")
    max_distance = 0.15

    for shape in shapes:
        if shape.kind != "windows" and shape.kind != "doors":
            continue

        for polygon in shape.polygons:
            envelope = polygon.oriented_envelope
            assert isinstance(envelope, Polygon)

            if envelope.area / polygon.area >= 1.1:
                # This polygon was generated from a door that the AI has incorrectly transformed
                # from the input image. In this case, the shape is of the door in the wall plus the
                # arc that marks where the door will open. We need to extract only the base part of
                # the polygon.
                base_marker = polygon.buffer(0.05).intersection(walls_geometry).oriented_envelope

                if DEBUG_OUTPUT and DEBUG_DOOR_FIX:
                    DEBUG_OUTPUT.print(polygon, "red")
                    DEBUG_OUTPUT.print(base_marker, "blue")

                new_polygon = base_marker.intersection(polygon)
                if new_polygon.area == 0:
                    # There is only one wall touching the door, we need to mirror the base marker on
                    # the other side of the door to get the base. We need to determine the axis by
                    # which to mirror and then mirror.
                    i, j, center = get_rectangle_from_envelope(envelope)

                    # Determine the axis for mirroring by finding which side of the rectangle
                    # envelope the base is.

                    # 1. Find the point nearest to the base and the center that is inside the envelope.
                    alignment_point = nearest_points(base_marker, Point(center))[0]
                    alignment_point = nearest_points(envelope, alignment_point)[0]

                    # 2. Find which side of the envelope the point is by getting the local position inside the envelope.
                    A = np.array(alignment_point.coords[0])
                    A_i = np.dot(A, i)
                    A_j = np.dot(A, j)

                    # 3. The mirror axis is the other one.
                    if abs(A_i) > abs(A_j):
                        mirror_axis = j
                    else:
                        mirror_axis = i

                    # Mirror (the axis needs to be a normalized vector)
                    mirrored_base = mirror_by_line(base_marker, center, mirror_axis / np.linalg.norm(mirror_axis))
                    envelope = GeometryCollection([mirrored_base, base_marker]).oriented_envelope

                    if DEBUG_OUTPUT and DEBUG_DOOR_FIX:
                        DEBUG_OUTPUT.print(LineString([center - i * 2, center + i * 2]), "green")
                        DEBUG_OUTPUT.print(LineString([center - j * 2, center + j * 2]), "green")
                        DEBUG_OUTPUT.print(LineString([center - mirror_axis * 2, center + mirror_axis * 2]), "blue")
                        DEBUG_OUTPUT.print(mirrored_base, "purple")
                else:
                    envelope = new_polygon.oriented_envelope

                assert isinstance(envelope, Polygon)

                if DEBUG_OUTPUT and DEBUG_DOOR_FIX:
                    DEBUG_OUTPUT.print(envelope, "orange")

            i, j, center = get_rectangle_from_envelope(envelope)

            if np.linalg.norm(i) > np.linalg.norm(j):
                i, j = j, i

            rects.append(Rectangle(shape, center, i, j))

            if DEBUG_OUTPUT and DEBUG_EXTENDING_OBJECTS:
                DEBUG_OUTPUT.print(envelope, "black")
                DEBUG_OUTPUT.print(Point(center), "green")
                DEBUG_OUTPUT.print(Point(center + i), "blue")
                DEBUG_OUTPUT.print(Point(center + j), "red")

    # Merge all windows/wall that are next to each other
    i = 0
    while i < len(rects):
        first = rects[i]
        replaced = False

        for ii in range(i + 1, len(rects)):
            second = rects[ii]

            for attachment_a in first.attachment_points:
                for attachment_b in second.attachment_points:
                    distance = np.linalg.norm(attachment_a - attachment_b)
                    if distance > 0.15:
                        continue

                    # Reflect the attachment points to get the endpoints of the new rectangle
                    end_a = 2 * first.center - attachment_a
                    end_b = 2 * second.center - attachment_b
                    new_center = (end_a + end_b) * 0.5
                    new_j = (end_a - end_b) * 0.5
                    new_rect = Rectangle(first.shape, new_center, first.i, new_j)

                    rects[i] = new_rect
                    rects.pop(ii)
                    replaced = True
                    print(f"Replaced {i} + {ii}")
                    break
                if replaced:
                    break
            if replaced:
                break

        if replaced:
            continue

        i += 1

    # Extend all windows/doors to touch their nearest wall, for each side of the window/door, cast
    # three rays (left, center, right) to determine the distance we can extend the aforementioned to
    # make contact with a wall.
    for rectangle in rects:
        ends: list[np.ndarray] = []
        center = rectangle.center
        i = rectangle.i * 0.9
        j = rectangle.j
        j_size = np.linalg.norm(j)
        j_norm = j / j_size

        for j_direction in [1, -1]:
            hits: list[np.floating] = []
            vector = j_norm * j_direction

            for i_direction in [1, -1, 0]:
                side_origin = center + i * i_direction
                origin = side_origin + j * j_direction

                hit_point = raycast(walls_boundary, origin, vector, max_distance)
                if hit_point is None:
                    continue

                distance = np.linalg.norm(hit_point.coords[0] - side_origin) - j_size
                if DEBUG_OUTPUT and DEBUG_EXTENDING_OBJECTS:
                    DEBUG_OUTPUT.print(Point(origin), "orange")
                    DEBUG_OUTPUT.print(LineString([origin, hit_point]), "orange")

                hits.append(distance)

            if len(hits) == 0:
                ends.append(center + j * j_direction)
                continue

            max_dist = max(hits)
            min_dist = min(hits)

            hit_distance = min(max_dist, min_dist * 2)
            ends.append(center + j_norm * (hit_distance + j_size) * j_direction)

        end_a, end_b = ends
        rectangle.center = (end_a + end_b) * 0.5
        rectangle.j = (end_a - end_b) * 0.5

    # Replace the shapes with modified ones. Since we don't change walls, keep them as is.
    shapes.clear()
    shapes.extend(walls)
    for rectangle in rects:
        center = rectangle.center
        i = rectangle.i
        j = rectangle.j

        polygon = Polygon(
            [
                center + i + j,
                center + i - j,
                center - i - j,
                center - i + j,
            ]
        )

        if DEBUG_OUTPUT and DEBUG_EXTENDING_OBJECTS:
            DEBUG_OUTPUT.print(polygon, "green")

        shapes.append(Shape(rectangle.shape.kind, [polygon]))


def simplify_polygon(polygon: Polygon, kind: str):
    # Walls are very complex, and we need to preserve their shapes, but the SVG path to vertex
    # conversion often creates too many vertices on curves, so simplify them to optimize the
    # resulting model
    polygon = polygon.simplify(tolerance=simplification_tolerance)  # pyright: ignore[reportAssignmentType]

    return polygon


def convert_to_gltf(name: str):
    print(f"convert_to_gltf({name})")
    builder = MeshBuilder()

    all_shapes: list[Shape] = []

    svg_transform: tuple[float, float, float] | None = None

    for component in ["windows", "walls", "doors", "balcony"]:
        filename = OUTPUT_FOLDER / f"{name}.{component}.svg"

        # Parse the potrace SVG transform from the first available SVG (before deletion)
        if svg_transform is None and filename.exists():
            svg_transform = _parse_potrace_svg_transform(filename)

        for shape_polygons in load_shapes(filename):
            # Create a shape from the polygons parsed from the SVG path. The path may be degenerate
            # (a line) due to antialiasing pixels being detected as the correct color, ignore all
            # lines by calculating the polygon thickness (expected thickness for good polygons is in
            # the range 30..70, but all degenerate polygons are 1..4).
            shape = Shape(component, list(polygon for polygon in shape_polygons if polygon.area / polygon.exterior.length > 5))

            if len(shape.polygons) == 0:
                continue

            all_shapes.append(shape)

        if DELETE_SVG:
            filename.unlink()

    # Calculate average door width in pixels
    door_average_sum = 0
    door_average_count = 0
    for shape in all_shapes:
        if shape.kind != "doors":
            continue

        for polygon in shape.polygons:
            envelope = oriented_envelope(polygon)
            assert isinstance(envelope, Polygon)

            x, y = envelope.exterior.coords.xy
            side1 = Point(x[0], y[0]).distance(Point(x[1], y[1]))
            side2 = Point(x[1], y[1]).distance(Point(x[2], y[2]))
            width = max(side1, side2)

            door_average_sum += width
            door_average_count += 1

    # The average door with is 0.8m, normalize the floorplan so this is true
    average_door = door_average_sum / door_average_count if door_average_count > 0 else 80
    normalizer = 1 / (average_door / 0.8)

    # Normalize and simplify
    for shape in all_shapes:
        shape.polygons = [simplify_polygon(scale(polygon, xfact=normalizer, yfact=normalizer, origin=(0, 0)), kind=shape.kind) for polygon in shape.polygons]

    extend_doors_and_windows_to_touch_walls(all_shapes)

    # --- Compute rooms FIRST (needed for room-based wall splitting) ---
    all_polygons = list(chain.from_iterable(shape.polygons for shape in all_shapes))
    walls_shape = union_all(all_polygons)
    walls_shape = buffer(walls_shape, 10 * normalizer)
    # Join nearby walls together to fix gaps
    walls_shape = buffer(walls_shape, 10 * normalizer)
    walls_shape = buffer(walls_shape, -10 * normalizer)
    # Get floor polygons
    background = box(*walls_shape.bounds)
    rooms_shape = difference(background, walls_shape).simplify(tolerance=simplification_tolerance)
    rooms = get_shapes_in_geometry(rooms_shape)

    # Get all space that is inside the walls
    room_mask = GeometryCollection(
        [Polygon(interior) for polygon in (walls_shape.geoms if isinstance(walls_shape, MultiPolygon) else [walls_shape]) for interior in polygon.interiors],
    )

    # Filter rooms to be only inside the walls
    rooms = [room for room in rooms if room_mask.contains(room.representative_point())]

    # --- Build geometry (walls, doors, windows use face decomposition) ---
    room_tree = STRtree(rooms) if rooms else None

    for shape in all_shapes:
        if shape.kind == "walls":
            build_walls(shape.polygons, rooms, room_tree, builder)
        elif shape.kind == "doors":
            build_doors(shape.polygons, rooms, room_tree, builder)
        elif shape.kind == "windows":
            build_windows(shape.polygons, rooms, room_tree, builder)
        else:
            build_shape(shape.kind, shape.polygons, builder)

    # Build floor meshes
    for room in rooms:
        build_shape("room", [buffer(room, 15 * normalizer)], builder)

    # --- Match AI room labels to room polygons, embed in GLB extras ---
    labels, label_image_shape = extract_room_labels(name)
    room_data: dict[str, dict] = {}

    # Convert label positions from pixel coordinates to SVG raw coordinate space.
    # svgpathtools does NOT apply the potrace group transform, so room polygons
    # are in potrace's internal coords (10x scale, y-up from bottom).  Labels
    # come from the pixel mask (1x scale, y-down from top).
    if labels and svg_transform is not None:
        svg_sx, svg_sy, svg_ty = svg_transform
        print(f"[convert_to_gltf] Converting {len(labels)} label(s) from pixel to SVG coords "
              f"(sx={svg_sx}, sy={svg_sy}, ty={svg_ty})")
        for label in labels:
            px, py = label["position"]
            # Potrace transform: display = translate(0,ty) · scale(sx,sy) · raw
            #   display_x = raw_x * sx
            #   display_y = raw_y * sy + ty
            # So: raw_x = display_x / sx = pixel_x / sx
            #     raw_y = (display_y - ty) / sy = (pixel_y - ty) / sy
            raw_x = px / svg_sx
            raw_y = (py - svg_ty) / svg_sy
            label["position"] = (raw_x, raw_y)
            print(f"  Label '{label['letter']}' pixel({px:.0f},{py:.0f}) → svg_raw({raw_x:.0f},{raw_y:.0f})")
    elif labels and svg_transform is None and label_image_shape is not None:
        # Fallback: assume standard potrace defaults (scale=0.1/-0.1, translate=height)
        img_h = label_image_shape[0]
        print(f"[convert_to_gltf] SVG transform not parsed, using fallback (10x, y-flip, h={img_h})")
        for label in labels:
            px, py = label["position"]
            raw_x = px * 10.0
            raw_y = (img_h - py) * 10.0
            label["position"] = (raw_x, raw_y)

    for i, room in enumerate(rooms):
        room_name = f"Room_{i + 1}"
        centroid = room.centroid
        bounds = room.bounds  # (minx, miny, maxx, maxy)
        room_entry: dict = {
            "type": "unknown",
            "center": {"x": float(centroid.x), "z": float(centroid.y)},
            "bounds": {
                "minX": float(bounds[0]),
                "minZ": float(bounds[1]),
                "maxX": float(bounds[2]),
                "maxZ": float(bounds[3]),
            },
            "area": float(room.area),
            "outline": [[float(x), float(y)] for x, y in room.exterior.coords],
        }

        # Find which label falls inside this room (labels are now in SVG raw coords)
        for label in labels:
            lx, ly = label["position"]
            label_point = Point(lx * normalizer, ly * normalizer)
            if room.contains(label_point):
                room_entry["type"] = label["room_type"]
                room_entry["labelConfidence"] = float(label["confidence"])
                labels.remove(label)  # each label used once
                print(f"  ✓ {room_name} = {label['room_type']} (containment)")
                break

        room_data[room_name] = room_entry

    # Handle unmatched labels: assign to nearest room that has no type yet
    for label in labels:
        lx, ly = label["position"]
        label_point = Point(lx * normalizer, ly * normalizer)
        best_dist = float("inf")
        best_room_name: str | None = None

        for room_name, entry in room_data.items():
            if entry["type"] != "unknown":
                continue
            room_poly = rooms[int(room_name.split("_")[1]) - 1]
            dist = room_poly.exterior.distance(label_point)
            if dist < best_dist:
                best_dist = dist
                best_room_name = room_name

        if best_room_name is not None:
            room_data[best_room_name]["type"] = label["room_type"]
            room_data[best_room_name]["labelConfidence"] = float(label["confidence"])
            print(f"  ~ {best_room_name} = {label['room_type']} (nearest, dist={best_dist:.1f})")

    if room_data:
        builder.scene_extras = {"rooms": room_data}
        typed_count = sum(1 for r in room_data.values() if r["type"] != "unknown")
        print(f"[convert_to_gltf] Embedded {len(room_data)} rooms ({typed_count} classified) in GLB extras")

    gltf = builder.build()
    return gltf


if __name__ == "__main__":
    name = argv[1]

    try:
        gltf = convert_to_gltf(name)

        with open(OUTPUT_FOLDER / f"{name}.glb", "wb") as file:
            gltf.write_glb(file, save_file_resources=False)
    finally:
        if DEBUG_OUTPUT:
            (OUTPUT_FOLDER / f"{name}.js").write_text(DEBUG_OUTPUT.build())
