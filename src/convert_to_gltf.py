from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from sys import argv

import numpy as np
from mapbox_earcut import triangulate_float32
from shapely import GeometryCollection, LineString, MultiPoint, Point, box, buffer, difference, oriented_envelope, union_all, make_valid
from shapely.affinity import affine_transform, scale
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from svgpathtools import svg2paths

from .config import DEBUG_ALL_SHAPES, DEBUG_DOOR_FIX, DEBUG_EXTENDING_OBJECTS, DEBUG_OUTPUT, DELETE_SVG, OUTPUT_FOLDER
from .MeshBuilder import MeshBuilder
from .wall_segmentation import build_all_wall_segments


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
        builder.extrude_shape(final_indices, final_vertices, height=height / 3)
        builder.extrude_shape(final_indices, final_vertices, height=height / 3, offset=height * 2 / 3)
        builder.create_mesh("Window_", None, None)
        return

    if name == "balcony":
        builder.extrude_shape(final_indices, final_vertices, height=height / 3)
        builder.create_mesh("Fence_", None, None)
        return

    if name == "doors":
        name = "Door"
        builder.extrude_shape(final_indices, final_vertices, height=height * 10 / 13)
        builder.create_mesh("Door_", None, None)
        builder.extrude_shape(final_indices, final_vertices, height=height * 3 / 13, offset=height * 10 / 13)
        builder.create_mesh("DoorTop_", None, None)
        return

    if name == "walls":
        # Steny sa spracúvajú cez build_wall_segments, nie tu
        builder.extrude_shape(final_indices, final_vertices, height=height)
        builder.create_mesh("Wall_", None, None)
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

    for component in ["windows", "walls", "doors", "balcony"]:
        filename = OUTPUT_FOLDER / f"{name}.{component}.svg"

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

    # Build geometry - steny sa spracúvajú samostatne so segmentáciou
    wall_polygons = []
    for shape in all_shapes:
        if shape.kind == "walls":
            wall_polygons.extend(shape.polygons)
        else:
            build_shape(shape.kind, shape.polygons, builder)

    # Segmentuj a postav steny - každý segment je samostatný mesh
    if wall_polygons:
        build_all_wall_segments(wall_polygons, builder)

    # Combine all walls for cutting out room shapes
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

    # Build floor meshes
    for room in rooms:
        build_shape("room", [buffer(room, 15 * normalizer)], builder)

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
