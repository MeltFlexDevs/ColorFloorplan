from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from sys import argv

import numpy as np
from mapbox_earcut import triangulate_float32
from shapely import GeometryCollection, LineString, MultiPoint, Point, box, buffer, difference, oriented_envelope, union_all
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from svgpathtools import svg2paths

from .config import DEBUG_ALL_SHAPES, DEBUG_OUTPUT, DELETE_SVG, OUTPUT_FOLDER
from .MeshBuilder import MeshBuilder


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
        merged = difference(union_all(outline), union_all(holes))
        yield get_shapes_in_geometry(merged)


def get_shapes_in_geometry(geometry: BaseGeometry):
    # Handle both single Polygons and MultiPolygons (multiple separate objects)
    if isinstance(geometry, Polygon):
        return [geometry]
    elif isinstance(geometry, MultiPolygon):
        return geometry.geoms
    else:
        # Unreachable
        assert False


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
        merged = difference(union_all(outline), union_all(holes))

        # Handle both single Polygons and MultiPolygons (multiple separate objects)
        if isinstance(merged, Polygon):
            merged = [merged]
        elif isinstance(merged, MultiPolygon):
            merged = merged.geoms
        else:
            # Unreachable
            assert False

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

    if name == "doors":
        name = "Door"
        builder.extrude_shape(final_indices, final_vertices, height=height * 10 / 13)
        builder.create_mesh("Door_", None, None)
        builder.extrude_shape(final_indices, final_vertices, height=height * 3 / 13, offset=height * 10 / 13)
        builder.create_mesh("DoorTop_", None, None)
        return

    if name == "walls":
        name = "Wall"
        builder.extrude_shape(final_indices, final_vertices, height=height)
        builder.create_mesh("Wall_", None, None)
        return

    assert False


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
    ray = LineString([origin, Point(origin + direction * max_distance)])
    hit_point = target.intersection(ray)
    if isinstance(hit_point, MultiPoint):
        return min(hit_point.geoms, key=lambda v: np.linalg.norm(origin - v.coords[0]))
    elif isinstance(hit_point, Point):
        return hit_point

    # There is either not hit or we are inside the target boundary which should also be considered a miss
    return None


def extend_doors_and_windows_to_touch_walls(shapes: list[Shape]):
    # Doors and windows are expected to be rectangular in shape, so reduce them to their oriented
    # envelope (a rectangle). This will simplify their model, but more importantly, we can easily
    # get their axes. This will allow us to extend them to touch their surrounding walls. Some
    # windows/walls should be touching other windows/walls.

    rects: list[Rectangle] = []
    walls: list[Shape] = []

    for shape in shapes:
        if shape.kind == "walls":
            walls.append(shape)
            continue

        for polygon in shape.polygons:
            envelope = polygon.oriented_envelope
            assert isinstance(envelope, Polygon)
            coords = np.array(envelope.exterior.coords)

            i = (coords[0] - coords[1]) * 0.5
            j = (coords[1] - coords[2]) * 0.5

            center = coords[1] + i - j

            if np.linalg.norm(i) > np.linalg.norm(j):
                i, j = j, i

            rects.append(Rectangle(shape, center, i, j))

            if DEBUG_OUTPUT:
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

    # Get the combined boundaries of all walls for raycasting
    walls_boundary = union_all(list(chain.from_iterable(wall.polygons for wall in walls))).boundary
    if DEBUG_OUTPUT:
        DEBUG_OUTPUT.print(walls_boundary, "#00000055")
    max_distance = 0.15

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
                if DEBUG_OUTPUT:
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

        if DEBUG_OUTPUT:
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

    for component in ["windows", "walls", "doors"]:
        filename = OUTPUT_FOLDER / f"{name}.{component}.svg"

        for shape in load_shapes(filename):
            all_shapes.append(Shape(component, list(shape)))

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
    average_door = door_average_sum / door_average_count
    normalizer = 1 / (average_door / 0.8)

    # Normalize and simplify
    for shape in all_shapes:
        shape.polygons = [simplify_polygon(scale(polygon, xfact=normalizer, yfact=normalizer, origin=(0, 0)), kind=shape.kind) for polygon in shape.polygons]

    extend_doors_and_windows_to_touch_walls(all_shapes)

    # Build wall geometry
    for shape in all_shapes:
        build_shape(shape.kind, shape.polygons, builder)

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
