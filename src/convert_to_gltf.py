from dataclasses import dataclass
from itertools import chain
from sys import argv

import numpy as np
from mapbox_earcut import triangulate_float32
from shapely import GeometryCollection, box, buffer, difference, union_all
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from svgpathtools import svg2paths

from MeshBuilder import MeshBuilder


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


def build_shape(name: str, merged: "list[Polygon]"):
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

        vertices = np.insert(vertices, 1, 0, axis=1)

        final_vertices.extend(vertices)
        final_indices.extend(indices + v_offset)
        v_offset += len(vertices)

    builder.create_mesh(f"{name}_", final_indices, final_vertices, invert_normals=True)


@dataclass
class Shape:
    kind: str
    value: list[Polygon]


def simplify_polygon(polygon: Polygon):
    polygon = polygon.simplify(tolerance=0.05)  # pyright: ignore[reportAssignmentType]
    return polygon


if __name__ == "__main__":
    builder = MeshBuilder()
    name = argv[1]

    all_shapes: list[Shape] = []

    for component in ["windows", "walls", "doors"]:
        for shape in load_shapes(f"output/{name}.{component}.svg"):
            all_shapes.append(Shape(component, list(shape)))

    # Calculate normalization factor
    all_polygons = list(chain.from_iterable(shape.value for shape in all_shapes))
    representative_width = sum((2 * wall.area) / wall.length for wall in all_polygons) / len(all_polygons)
    print(f"Wall thickness: {representative_width}")
    normalizer = 1 / representative_width

    # Normalize and simplify
    for shape in all_shapes:
        shape.value = [simplify_polygon(scale(polygon, xfact=normalizer, yfact=normalizer, origin=(0, 0))) for polygon in shape.value]

    # Build wall geometry
    for shape in all_shapes:
        build_shape(shape.kind, shape.value)

    # Need to recalculate, scale operation does not mutate
    all_polygons = list(chain.from_iterable(shape.value for shape in all_shapes))

    # Combine all walls for cutting out room shapes
    walls_shape = union_all(all_polygons)
    walls_shape = buffer(walls_shape, 10 * normalizer)
    # Join nearby walls together to fix gaps
    walls_shape = buffer(walls_shape, 30 * normalizer)
    walls_shape = buffer(walls_shape, -30 * normalizer)
    # Get floor polygons
    background = box(*walls_shape.bounds)
    rooms_shape = difference(background, walls_shape).simplify(tolerance=0.05)
    rooms = get_shapes_in_geometry(rooms_shape)

    # Get all space that is inside the walls
    room_mask = GeometryCollection(
        [Polygon(interior) for polygon in (walls_shape.geoms if isinstance(walls_shape, MultiPolygon) else [walls_shape]) for interior in polygon.interiors],
    )

    # Filter rooms to be only inside the walls
    rooms = [room for room in rooms if room_mask.contains(room.representative_point())]

    # Build floor meshes
    for room in rooms:
        build_shape("room", [room])

    gltf = builder.build()
    with open(f"output/{name}.glb", "wb") as file:
        gltf.write_glb(file, save_file_resources=False)
