import numpy as np
from mapbox_earcut import triangulate_float32
from shapely import difference, union_all
from shapely.geometry import MultiPolygon, Polygon
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

        builder.create_mesh(f"{name}_{counter}", final_indices, final_vertices, invert_normals=True)
        counter += 1


def save_as_obj(vertices, triangles, filename="output.obj"):
    """
    Saves vertices and triangles to a Wavefront .obj file.

    Args:
        vertices: numpy array of shape (N, 2) or (N, 3)
        triangles: numpy array of shape (M, 3) - indices of vertices
        filename: string path to the output file
    """
    with open(filename, "w") as f:
        f.write("# Generated SVG to OBJ Converter\n")

        # 1. Write Vertices
        for v in vertices:
            # If 2D (x, y), add a 0 for z.
            height = v[2] if len(v) > 2 else 0.0
            f.write(f"v {v[0]} {height} {v[1]}\n")

        # 2. Write Faces (Triangles)
        # OBJ indices start at 1, so we add 1 to every index
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

    print(f"Successfully saved to {filename}")


# Usage:
# verts, tris = svg_to_triangles_with_holes('complex_logo.svg')
# save_as_obj(verts, tris, "my_model.obj")

# Usage
if __name__ == "__main__":
    builder = MeshBuilder()
    builder.create_mesh(
        "test",
        [0, 1, 2, 2, 3, 1],
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
    )

    for component in ["windows", "walls", "doors", "rooms"]:
        print(component)
        svg_to_triangles_with_holes(f"output/example.{component}.svg", component, builder)

    gltf = builder.build()
    gltf.export_glb("output/output.glb", embed_buffer_resources=True)
