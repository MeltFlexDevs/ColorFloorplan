from typing import Any

import numpy as np
from gltflib import GLTF, Accessor, AccessorType, Asset, Attributes, Buffer, BufferView, ComponentType, FileResource, GLTFModel, Mesh, Node, Primitive, PrimitiveMode, Scene


def create(list: list, resource: Any):
    id = len(list)
    list.append(resource)
    return id


class MeshBuilder:
    def __init__(self) -> None:
        self.gltf_nodes: list[Node] = []
        self.gltf_buffers: list[Buffer] = []
        self.gltf_resources: list[Any] = []
        self.gltf_buffer_views: list[BufferView] = []
        self.gltf_accessors: list[Accessor] = []
        self.gltf_meshes: list[Mesh] = []
        self.vertex_stack: list[list[float]] = []
        self.index_stack: list[int] = []

        self.counters: dict[str, int] = {}
        pass

    def add_quad(self, a, b, c, d, invert_normals=False):
        i = len(self.vertex_stack)
        if invert_normals:
            self.index_stack.extend([i + 0, i + 3, i + 2, i + 2, i + 1, i + 0])
        else:
            self.index_stack.extend([i + 0, i + 1, i + 2, i + 2, i + 3, i + 0])
        self.vertex_stack.extend([a, b, c, d])

    def extrude_shape(self, indices, vertices, height: float = 0, offset: float = 0):
        if height == 0:
            # If the height is zero, the result is a floor
            self.add_mesh_segment(indices[::-1], np.insert(vertices, 1, offset, axis=1))
            return

        # Bottom
        self.add_mesh_segment(indices, np.insert(vertices, 1, offset, axis=1))
        # Top
        self.add_mesh_segment(indices[::-1], np.insert(vertices, 1, offset + height, axis=1))
        # Walls; we need to find all exterior edges, to do this, find all edges that are only used
        # by one triangle in the input. Because the edges are sorted for lookup, also maintain a
        # dict for the original edge order
        edge_usage_count: dict[tuple[int, int], int] = {}
        actual_edges: dict[tuple[int, int], tuple[int, int]] = {}

        for i in range(0, len(indices), 3):
            triangle = (indices[i], indices[i + 1], indices[i + 2])

            for edge in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]:
                # Sort to count occurrences regardless of direction
                sorted_edge = tuple(sorted(edge))
                edge_usage_count[sorted_edge] = edge_usage_count.get(sorted_edge, 0) + 1
                actual_edges[sorted_edge] = edge

        for edge in edge_usage_count.keys():
            count = edge_usage_count[edge]
            if count != 1:
                continue

            actual_edge = actual_edges[edge]
            a = vertices[actual_edge[1]]
            b = vertices[actual_edge[0]]
            self.add_quad(
                [a[0], offset, a[1]],
                [b[0], offset, b[1]],
                [b[0], offset + height, b[1]],
                [a[0], offset + height, a[1]],
            )

    def add_mesh_segment(self, indices, vertices):
        offset = len(self.vertex_stack)
        self.vertex_stack.extend(vertices)
        self.index_stack.extend(np.add(indices, offset))

    def resolve_name_with_counter(self, name: str):
        if not name.endswith("_"):
            return name

        count = self.counters.get(name, 1)
        self.counters[name] = count + 1
        return f"{name}{count}"

    def create_mesh(self, name: str | list[str], indices, vertices, invert_normals=False):
        if indices is None:
            indices = self.index_stack
            self.index_stack = []

        if vertices is None:
            vertices = self.vertex_stack
            self.vertex_stack = []

        if type(name) == str:
            name = self.resolve_name_with_counter(name)
            names = [name]
        else:
            names = [self.resolve_name_with_counter(v) for v in name]
            name = names[0]

        # 1. Define the Mesh Data (Vertices and Indices)

        # Example: A simple square (two triangles)
        # Vertices (x, y, z) - float32 is common for positions
        # Note: glTF uses a right-handed coordinate system, typically Y-up.
        vertices = np.array(
            vertices,
            dtype=np.float32,
        )

        # Indices (to form two triangles: 0-1-2 and 1-3-2) - uint16 is common for indices

        indices = np.array(
            indices,
            dtype=np.uint16,
        )

        if invert_normals:
            indices = indices[::-1]

        # 2. Convert Data to Binary Buffers

        # Combine all data into a single binary buffer for efficiency
        # gltflib handles the packing into a single bytes object.
        binary_blob = vertices.tobytes() + indices.tobytes()

        resource_name = name + ".bin"
        resource = FileResource(resource_name, data=binary_blob)
        self.gltf_resources.append(resource)

        buffer = create(self.gltf_buffers, Buffer(byteLength=len(binary_blob), uri=resource_name))

        # 3. Define Buffer Views

        # A BufferView describes a segment of a Buffer.

        # For Vertices (Position data)
        vertex_byte_offset = 0
        vertex_byte_length = vertices.nbytes
        vertex_buffer_id = create(
            self.gltf_buffer_views,
            BufferView(
                buffer=buffer,
                byteOffset=vertex_byte_offset,
                byteLength=vertex_byte_length,
                # target=34962,  # ARRAY_BUFFER (Optional, but good practice)
            ),
        )

        # For Indices
        index_byte_offset = vertex_byte_length  # Starts after the vertices
        index_byte_length = indices.nbytes
        index_buffer_id = create(
            self.gltf_buffer_views,
            BufferView(
                buffer=buffer,
                byteOffset=index_byte_offset,
                byteLength=index_byte_length,
                # target=34963,  # ELEMENT_ARRAY_BUFFER (Optional, but good practice)
            ),
        )

        # 4. Define Accessors

        # Accessors define how to interpret the data in a BufferView.

        # Accessor for Positions (Vertices)
        position_accessor = create(
            self.gltf_accessors,
            Accessor(
                bufferView=vertex_buffer_id,
                byteOffset=0,
                componentType=ComponentType.FLOAT,
                count=len(vertices),
                type=AccessorType.VEC3.value,
                max=vertices.max(axis=0).tolist(),
                min=vertices.min(axis=0).tolist(),
            ),
        )

        # Accessor for Indices
        index_accessor = create(
            self.gltf_accessors,
            Accessor(
                bufferView=index_buffer_id,
                byteOffset=0,
                componentType=ComponentType.UNSIGNED_SHORT,  # Corresponds to np.uint16
                count=len(indices),
                type=AccessorType.SCALAR.value,  # Single value per index
            ),
        )

        # 5. Build the Mesh Primitive

        # A Primitive is the actual drawing geometry (e.g., a set of triangles).
        primitive = Primitive(
            attributes=Attributes(POSITION=position_accessor),  # POSITION uses the first accessor (index 0)
            indices=index_accessor,  # Indices use the second accessor (index 1)
            mode=PrimitiveMode.TRIANGLES.value,
        )

        mesh = create(self.gltf_meshes, Mesh(primitives=[primitive], name=name))
        for node_name in names:
            create(self.gltf_nodes, Node(mesh=mesh, name=node_name))

    def build(self):
        model = GLTFModel(
            asset=Asset(version="2.0"),
            scenes=[Scene(nodes=list(range(len(self.gltf_nodes))))],
            nodes=self.gltf_nodes,
            meshes=self.gltf_meshes,
            buffers=self.gltf_buffers,
            bufferViews=self.gltf_buffer_views,
            accessors=self.gltf_accessors,
        )

        return GLTF(model=model, resources=self.gltf_resources)
