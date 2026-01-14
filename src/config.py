from math import inf
from pathlib import Path
from typing import Any, Callable

from shapely import LineString, MultiLineString, MultiPolygon, Point, Polygon

DELETE_BITMAPS = True
DELETE_SVG = True
OUTPUT_FOLDER = Path("output")


class DebugOutput:
    def __init__(self):
        self.fills: list[Callable[[], str]] = []
        self.strokes: list[Callable[[], str]] = []
        self.labels: list[Callable[[], str]] = []
        self.min_x = inf
        self.min_y = inf
        self.max_x = -inf
        self.max_y = -inf
        self.scale = 1

    def normalize(self, point: tuple[float, ...]):
        x, y = point
        return f"{(x - self.min_x) * (self.scale)}, {(y - self.min_y) * (self.scale)}"

    def include_point(self, point: tuple[float, ...]):
        assert len(point) == 2

        self.min_x = min(self.min_x, point[0])
        self.max_x = max(self.max_x, point[0])
        self.min_y = min(self.min_y, point[1])
        self.max_y = max(self.max_y, point[1])

        if self.min_x == self.max_x:
            self.max_x = self.min_x + 0.01

        if self.min_y == self.max_y:
            self.max_y = self.min_y + 0.01

        self.scale = min(
            1 / (self.max_x - self.min_x) if self.max_x != self.min_x else 1,
            1 / (self.max_y - self.min_y) if self.max_y != self.min_y else 1,
        )

    def print(self, shape: Any, stroke="#000000", fill="#00ff00", label: str | None = None, labelColor="#ff0000"):
        self.strokes.append(lambda: f'drawer.setStyle("{stroke}")')
        self.fills.append(lambda: f'drawer.setStyle(Color.fromHex("{fill}").opacity(0.25))')

        center: tuple[float, ...]

        if isinstance(shape, Point):
            center = shape.coords[0]
            self.include_point(center)
            self.fills.pop()
            self.strokes.append(lambda: f"drawer.beginPath().arc(new Point({self.normalize(center)}), 2 / scale).fill()")
        elif isinstance(shape, Polygon):
            center = shape.representative_point().coords[0]
            for point in shape.exterior.coords:
                self.include_point(point)

            self.fills.pop()
            self.strokes.append(lambda: f"drawer.beginPath().shape([{",".join(f"new Point({self.normalize(point)})" for point in shape.exterior.coords)}]).stroke()")
        elif isinstance(shape, MultiPolygon) or isinstance(shape, MultiLineString):
            self.strokes.pop()
            self.fills.pop()
            for geom in shape.geoms:
                self.print(geom, stroke=stroke, label=label, labelColor=labelColor)
            return
        elif isinstance(shape, LineString):
            center = shape.centroid.coords[0]
            for point in shape.coords:
                self.include_point(point)

            self.fills.pop()
            self.strokes.append(lambda: f"drawer.beginPath().shape([{",".join(f"new Point({self.normalize(point)})" for point in shape.coords)}]).stroke()")
        else:
            print(shape)
            assert False

        if label is not None:
            self.labels.append(lambda: f'drawer.setStyle("{labelColor}").fillText("{label}", new Point({self.normalize(center)}).mul(scale), font)')

    def build(self):
        self.include_point((self.min_x - 0.01 / self.scale, self.min_y - 0.01 / self.scale))
        self.include_point((self.max_x + 0.01 / self.scale, self.max_y + 0.01 / self.scale))

        result = [
            "function update() {",
            f"const scale = Math.min(...drawer.size.size())",
            'const font = {size: 12, font: "Arial", baseline: "middle", align: "center"}',
            "drawer.save().setNativeSize().scale(scale).setStrokeWidth(1 / scale)",
            *(callback() for callback in self.fills),
            *(callback() for callback in self.strokes),
            "drawer.restore()",
            *(callback() for callback in self.labels),
            "}",
        ]

        self.fills.clear()
        self.strokes.clear()
        self.labels.clear()

        self.min_x = inf
        self.min_y = inf
        self.max_x = -inf
        self.max_y = -inf
        self.scale = 1

        return "\n".join(result)


DEBUG_ALL_SHAPES = False
DEBUG_OUTPUT: DebugOutput | None = None
