from io import BytesIO
from pathlib import Path
from sys import argv
from typing import IO

from .config import DEBUG_OUTPUT, OUTPUT_FOLDER
from .convert_to_gltf import convert_to_gltf
from .convert_to_svg import convert_to_svg


def process_input(name: str, input: Path | IO[bytes] | BytesIO):
    try:
        convert_to_svg(name, input)
        return convert_to_gltf(name)
    finally:
        if DEBUG_OUTPUT:
            (OUTPUT_FOLDER / f"{name}.js").write_text(DEBUG_OUTPUT.build())


if __name__ == "__main__":
    input = Path(argv[1])
    name = input.name.split(".")[0]
    gltf = process_input(name, input)
    with open(OUTPUT_FOLDER / f"{name}.glb", "wb") as file:
        gltf.write_glb(file, save_file_resources=False)
