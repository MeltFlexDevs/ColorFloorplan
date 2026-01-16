import subprocess
from io import BytesIO
from pathlib import Path
from sys import argv
from typing import IO

import numpy as np
import PIL
import PIL.Image

from .config import DELETE_BITMAPS, OUTPUT_FOLDER


def convert_to_svg(name: str, input: Path | IO[bytes] | BytesIO):
    print(f"convert_to_svg({name}, {input})")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    input_image = PIL.Image.open(input).convert("RGB")
    pixels = np.array(input_image)

    output = OUTPUT_FOLDER / name
    threshold = 200

    processes: list[subprocess.Popen] = []
    intermediates: list[Path] = []

    for component, color in [
        ("windows", np.array([255, 0, 0])),
        ("doors", np.array([0, 255, 0])),
        ("walls", np.array([0, 0, 255])),
        ("balcony", np.array([255, 255, 0])),
    ]:
        # Get pixels matching the required colors
        distances = np.linalg.norm(pixels - color, axis=2)

        # Convert boolean mask to 0/255 uint8 format
        matches = np.where(distances < threshold, 0, 255).astype(np.uint8)

        # Save the intermediate image
        grayscale = np.repeat(matches[:, :, np.newaxis], 3, axis=2)
        intermediate = output.with_suffix(f".{component}.bmp")
        output_image = PIL.Image.fromarray(grayscale)
        output_image.save(intermediate)

        # Convert to SVG
        process = subprocess.Popen(["vendor/potrace/potrace", "-b", "svg", intermediate, "-o", intermediate.with_suffix(".svg")])
        processes.append(process)
        intermediates.append(intermediate)

    # Wait for child processes to exit
    for process in processes:
        code = process.wait()
        assert code == 0

    # Delete the intermediate file
    if DELETE_BITMAPS:
        for intermediate in intermediates:
            intermediate.unlink()


if __name__ == "__main__":
    input = Path(argv[1])
    convert_to_svg(input.name.split(".")[0], input)
