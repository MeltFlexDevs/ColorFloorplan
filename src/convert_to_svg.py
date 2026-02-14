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

    # Define all components and their colors
    # The 5th component (labels/magenta) is processed separately – saved as a
    # numpy mask for OCR instead of being vectorised with potrace.
    components = [
        ("windows", np.array([255, 0, 0])),
        ("doors", np.array([0, 255, 0])),
        ("walls", np.array([0, 0, 255])),
        ("balcony", np.array([255, 255, 0])),
        ("labels", np.array([255, 0, 255])),
    ]

    # Calculate distances for all colors at once
    all_distances = np.stack([np.linalg.norm(pixels - color, axis=2) for _, color in components], axis=0)

    # Find the closest color for each pixel
    closest_color_idx = np.argmin(all_distances, axis=0)

    # Get the minimum distance for each pixel
    min_distances = np.min(all_distances, axis=0)

    processes: list[subprocess.Popen] = []
    intermediates: list[Path] = []

    for idx, (component, color) in enumerate(components):
        # Pixel belongs to this component only if:
        # 1. This is the closest color (exclusive assignment)
        # 2. The distance is below threshold
        matches = np.where((closest_color_idx == idx) & (min_distances < threshold), 0, 255).astype(np.uint8)

        if component == "labels":
            # Save magenta mask as numpy array for room label extraction.
            # 1 = magenta pixel, 0 = background.  Don't vectorise with potrace.
            label_mask = (matches == 0).astype(np.uint8)
            np.save(output.with_suffix(".labels.npy"), label_mask)
            continue

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
