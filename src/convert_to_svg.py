from pathlib import Path
from subprocess import run
from sys import argv

import numpy as np
import PIL
import PIL.Image


def main(filename: str):
    input = Path(filename)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = PIL.Image.open(input).convert("RGB")
    pixels = np.array(input_image)

    output = output_dir / input.stem
    threshold = 200

    for component, colors in [
        ("windows", [np.array([255, 0, 0])]),
        ("doors", [np.array([0, 255, 0])]),
        ("walls", [np.array([0, 0, 255])]),
    ]:
        # Get pixels of the required colors
        combined_matches = np.zeros(pixels.shape[:2], dtype=bool)

        for color in colors:
            # Calculate distance for this specific color
            distances = np.linalg.norm(pixels - color, axis=2)

            # Combine with previous matches using logical OR (at least one match)
            combined_matches = np.logical_or(combined_matches, distances < threshold)

        # Convert boolean mask to 0/255 uint8 format
        matches = np.where(combined_matches, 0, 255).astype(np.uint8)

        # Save the intermediate image
        grayscale = np.repeat(matches[:, :, np.newaxis], 3, axis=2)
        intermediate = output.with_suffix(f".{component}.bmp")
        output_image = PIL.Image.fromarray(grayscale)
        output_image.save(intermediate)

        # Convert to SVG
        run(["vendor/potrace/potrace", "-b", "svg", intermediate, "-o", intermediate.with_suffix(".svg")], check=True)

        # Delete the intermediate file
        intermediate.unlink()


if __name__ == "__main__":
    main(argv[1])
