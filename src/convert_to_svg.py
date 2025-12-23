from pathlib import Path
from subprocess import run

import numpy as np
import PIL
import PIL.Image


def main():
    input = Path("examples/example.ai.png")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = PIL.Image.open(input).convert("RGB")
    pixels = np.array(input_image)

    output = output_dir / input.stem

    for component, color in [
        ("windows", np.array([255, 0, 0])),
        ("doors", np.array([0, 255, 0])),
        ("walls", np.array([0, 0, 255])),
        ("rooms", np.array([255, 255, 255])),
    ]:
        # Get pixels of the required colors
        distances = np.linalg.norm(pixels - color, axis=2)
        matches = np.where(distances < 127, 0, 255).astype(np.uint8)

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
    main()
