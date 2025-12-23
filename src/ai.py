from mimetypes import guess_extension, guess_type
from pathlib import Path
from time import strftime

from google import genai
from google.oauth2 import service_account

prompt = """Clean this architectural floorplan for 1:1 conversion.

REFERENCE IMAGE (first image): Shows EXACT symbols for doors and windows. Detect these precisely:
- Door: Wall opening + quarter-circle arc
- Window: 3 parallel lines in wall

EXAMPLE OUTPUT (second image): Shows the REQUIRED style of the output image
- The OUTPUT style must match EXAMPLE OUTPUT (second image) image.
- The EXAMPLE should only be used for determining the style, the layout of the output floorplan MUST be according to the INPUT IMAGE

INPUT IMAGE (third image):
- This is the floorplan, based on which the layout of the OUTPUT image should be determined
- The layout (positions and shapes of walls, doors and windows) of the output floorplan MUST be according to the INPUT IMAGE

CRITICAL - PRESERVE PROPORTIONS:
- Keep exact aspect ratio and dimensions of floorplan
- Maintain precise room sizes and relationships

REMOVE:
- Balconies (replace with solid wall - completely closed, no openings)
- Furniture, appliances, decorations
- Text, labels, dimensions, annotations
- Textures, patterns, symbols

Transform (using reference symbols):
- Walls: Solid blue color
- Doors: Solid green color rectangles
- Windows: Solid red color rectangles

The OUTPUT style must match EXAMPLE OUTPUT (second image) image.

OUTPUT: Clean drawing on white background with walls filled solid blue, doors with solid green, windows with solid red, exact proportions preserved."""


def main():
    credentials = service_account.Credentials.from_service_account_file(
        filename="gcp-credentials.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    input_file = Path("examples/complex.png")
    input_image = genai.types.Part.from_bytes(mime_type=guess_type(input_file)[0] or "", data=input_file.read_bytes())

    with genai.Client(
        vertexai=True,
        project=credentials.project_id,
        location="global",
        credentials=credentials,
        http_options=genai.types.HttpOptions(
            client_args={"proxy": None},
            async_client_args={"proxy": None},
        ),
    ) as client:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=genai.types.Content(
                role="user",
                parts=[
                    genai.types.Part(text=prompt),
                    genai.types.Part.from_bytes(mime_type="image/png", data=Path("reference.png").read_bytes()),
                    genai.types.Part.from_bytes(mime_type="image/png", data=Path("example.png").read_bytes()),
                    input_image,
                ],
            ),
            config=genai.types.GenerateContentConfig(
                response_modalities=[genai.types.Modality.IMAGE],
            ),
        )

        for candidate in response.candidates:  # pyright: ignore[reportOptionalIterable]
            for part in candidate.content.parts:  # pyright: ignore[reportOptionalIterable, reportOptionalMemberAccess]
                if not part.inline_data:
                    continue

                image_bytes = part.inline_data.data
                assert image_bytes is not None
                mime_type = part.inline_data.mime_type or "image/png"
                extension = guess_extension(mime_type) or ".png"

                Path(f"{strftime("%Y-%m-%d-%H-%M-%S")}{extension}").write_bytes(image_bytes)


if __name__ == "__main__":
    main()
