from mimetypes import guess_extension, guess_type
from pathlib import Path
from sys import argv
from time import strftime

from google import genai
from google.oauth2 import service_account

prompt = """You are an expert in TRANSFORMING IMAGES, YOU TRANSFORM FIRST IMAGE INTO A CLEAN FLOORPLAN DRAWING.
 
⚠️ CRITICAL: The FIRST IMAGE (immediately after this text) is YOUR INPUT. Transform ONLY this image.
The other image is just reference examples - DO NOT copy or return them.
 
YOUR TASK:
Look at the FIRST IMAGE - this is the floorplan you must transform
Trace its walls, doors, balcony railing, and windows
Apply these colors: walls=BLUE, balcony railing=YELLOW, doors=GREEN only one solid rectangle for one door (represents closed doors), windows=RED one window = one solid rectangle
Keep the EXACT same room layout, proportions, and structure as the FIRST IMAGE
 
REMOVE from output:
All furniture, appliances, text, labels, dimensions, and annotations
whole door arcs that show door swings
 
OUTPUT REQUIREMENTS:
White background
Walls: solid blue fill (thick line)
Balcony railings: solid yellow rectangle (thick line)
Doors: solid green rectangle (thick line)
Windows: solid red rectangle (thick line)
 
The second image shows the color style and shape to use on doors (no arcs, just one thick line connected to walls), walls, balcony railings, and windows (reference only - DO NOT return this image).
 
VERIFY BEFORE GENERATING: Your output MUST match the room layout of the FIRST IMAGE, not any reference image, and must be closed from outside, THERE CANNOT BE ANY GAPS. And door must be solid rectangle with both ends connected to walls, as u can see it on the reference image - this is very IMPORTANT, often time you fail to do this. Make it exactly like i told you, this is VERY IMPORTANT for me, it can save my life."""


def main(filename: str):
    credentials = service_account.Credentials.from_service_account_file(
        filename="gcp-credentials.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    input_file = Path(filename)
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
                    input_image,
                    genai.types.Part.from_bytes(mime_type="image/png", data=Path("example.png").read_bytes()),
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
    main(argv[1])
