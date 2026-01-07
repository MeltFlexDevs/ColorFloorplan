from io import BytesIO
from secrets import token_urlsafe
from traceback import print_exc

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from .pipeline import process_input

application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})

application.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@application.route("/", methods=["POST"])
def convert():
    file = request.files["image"]

    if file.filename == "":
        return jsonify(error="No selected file"), 400

    try:
        name = token_urlsafe(16)
        gltf = process_input(name, request.files["image"].stream)

        bytes = BytesIO()
        gltf.write_glb(bytes)
        bytes.seek(0)
        return send_file(bytes, mimetype="model/gltf-binary")
    except Exception:
        print_exc()
        return jsonify(error=f"Processing failed"), 500


@application.errorhandler(413)
def request_entity_too_large(error):
    return jsonify(error="File is too large. Max size is 16MB."), 413


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=8081, debug=True)
