from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from model.detection_service import detect_objects
from preProcess.preprocessor import preprocess_image
from utils.metadata_utils import get_gps_info
from utils.remove_bg import remove_background
from utils.image_utils import crop_by_bbox
from utils.color_utils import kmeans_color_analysis, rgb_to_color_name
from utils.geocode_utils import reverse_geocode

import os
import tempfile
import cv2
import numpy as np
import logging

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 5

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def file_size_within_limit(file):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    return size <= MAX_FILE_SIZE_MB * 1024 * 1024

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image provided."}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"success": False, "message": "Empty filename."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"success": False, "message": "Invalid file type."}), 400

    if not file_size_within_limit(image_file):
        return jsonify({"success": False, "message": "File too large."}), 400

    try:
        image_bytes = image_file.read()

        # Görseli diske kaydet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(image_bytes)
            temp.flush()
            image_path = temp.name

        # GPS ve adres bilgisi
        gps = get_gps_info(image_path)
        address = reverse_geocode(gps["latitude"], gps["longitude"]) if gps else None

        # Görseli OpenCV formatına çevir
        np_arr = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if original_image is None:
            raise Exception("Invalid image content.")

        # Preprocessing
        preprocessed_image = preprocess_image(image_bytes)
        cv2.imwrite(f"debug_bbox_.jpg", preprocessed_image)

        # Nesne tespiti
        detections = detect_objects(preprocessed_image)
        filtered = [d for d in detections if d.get('score', 0) >= 0.5]

        results = []
        for i, det in enumerate(filtered):
            bbox_dict = det['bounding_box']
            bbox = [bbox_dict["ymin"], bbox_dict["xmin"], bbox_dict["ymax"], bbox_dict["xmax"]]
            class_name = det['class']
            score = det['score']

            # Bounding box alanını kırp
            crop = crop_by_bbox(original_image, bbox, original_image.shape[:2])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as crop_temp:
                cv2.imwrite(crop_temp.name, crop)
                crop_path = crop_temp.name

            # Arka planı kaldır
            try:
                output_path = crop_path.replace(".jpg", "_nobg.png")
                remove_background(crop_path, output_path)
                masked_image = cv2.imread(output_path)
                if masked_image is None:
                    raise Exception("Masked image could not be loaded.")
            except Exception as e:
                app.logger.warning(f"Remove.bg failed for object {i}: {e}")
                masked_image = crop  # fallback
            cv2.imwrite(f"debug_bbox_{i}.jpg", crop)
            cv2.imwrite(f"debug_masked_{i}.png", masked_image)

            # Renk analizi
            dominant_rgb, _ = kmeans_color_analysis(masked_image)
            color_name = rgb_to_color_name(dominant_rgb) if dominant_rgb else "undefined"

            results.append({
                "class": class_name,
                "score": score,
                "dominant_color_rgb": dominant_rgb,
                "color_name": color_name
            })

        return jsonify({
            "success": True,
            "objects": results,
            "location": {
                "latitude": gps["latitude"] if gps else None,
                "longitude": gps["longitude"] if gps else None,
                "country": address.get("country") if address else None,
                "state": address.get("state") if address else None,
                "city": address.get("city") if address else None,
                "road": address.get("road") if address else None
            }
        }), 200

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Internal error."}), 500

if __name__ == "__main__":
    app.run(debug=True)
