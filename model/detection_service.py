import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import logging

# ==========================
# MODEL YÜKLEME VE AYARLAR
# ==========================
MODEL_PATH = "model/saved_model"
LABEL_MAP_PATH = "model/label_map.pbtxt"
SCORE_THRESHOLD = 0.5

try:
    detect_fn = tf.saved_model.load(MODEL_PATH)
    category_index = label_map_util.create_category_index_from_labelmap(
        LABEL_MAP_PATH, use_display_name=True
    )
    logging.info("Model ve etiketler başarıyla yüklendi.")
except Exception as e:
    logging.error(f"Model veya label_map yüklenemedi: {e}")
    raise

# ==========================
# TAHMİN FONKSİYONU
# ==========================
def detect_objects(image_np):
    try:
        if image_np is None or not isinstance(image_np, np.ndarray):
            raise ValueError("Görüntü geçersiz veya boş.")

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        h, w, _ = image_np.shape
        results = []

        for i in range(num_detections):
            score = detections['detection_scores'][i]
            if score < SCORE_THRESHOLD:
                continue

            cls_id = detections['detection_classes'][i]
            box = detections['detection_boxes'][i]  # [ymin, xmin, ymax, xmax]
            class_name = category_index.get(cls_id, {'name': 'unknown'})['name']

            results.append({
                "class": class_name,
                "score": float(score),
                "bounding_box": {
                    "ymin": float(box[0]),
                    "xmin": float(box[1]),
                    "ymax": float(box[2]),
                    "xmax": float(box[3])
                },
                "image_width": w,
                "image_height": h
            })

        return results

    except Exception as e:
        logging.error(f"Detection işlemi sırasında hata oluştu: {e}", exc_info=True)
        raise
