# Object Detection Python API

This project provides a Flask-based REST API that performs the following operations on images:

1. 📸 Image upload
2. 📍 Location and address extraction from GPS metadata (reverse geocoding)
3. 🤖 Object detection (TensorFlow, Faster R-CNN)
4. 🧼 Background removal of detected objects (via remove.bg API)
5. 🎨 Dominant color detection in cropped objects (using KMeans)

## 🔧 Setup

### 1. Activate Conda Environment

```bash
conda activate objtanima
```

### 2. Navigate to Project Directory

```bash
cd /path/to/model_server
```

### 3. Set Environment Variables

```bash
set PYTHONPATH=/path/to/models/research
set REMOVE_BG_API_KEY=your_api_key
```

> Replace `your_api_key` with your actual remove.bg API key.

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Start Flask Server

```bash
python app.py
```

The API will now accept POST requests at `http://127.0.0.1:5000/predict`.

---

## 🧠 SavedModel

Due to size constraints, model files are not included in this repository. Download the model from the link below and place it under `model/saved_model/`:

> [🔗 Google Drive: SavedModel](https://drive.google.com/drive/folders/1Qu_jDYV0iW4pQ1B9ndoug1N9WawwP6Wk?usp=sharing)

---

## 🔁 Workflow

1. User uploads an image to the API.
2. If available, location and address information are extracted from EXIF GPS data.
3. The image is converted to OpenCV format.
4. `preprocess_image()` enhances color saturation.
5. Object detection is performed using TensorFlow (Faster R-CNN).
6. Detected objects are cropped from the original image.
7. Each cropped object is sent to remove.bg for background removal.
8. Dominant color analysis is applied on the masked (background-removed) and preprocessed image.
9. Results are returned in JSON format.

---

## 📡 API Usage

### Endpoint:

```
POST /predict
```

### Form-Data:

* `image`: Image file in JPG, PNG, or JPEG format

### Sample Response:

```json
{
  "success": true,
  "objects": [
    {
      "class": "car",
      "score": 0.97,
      "color_name": "white",
      "dominant_color_rgb": [230, 230, 230]
    }
  ],
  "location": {
    "latitude": 41.01,
    "longitude": 29.01,
    "city": "Istanbul",
    "country": "Turkey"
  }
}
```

## 📦 Features

* Object detection using TensorFlow 2 (Faster R-CNN, ResNet101)
* Background removal using remove.bg API
* Color analysis with KMeans (on background-removed and preprocessed images)
* Location/address extraction from EXIF GPS data

---

## 📊 Model Information

Model: Faster R-CNN (ResNet101)

Training method: Fine-tuned from scratch using a balanced and custom dataset

Classes: car, cat, dog, bicycle, motorcycle

mAP: ~0.69 

## 📈 Model Limitations & Improvement Areas
The current model achieves an mAP of ~0.69, which is considered acceptable for a moderately sized dataset, but may be insufficient for production-grade applications.

* Performance drops significantly when detecting:

* Small objects (e.g., distant pets, small parts of vehicles)

* Overlapping objects in cluttered scenes

* Low-contrast objects in dark or poorly lit conditions

🔁 Improvement Suggestions
* Increase dataset diversity and size

* Apply targeted augmentation (e.g., blur, shadow, noise)

* Experiment with other models (e.g., EfficientDet, YOLOv8)
  
## 📄 License

MIT License
