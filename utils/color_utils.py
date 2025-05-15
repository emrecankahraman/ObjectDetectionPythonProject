import cv2
import numpy as np
from sklearn.cluster import KMeans

# =======================
# GÜNCELLENMİŞ RENK TESPİTİ ARAÇLARI
# =======================

def crop_center_of_bbox(image, bbox, margin=0.15):
    """Bounding box'ın merkezini kes - kenarları dışarda bırak (camlar, tekerlekler, vb.)"""
    h, w, _ = image.shape
    y1, x1, y2, x2 = bbox
    x1 = int((x1 + margin * (x2 - x1)) * w)
    x2 = int((x2 - margin * (x2 - x1)) * w)
    y1 = int((y1 + margin * (y2 - y1)) * h)
    y2 = int((y2 - margin * (y2 - y1)) * h)
    return image[y1:y2, x1:x2]

def detect_dominant_color_hsv(image_crop, is_night=False, class_name=None):
    if image_crop.size == 0:
        return "undefined", 0, "none", 0, "undefined"

    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'red':      ((0, 70, 70), (10, 255, 255)),
        'green':    ((36, 70, 70), (85, 255, 255)),
        'blue':     ((85, 50, 50), (130, 255, 255)),
        'yellow':   ((26, 70, 70), (35, 255, 255)),
        'black':    ((0, 0, 0), (180, 255, 30)),
        'white':    ((0, 0, 200), (180, 40, 255)),
        'gray':     ((0, 0, 70), (180, 40, 199)),
        'red2':     ((170, 70, 70), (180, 255, 255)),
    }

    counts = {}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        counts[color] = cv2.countNonZero(mask)

    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    dominant_color = sorted_colors[0][0] if sorted_colors else "undefined"
    dominant_count = sorted_colors[0][1] if sorted_colors else 0
    second_color = sorted_colors[1][0] if len(sorted_colors) > 1 else "none"
    second_count = sorted_colors[1][1] if len(sorted_colors) > 1 else 0

    final_color = dominant_color
    if dominant_color == 'red2':
        final_color = 'red'
    elif second_color == 'red2' and dominant_color == 'red':
        final_color = 'red'

    return dominant_color, dominant_count, second_color, second_count, final_color

def kmeans_color_analysis(image_crop, k=3):
    image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    reshaped = image_crop.reshape((-1, 3))

    # Siyah alanları filtrele
    mask = ~np.all(reshaped < [40, 40, 40], axis=1)
    filtered = reshaped[mask]

    if filtered.size == 0:
        return None, []

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(filtered)
    counts = np.bincount(kmeans.labels_)
    sorted_idx = np.argsort(counts)[::-1]
    dominant_index = sorted_idx[0]
    dominant_rgb = tuple(map(int, kmeans.cluster_centers_[dominant_index]))

    all_colors = [(tuple(map(int, kmeans.cluster_centers_[i])), int(counts[i])) for i in sorted_idx]
    return dominant_rgb, all_colors

def rgb_to_color_name(rgb):
    r, g, b = rgb
    if max(r, g, b) < 50:
        return "black"
    if min(r, g, b) > 200:
        return "white"
    if abs(r - g) < 30 and abs(r - b) < 30 and abs(g - b) < 30:
        return "gray"
    if r > 1.5 * g and r > 1.5 * b:
        return "red"
    if g > 1.5 * r and g > 1.5 * b:
        return "green"
    if b > 1.5 * r and b > 1.5 * g:
        return "blue"
    if r > 200 and g > 200:
        return "yellow"
    if r > 1.2 * b and g > 1.2 * b:
        return "yellow"
    if r > 1.2 * g and b > 1.2 * g:
        return "purple"
    return "undefined"
