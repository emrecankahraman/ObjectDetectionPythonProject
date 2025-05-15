import cv2
import numpy as np

def preprocess_image(image_bytes):
    """
    Preprocess an image to enhance clarity and quality while preserving color information.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Processed image as numpy array in BGR format
    """
    # Decode image from bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image.")
    
    # Make a copy of the original image for processing
    processed = image.copy()
    
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    processed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Reduce noise with a subtle bilateral filter (preserves edges better than Gaussian)
    processed = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # Apply sharpening
    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    processed = cv2.filter2D(processed, -1, kernel)
    
    # Slightly increase saturation
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # Increase saturation by 20%
    s = np.clip(s, 0, 255).astype(np.uint8)
    enhanced_hsv = cv2.merge([h, s, v])
    processed = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return processed