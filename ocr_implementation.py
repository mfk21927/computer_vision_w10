import os
import cv2
import json
import numpy as np
import pytesseract
import easyocr
from flask import Flask, request, jsonify
from pytesseract import Output
import shutil

# --- STEP 1 & 2: Configuration ---
# Hardcoded path for Windows - Update if your path is different
TESS_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(TESS_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESS_PATH
else:
    # Fallback to system search
    detected = shutil.which("tesseract")
    if detected:
        pytesseract.pytesseract.tesseract_cmd = detected

# Initialize EasyOCR
print("Initializing EasyOCR (this may take a moment)...")
reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have CUDA

app = Flask(__name__)

# --- STEP 4, 5, 7, 8: OCR Core Logic ---
def run_tesseract(img):
    # Step 4 & 7: Extract text and bounding boxes
    # image_to_data returns a dictionary of every word found
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    results = []
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0: # Filter out empty detections
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            results.append({
                "text": d['text'][i],
                "confidence": float(d['conf'][i]),
                "box": [x, y, x + w, y + h]
            })
    full_text = pytesseract.image_to_string(img).strip()
    return full_text, results

def run_easyocr(img):
    # Step 5 & 7: Extract text and bounding boxes
    # returns list of (bbox, text, prob)
    raw_results = reader.readtext(img)
    results = []
    full_text_list = []
    for (bbox, text, prob) in raw_results:
        # Convert bbox list of lists to [x_min, y_min, x_max, y_max]
        (tl, tr, br, bl) = bbox
        results.append({
            "text": text,
            "confidence": round(float(prob) * 100, 2),
            "box": [int(tl[0]), int(tl[1]), int(br[0]), int(br[1])]
        })
        full_text_list.append(text)
    return " ".join(full_text_list), results

# --- STEP 9, 10, 11: Flask API ---
@app.route('/ocr', methods=['POST'])
def ocr_service():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Accept image upload
    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Run both engines
    t_text, t_boxes = run_tesseract(img)
    e_text, e_boxes = run_easyocr(img)

    # Return extracted text and boxes as JSON
    return jsonify({
        "tesseract": {
            "full_text": t_text,
            "confidence_avg": np.mean([b['confidence'] for b in t_boxes]) if t_boxes else 0,
            "detections": t_boxes
        },
        "easyocr": {
            "full_text": e_text,
            "confidence_avg": np.mean([b['confidence'] for b in e_boxes]) if e_boxes else 0,
            "detections": e_boxes
        }
    })

# --- STEP 6: Accuracy Comparison Helper ---
def compare_and_visualize(img_path):
    img = cv2.imread(img_path)
    t_text, t_boxes = run_tesseract(img)
    e_text, e_boxes = run_easyocr(img)
    
    print(f"\n--- Comparison for {os.path.basename(img_path)} ---")
    print(f"Tesseract Word Count: {len(t_boxes)}")
    print(f"EasyOCR Word Count: {len(e_boxes)}")
    
    # Visualization
    vis_img = img.copy()
    # Draw Tesseract in Green
    for b in t_boxes:
        x1, y1, x2, y2 = b['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw EasyOCR in Blue
    for b in e_boxes:
        x1, y1, x2, y2 = b['box']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    cv2.imshow("Green=Tess, Blue=Easy", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If you want to run a local test on a folder before starting the API:
    # compare_and_visualize("test_handwritten.jpg")
    
    app.run(debug=True, port=5000)