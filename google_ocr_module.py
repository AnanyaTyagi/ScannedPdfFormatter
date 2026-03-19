

import os
import io
from google.cloud import vision
from PIL import Image

def run_google_ocr(image_path, lang_hint="en"):
    """
    Run Google Vision OCR on a given image and return lines with bounding boxes,
    rescaled to PDF coordinates.
    """
    # 🌍 Prioritize JSON from environment variable if present
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        creds_path = "/tmp/google_creds.json"
        with open(creds_path, "w") as f:
            f.write(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/creds.json"

    client = vision.ImageAnnotatorClient()

    # Load image
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    image_context = vision.ImageContext(language_hints=[lang_hint])
    response = client.document_text_detection(image=image, image_context=image_context)
    doc = response.full_text_annotation

    results = []
    pixel_to_point = 72.0 / 300  # if your image is rendered at 300 DPI

    for page in doc.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                para_text = []
                x_min, y_min = float("inf"), float("inf")
                x_max, y_max = float("-inf"), float("-inf")

                for word in paragraph.words:
                    word_text = "".join([s.text for s in word.symbols])
                    para_text.append(word_text)

                    for v in word.bounding_box.vertices:
                        x_min = min(x_min, v.x or 0)
                        y_min = min(y_min, v.y or 0)
                        x_max = max(x_max, v.x or 0)
                        y_max = max(y_max, v.y or 0)

                text = " ".join(para_text).strip()
                if not text:
                    continue

                bbox = [
                    x_min * pixel_to_point,
                    y_min * pixel_to_point,
                    x_max * pixel_to_point,
                    y_max * pixel_to_point,
                ]

                results.append({"text": text, "bbox": bbox})

    return results


# Example usage:
if __name__ == "__main__":
    image_file = "page1.png"  # convert a PDF page to PNG first
    ocr_results = run_google_ocr(image_file)
    for r in ocr_results:
        print(r)
