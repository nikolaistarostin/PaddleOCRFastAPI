import os
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from paddleocr import PaddleOCR
from PIL import Image
import io
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PaddleOCR FastAPI Wrapper",
    description="A simple FastAPI wrapper for PaddleOCR with API key protection",
    version="1.0.0"
)

# API Key configuration
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# OCR Language configuration
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en")
# Parse comma-separated languages
LANGUAGES_LIST = [lang.strip() for lang in OCR_LANGUAGES.split(",")]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Initialize PaddleOCR instances (one per language)
ocr_instances = {}


def get_ocr(lang: str = None):
    """
    Lazy initialization of PaddleOCR for specified language

    Args:
        lang: Language code (if None, uses first language from OCR_LANGUAGES)

    Returns:
        PaddleOCR instance for the specified language
    """
    global ocr_instances

    # Use first language if none specified
    if lang is None:
        lang = LANGUAGES_LIST[0]

    # Create OCR instance if it doesn't exist for this language
    if lang not in ocr_instances:
        ocr_instances[lang] = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,
            show_log=False
        )

    return ocr_instances[lang]


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the request header"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key


def process_file_to_images(contents: bytes, filename: str) -> List[Image.Image]:
    """
    Convert file (image or PDF) to list of PIL Images

    Args:
        contents: File contents as bytes
        filename: Original filename

    Returns:
        List of PIL Images
    """
    # Check if it's a PDF
    if filename.lower().endswith('.pdf'):
        try:
            # Convert PDF pages to images
            images = convert_from_bytes(contents)
            return images
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process PDF: {str(e)}"
            )
    else:
        # Try to open as image
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return [image]
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"File must be a valid image or PDF: {str(e)}"
            )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PaddleOCR FastAPI Wrapper",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint (no auth required)"""
    return {"status": "ok"}


@app.get("/languages")
async def get_languages():
    """Get list of configured OCR languages (no auth required)"""
    return {
        "configured_languages": LANGUAGES_LIST,
        "default_language": LANGUAGES_LIST[0]
    }


@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    lang: str = None,
    multilingual: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Perform OCR on an uploaded image or PDF file

    Args:
        file: Image file (jpg, png, etc.) or PDF
        lang: Optional language code (e.g., 'en', 'fr', 'ch'). If not specified, uses first configured language.
              For multilingual mode, use comma-separated languages (e.g., 'en,ru').
        multilingual: If True, processes document with multiple configured languages and merges results.

    Returns:
        JSON response with detected text and bounding boxes
    """
    try:
        # Parse languages for processing
        if multilingual:
            # Use all configured languages if multilingual=True and no lang specified
            process_langs = LANGUAGES_LIST if not lang else [l.strip() for l in lang.split(',')]
        else:
            # Single language mode
            process_langs = [lang] if lang else [LANGUAGES_LIST[0]]

        # Validate languages
        for l in process_langs:
            if l not in LANGUAGES_LIST:
                raise HTTPException(
                    status_code=400,
                    detail=f"Language '{l}' not configured. Available languages: {', '.join(LANGUAGES_LIST)}"
                )

        # Read file
        contents = await file.read()

        # Process file to images (handles both images and PDFs)
        images = process_file_to_images(contents, file.filename)

        all_text_blocks = []
        all_text_parts = []

        # Process with each language
        for page_num, image in enumerate(images, start=1):
            # Convert PIL Image to bytes for PaddleOCR
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # In multilingual mode, merge results from all languages
            if multilingual:
                page_results = {}  # Store results by bbox for deduplication

                for current_lang in process_langs:
                    ocr_engine = get_ocr(current_lang)
                    result = ocr_engine.ocr(img_bytes, cls=True)

                    if result is not None and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line is None:
                                continue

                            bbox = line[0]
                            text_info = line[1]

                            # Use bbox as key for deduplication (convert to tuple for hashing)
                            bbox_key = tuple(tuple(point) for point in bbox)

                            # Keep result with highest confidence
                            if bbox_key not in page_results or text_info[1] > page_results[bbox_key]['confidence']:
                                page_results[bbox_key] = {
                                    'text': text_info[0],
                                    'confidence': float(text_info[1]),
                                    'bounding_box': bbox,
                                    'page': page_num,
                                    'detected_lang': current_lang
                                }

                # Add merged results
                for result_data in page_results.values():
                    all_text_blocks.append(result_data)
                    all_text_parts.append(result_data['text'])
            else:
                # Single language mode
                ocr_engine = get_ocr(process_langs[0])
                result = ocr_engine.ocr(img_bytes, cls=True)

                if result is not None and len(result) > 0 and result[0] is not None:
                    for line in result[0]:
                        if line is None:
                            continue

                        bbox = line[0]
                        text_info = line[1]

                        all_text_blocks.append({
                            "text": text_info[0],
                            "confidence": float(text_info[1]),
                            "bounding_box": bbox,
                            "page": page_num
                        })
                        all_text_parts.append(text_info[0])

        return {
            "success": True,
            "filename": file.filename,
            "pages": len(images),
            "language": ','.join(process_langs),
            "multilingual": multilingual,
            "text_blocks": all_text_blocks,
            "full_text": "\n".join(all_text_parts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/text-only")
async def perform_ocr_text_only(
    file: UploadFile = File(...),
    lang: str = None,
    multilingual: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Perform OCR on an uploaded image or PDF file and return only the extracted text

    Args:
        file: Image file (jpg, png, etc.) or PDF
        lang: Optional language code (e.g., 'en', 'fr', 'ch'). If not specified, uses first configured language.
              For multilingual mode, use comma-separated languages (e.g., 'en,ru').
        multilingual: If True, processes document with multiple configured languages and merges results.

    Returns:
        JSON response with only the extracted text
    """
    try:
        # Parse languages for processing
        if multilingual:
            process_langs = LANGUAGES_LIST if not lang else [l.strip() for l in lang.split(',')]
        else:
            process_langs = [lang] if lang else [LANGUAGES_LIST[0]]

        # Validate languages
        for l in process_langs:
            if l not in LANGUAGES_LIST:
                raise HTTPException(
                    status_code=400,
                    detail=f"Language '{l}' not configured. Available languages: {', '.join(LANGUAGES_LIST)}"
                )

        # Read file
        contents = await file.read()

        # Process file to images (handles both images and PDFs)
        images = process_file_to_images(contents, file.filename)

        text_parts = []

        for image in images:
            # Convert PIL Image to bytes for PaddleOCR
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            if multilingual:
                # Collect results from all languages
                page_results = {}

                for current_lang in process_langs:
                    ocr_engine = get_ocr(current_lang)
                    result = ocr_engine.ocr(img_bytes, cls=True)

                    if result is not None and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line is None:
                                continue

                            bbox = line[0]
                            text_info = line[1]

                            bbox_key = tuple(tuple(point) for point in bbox)

                            # Keep result with highest confidence
                            if bbox_key not in page_results or text_info[1] > page_results[bbox_key]['confidence']:
                                page_results[bbox_key] = {
                                    'text': text_info[0],
                                    'confidence': text_info[1]
                                }

                # Add merged text
                for result_data in page_results.values():
                    text_parts.append(result_data['text'])
            else:
                # Single language mode
                ocr_engine = get_ocr(process_langs[0])
                result = ocr_engine.ocr(img_bytes, cls=True)

                if result is not None and len(result) > 0 and result[0] is not None:
                    for line in result[0]:
                        if line is None:
                            continue
                        text_parts.append(line[1][0])

        return {
            "success": True,
            "filename": file.filename,
            "pages": len(images),
            "language": ','.join(process_langs),
            "multilingual": multilingual,
            "text": "\n".join(text_parts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
