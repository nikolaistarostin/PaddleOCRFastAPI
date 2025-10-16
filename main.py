import os
import tempfile
import re
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from paddleocr import PaddleOCR, PPStructureV3
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

# Initialize PP-StructureV3 instances (lazy loaded, one per language)
structure_instances = {}


def get_structure(lang: str = None):
    """
    Lazy initialization of PPStructureV3 for document analysis with language support

    Args:
        lang: Language code (if None, uses first language from OCR_LANGUAGES)

    Returns:
        PPStructureV3 instance for the specified language
    """
    global structure_instances

    # Use first language if none specified
    if lang is None:
        lang = LANGUAGES_LIST[0]

    # Create structure instance if it doesn't exist for this language
    if lang not in structure_instances:
        try:
            # PPStructureV3 with language support
            structure_instances[lang] = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=lang
            )
        except Exception as e:
            # Handle initialization errors properly
            raise HTTPException(
                status_code=500,
                detail=f"PPStructureV3 initialization failed for language '{lang}': {str(e)}. Models may be downloading or missing dependencies."
            )

    return structure_instances[lang]


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


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes

    Args:
        bbox1, bbox2: Bounding boxes as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    Returns:
        IoU score (0-1)
    """
    # Convert to [x_min, y_min, x_max, y_max]
    x1_min = min(p[0] for p in bbox1)
    y1_min = min(p[1] for p in bbox1)
    x1_max = max(p[0] for p in bbox1)
    y1_max = max(p[1] for p in bbox1)

    x2_min = min(p[0] for p in bbox2)
    y2_min = min(p[1] for p in bbox2)
    x2_max = max(p[0] for p in bbox2)
    y2_max = max(p[1] for p in bbox2)

    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    intersection_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Calculate union
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


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
                page_results = []  # Store all results with language info

                # Collect results from all languages
                for current_lang in process_langs:
                    ocr_engine = get_ocr(current_lang)
                    result = ocr_engine.ocr(img_bytes, cls=True)

                    if result is not None and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line is None:
                                continue

                            bbox = line[0]
                            text_info = line[1]

                            page_results.append({
                                'text': text_info[0],
                                'confidence': float(text_info[1]),
                                'bounding_box': bbox,
                                'page': page_num,
                                'detected_lang': current_lang
                            })

                # Merge overlapping results (keep highest confidence)
                merged_results = []
                used_indices = set()

                for i, result1 in enumerate(page_results):
                    if i in used_indices:
                        continue

                    # Find all overlapping results for this bbox
                    overlapping = [result1]
                    used_indices.add(i)

                    for j, result2 in enumerate(page_results):
                        if j <= i or j in used_indices:
                            continue

                        # Check if bboxes overlap significantly (IoU > 0.5)
                        iou = calculate_iou(result1['bounding_box'], result2['bounding_box'])
                        if iou > 0.5:
                            overlapping.append(result2)
                            used_indices.add(j)

                    # Keep the result with highest confidence
                    best_result = max(overlapping, key=lambda x: x['confidence'])
                    merged_results.append(best_result)

                # Add merged results
                for result_data in merged_results:
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
                page_results = []

                for current_lang in process_langs:
                    ocr_engine = get_ocr(current_lang)
                    result = ocr_engine.ocr(img_bytes, cls=True)

                    if result is not None and len(result) > 0 and result[0] is not None:
                        for line in result[0]:
                            if line is None:
                                continue

                            bbox = line[0]
                            text_info = line[1]

                            page_results.append({
                                'text': text_info[0],
                                'confidence': float(text_info[1]),
                                'bounding_box': bbox
                            })

                # Merge overlapping results (keep highest confidence)
                merged_results = []
                used_indices = set()

                for i, result1 in enumerate(page_results):
                    if i in used_indices:
                        continue

                    overlapping = [result1]
                    used_indices.add(i)

                    for j, result2 in enumerate(page_results):
                        if j <= i or j in used_indices:
                            continue

                        iou = calculate_iou(result1['bounding_box'], result2['bounding_box'])
                        if iou > 0.5:
                            overlapping.append(result2)
                            used_indices.add(j)

                    best_result = max(overlapping, key=lambda x: x['confidence'])
                    merged_results.append(best_result)

                # Add merged text
                for result_data in merged_results:
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


@app.post("/structure")
async def perform_structure_analysis(
    file: UploadFile = File(...),
    lang: str = None,
    multilingual: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Perform document structure analysis using PP-StructureV3

    This endpoint provides advanced document analysis including:
    - Layout detection (text regions, titles, images, tables, etc.)
    - Table recognition with cell structure
    - Formula recognition
    - Proper reading order for complex layouts
    - Markdown output with structured content

    Args:
        file: PDF file or image file
        lang: Optional language code (e.g., 'en', 'fr', 'ch'). If not specified, uses first configured language.
              For multilingual mode, use comma-separated languages (e.g., 'en,ru').
        multilingual: If True, processes document with multiple configured languages and merges results.

    Returns:
        JSON response with structured document analysis including:
        - Markdown text with proper formatting
        - Layout regions with types and bounding boxes (if available)
        - Extracted tables in structured format
        - OCR text for each region
        - Full document text in reading order
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

        # Process file to images
        images = process_file_to_images(contents, file.filename)

        all_pages_results = []
        full_document_text = []

        for page_num, image in enumerate(images, start=1):
            # Save image temporarily for PPStructureV3 (it expects file path or URL)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, format='PNG')
                tmp_path = tmp_file.name

            page_data = {
                "page": page_num,
                "markdown": "",
                "regions": []
            }

            try:
                # In multilingual mode, process with all languages and merge
                if multilingual:
                    # Collect results from all languages
                    all_lang_results = []
                    
                    for current_lang in process_langs:
                        structure_engine = get_structure(current_lang)
                        output = structure_engine.predict(input=tmp_path)

                        if output and len(output) > 0:
                            all_lang_results.append({
                                'lang': current_lang,
                                'result': output[0]
                            })

                    # For multilingual, combine markdown texts from all languages
                    # (in practice, you might want to choose the best one or merge intelligently)
                    for lang_result in all_lang_results:
                        result = lang_result['result']
                        current_lang = lang_result['lang']

                        # Extract markdown
                        if hasattr(result, 'markdown'):
                            md_info = result.markdown
                            if isinstance(md_info, dict):
                                markdown_text = md_info.get('markdown_texts', '')
                                if markdown_text:
                                    # Add language marker if multiple languages
                                    if len(process_langs) > 1:
                                        page_data["markdown"] += f"\n\n<!-- Language: {current_lang} -->\n\n{markdown_text}"
                                    else:
                                        page_data["markdown"] += markdown_text
                                    
                                    full_document_text.append(markdown_text)

                else:
                    # Single language mode
                    structure_engine = get_structure(process_langs[0])
                    output = structure_engine.predict(input=tmp_path)

                    # Extract result from output
                    if not output or len(output) == 0:
                        all_pages_results.append(page_data)
                        continue

                    result = output[0]

                    # Extract markdown - this is the main output format
                    if hasattr(result, 'markdown'):
                        md_info = result.markdown
                        if isinstance(md_info, dict):
                            markdown_text = md_info.get('markdown_texts', '')
                            markdown_images = md_info.get('markdown_images', {})
                            
                            page_data["markdown"] = markdown_text
                            
                            # Store image info if present
                            if markdown_images:
                                page_data["images"] = list(markdown_images.keys())
                            
                            full_document_text.append(markdown_text)

                    # Also extract layout parsing result if available (for additional metadata)
                    if hasattr(result, 'layout_parsing_result'):
                        layout_result = result.layout_parsing_result
                        
                        if layout_result:
                            for region in layout_result:
                                region_type = region.get('layout_label', 'unknown')
                                bbox = region.get('bbox', [])

                                region_info = {
                                    "type": region_type,
                                    "bbox": bbox
                                }

                                # Extract text content
                                if 'text' in region:
                                    region_info['text'] = region['text']

                                # Extract table HTML if available
                                if 'html' in region:
                                    region_info['table_html'] = region['html']

                                page_data["regions"].append(region_info)

            finally:
                # Clean up temp image file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            all_pages_results.append(page_data)

        return {
            "success": True,
            "filename": file.filename,
            "pages": len(images),
            "language": ','.join(process_langs),
            "multilingual": multilingual,
            "document_structure": all_pages_results,
            "full_text": "\n\n".join(full_document_text)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Structure analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
