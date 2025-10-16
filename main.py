import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from paddleocr import PaddleOCR
from PIL import Image
import io
from dotenv import load_dotenv

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

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Initialize PaddleOCR
ocr = None


def get_ocr():
    """Lazy initialization of PaddleOCR"""
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )
    return ocr


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify the API key from the request header"""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key


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


@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform OCR on an uploaded image file

    Args:
        file: Image file (jpg, png, etc.)

    Returns:
        JSON response with detected text and bounding boxes
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR
        ocr_engine = get_ocr()
        result = ocr_engine.ocr(contents, cls=True)

        # Format results
        if result is None or len(result) == 0 or result[0] is None:
            return {
                "success": True,
                "filename": file.filename,
                "text_blocks": [],
                "full_text": ""
            }

        text_blocks = []
        full_text_parts = []

        for line in result[0]:
            if line is None:
                continue

            bbox = line[0]  # Bounding box coordinates
            text_info = line[1]  # (text, confidence)

            text_blocks.append({
                "text": text_info[0],
                "confidence": float(text_info[1]),
                "bounding_box": bbox
            })
            full_text_parts.append(text_info[0])

        return {
            "success": True,
            "filename": file.filename,
            "text_blocks": text_blocks,
            "full_text": "\n".join(full_text_parts)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/ocr/text-only")
async def perform_ocr_text_only(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Perform OCR on an uploaded image file and return only the extracted text

    Args:
        file: Image file (jpg, png, etc.)

    Returns:
        JSON response with only the extracted text
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Perform OCR
        ocr_engine = get_ocr()
        result = ocr_engine.ocr(contents, cls=True)

        # Extract only text
        if result is None or len(result) == 0 or result[0] is None:
            return {
                "success": True,
                "filename": file.filename,
                "text": ""
            }

        text_parts = []
        for line in result[0]:
            if line is None:
                continue
            text_parts.append(line[1][0])

        return {
            "success": True,
            "filename": file.filename,
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
