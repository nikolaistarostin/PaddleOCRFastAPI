# PaddleOCR FastAPI Wrapper

A simple FastAPI wrapper for PaddleOCR with API key authentication.

## Features

- RESTful API for OCR operations
- API key authentication for secure access
- Multiple endpoints for different use cases
- Support for various image formats
- Detailed OCR results with bounding boxes and confidence scores
- Text-only endpoint for simplified output
- Docker and Docker Compose support for easy deployment

## Installation

### Option 1: Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd PaddleOCRFastAPI
```

2. Create a `.env` file from the example:
```bash
cp .env.example .env
```

3. Edit `.env` and set your API key:
```
API_KEY=your-secure-api-key-here
```

4. Start the service with Docker Compose:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:8023`

To stop the service:
```bash
docker-compose down
```

To view logs:
```bash
docker-compose logs -f
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PaddleOCRFastAPI
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the example:
```bash
cp .env.example .env
```

5. Edit `.env` and set your API key:
```
API_KEY=your-secure-api-key-here
```

## Usage

### Starting the Server

Run the server with:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8023 --reload
```

The API will be available at `http://localhost:8023`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8023/docs`
- ReDoc: `http://localhost:8023/redoc`

## API Endpoints

### Health Check
```bash
GET /
GET /health
GET /languages
```
No authentication required.

### Document Structure Analysis (PP-StructureV3)
```bash
POST /structure
```
**NEW!** Advanced document analysis with layout detection, table recognition, and structured extraction.

Perfect for:
- PDFs with tables and complex layouts
- Multi-column documents
- Forms and invoices
- Research papers
- Multilingual documents

**Headers:**
- `X-API-Key`: Your API key

**Parameters:**
- `file`: PDF or image file (multipart/form-data)
- `lang` (optional): Language code (e.g., 'en', 'ru', 'ch'). If not specified, uses first configured language.
- `multilingual` (optional): Boolean flag to process with multiple languages (default: false)

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "pages": 2,
  "language": "en",
  "multilingual": false,
  "document_structure": [
    {
      "page": 1,
      "markdown": "# Document Title\n\nParagraph content...\n\n| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |",
      "images": ["imgs/img_in_image_box_123_456_789_012.jpg"],
      "regions": [
        {
          "type": "title",
          "bbox": [x1, y1, x2, y2],
          "text": "Document Title"
        },
        {
          "type": "table",
          "bbox": [x1, y1, x2, y2],
          "table_html": "<table>...</table>",
          "text": "Table content as text"
        },
        {
          "type": "text",
          "bbox": [x1, y1, x2, y2],
          "text": "Paragraph content"
        }
      ]
    }
  ],
  "full_text": "Document Title\n\nTable content...\n\nParagraph content"
}
```

**Key Features:**
- **Markdown Output:** Primary output format with properly formatted text, tables, and images
- **Layout Regions:** Structured metadata about document regions (titles, text, tables, figures, etc.)
- **Table Recognition:** Tables extracted as HTML with proper structure
- **Multilingual Support:** Process documents in multiple languages
- **Image Extraction:** List of extracted images from the document

**Region Types:**
- `title` - Document/section titles
- `text` - Paragraph text
- `figure` - Images and captions
- `table` - Tables (includes HTML structure)
- `list` - Lists
- And more...

**Multilingual Structure Analysis:**

For multilingual documents, use the `multilingual=true` parameter:

```bash
# Process with specific languages
curl -X POST "http://localhost:8023/structure?lang=en,ru&multilingual=true" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"

# Process with all configured languages
curl -X POST "http://localhost:8023/structure?multilingual=true" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"
```

### OCR with Full Details
```bash
POST /ocr
```
Returns detected text with bounding boxes and confidence scores.

**Headers:**
- `X-API-Key`: Your API key

**Body:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "filename": "example.jpg",
  "text_blocks": [
    {
      "text": "Hello World",
      "confidence": 0.98,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "full_text": "Hello World"
}
```

### OCR Text Only
```bash
POST /ocr/text-only
```
Returns only the extracted text without bounding boxes.

**Headers:**
- `X-API-Key`: Your API key

**Body:**
- `file`: Image file (multipart/form-data)

**Response:**
```json
{
  "success": true,
  "filename": "example.jpg",
  "text": "Hello World"
}
```

## Example Usage

### Using cURL

```bash
# Document structure analysis (recommended for PDFs with tables)
curl -X POST "http://localhost:8023/structure" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"

# Document structure analysis with specific language
curl -X POST "http://localhost:8023/structure?lang=ru" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"

# Document structure analysis with multilingual support
curl -X POST "http://localhost:8023/structure?lang=en,ru&multilingual=true" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"

# Full OCR with details
curl -X POST "http://localhost:8023/ocr" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@/path/to/image.jpg"

# Text only
curl -X POST "http://localhost:8023/ocr/text-only" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@/path/to/image.jpg"
```

### Using Python

```python
import requests

# Document structure analysis
url = "http://localhost:8023/structure"
headers = {"X-API-Key": "your-api-key-here"}
files = {"file": open("document.pdf", "rb")}

response = requests.post(url, headers=headers, files=files)
result = response.json()

# Access markdown output
for page in result['document_structure']:
    print(f"Page {page['page']}:")
    print(page['markdown'])

# OCR with full details
url = "http://localhost:8023/ocr"
files = {"file": open("image.jpg", "rb")}

response = requests.post(url, headers=headers, files=files)
print(response.json())

# Multilingual structure analysis
url = "http://localhost:8023/structure"
params = {"lang": "en,ru", "multilingual": "true"}
files = {"file": open("document.pdf", "rb")}

response = requests.post(url, headers=headers, files=files, params=params)
print(response.json())
```

### Using JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8023/ocr', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key-here'
  },
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Configuration

### Environment Variables

- `API_KEY` (required): API key for authentication
- `OCR_LANGUAGES` (optional): Comma-separated list of language codes. Default: `en`

### Language Configuration

PaddleOCR supports 80+ languages. You can configure multiple languages via the `OCR_LANGUAGES` environment variable.

**Examples:**
```bash
# Single language (English)
OCR_LANGUAGES=en

# Multiple languages (English, French, German)
OCR_LANGUAGES=en,fr,german

# Chinese + English
OCR_LANGUAGES=ch,en

# Spanish, Portuguese, Italian
OCR_LANGUAGES=es,pt,it
```

**Supported Languages:**
- `en` - English
- `ch` - Chinese (Simplified & Traditional)
- `fr` - French
- `german` - German
- `japan` - Japanese
- `korean` - Korean
- `es` - Spanish
- `pt` - Portuguese
- `ru` - Russian
- `ar` - Arabic
- `hi` - Hindi
- `it` - Italian
- And 70+ more...

Full list: [PaddleOCR Multilingual Documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md)

**Using Language Parameter:**

You can specify which language to use per request:

```bash
# Use default language
curl -X POST "http://localhost:8023/ocr" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@image.jpg"

# Specify language
curl -X POST "http://localhost:8023/ocr?lang=fr" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@image.jpg"

# Multilingual mode (for documents with mixed languages like Russian + English)
curl -X POST "http://localhost:8023/ocr?lang=en,ru&multilingual=true" \
  -H "X-API-Key: your-api-key-here" \
  -F "file=@document.pdf"
```

**Multilingual OCR:**

For documents containing multiple languages (e.g., Russian and English text on the same page), use the `multilingual=true` parameter:

```bash
# Process with specific languages
POST /ocr?lang=en,ru&multilingual=true

# Process with all configured languages
POST /ocr?multilingual=true
```

The multilingual mode:
- Processes the document with each specified language
- Merges results by bounding box location
- Keeps the result with highest confidence for each text region
- Includes `detected_lang` field showing which language produced the best result

Example response:
```json
{
  "success": true,
  "filename": "mixed-lang.pdf",
  "pages": 1,
  "language": "en,ru",
  "multilingual": true,
  "text_blocks": [
    {
      "text": "Hello World",
      "confidence": 0.98,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "page": 1,
      "detected_lang": "en"
    },
    {
      "text": "Привет мир",
      "confidence": 0.96,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "page": 1,
      "detected_lang": "ru"
    }
  ],
  "full_text": "Hello World\nПривет мир"
}
```

**Check Available Languages:**

```bash
curl http://localhost:8023/languages
```

Response:
```json
{
  "configured_languages": ["en", "fr", "german"],
  "default_language": "en"
}
```

### PaddleOCR Settings

PaddleOCR is configured in [main.py](main.py) with the following default settings:
- Languages: Configured via `OCR_LANGUAGES` environment variable
- Angle classification: Enabled (`use_angle_cls=True`)
- GPU: Disabled (`use_gpu=False`)

You can modify these settings in the `get_ocr()` function.

## Security Notes

- Always use a strong, randomly generated API key in production
- Keep your `.env` file secure and never commit it to version control
- Consider using HTTPS in production
- Implement rate limiting for production deployments

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (e.g., invalid file type)
- `403`: Invalid API key
- `500`: Server error during OCR processing

## Requirements

- Python 3.8+
- FastAPI
- PaddleOCR
- PaddlePaddle
- Pillow
- python-dotenv

See [requirements.txt](requirements.txt) for full dependencies.

### Docker Requirements

- Docker 20.10+
- Docker Compose 1.29+

## Docker Details

### Dockerfile

The [Dockerfile](Dockerfile) uses Python 3.10-slim as the base image and includes:
- System dependencies required for PaddleOCR (OpenGL, libgomp, etc.)
- Automatic model caching in `/root/.paddleocr`
- Health check endpoint integration
- Optimized layer caching for faster rebuilds

### Docker Compose

The [docker-compose.yml](docker-compose.yml) file provides:
- Automatic container restart policy
- Environment variable injection from `.env` file
- Named volume for persistent model storage
- Port mapping (8023:8023)
- Integrated health checks

### Building and Running

Build the image:
```bash
docker-compose build
```

Run in detached mode:
```bash
docker-compose up -d
```

Run with live logs:
```bash
docker-compose up
```

Rebuild and restart:
```bash
docker-compose up -d --build
```

### Persistent Storage

PaddleOCR models are downloaded on first use and stored in a Docker volume named `paddleocr-models`. This ensures:
- Models persist between container restarts
- Faster startup times after initial download
- Reduced bandwidth usage

To remove the volume (will require re-downloading models):
```bash
docker-compose down -v
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
