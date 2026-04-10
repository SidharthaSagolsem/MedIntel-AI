import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Fix for Windows: Point to your installed Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_page_count(file_bytes: bytes) -> int:
    """Returns the number of pages in a single PDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        print(f"Error getting page count: {e}")
        return 0

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a single PDF. Uses PyMuPDF for digital text,
    and falls back to Tesseract OCR for scanned images.
    """
    text = ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            # If the page has very little native text, it's likely a scanned image
            if len(page_text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes()))
                page_text = pytesseract.image_to_string(img)
            
            text += page_text + "\n"
            
        doc.close()
    except Exception as e:
        print(f"Error during extraction: {e}")
        
    return text