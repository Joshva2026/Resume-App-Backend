import PyPDF2
from docx import Document
from io import BytesIO
from typing import Dict, Any
import re

class AtsParser:
    def parse_pdf(self, file_bytes: bytes) -> str:
        pdf = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def parse_docx(self, file_bytes: bytes) -> str:
        doc = Document(BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        return text

    def structure_text(self, text: str) -> Dict[str, Any]:
        """
        Naive structure extraction to support UI Preview.
        """
        lines = text.split("\n")
        pii = []
        body = []
        
        email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+")
        phone_pattern = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
        url_pattern = re.compile(r"https?://|www\.|linkedin\.com/|github\.com/")
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Assume first 10 lines contain PII if they have emails/phones/URLs or are very short
            if i < 15 and (email_pattern.search(line_clean) or phone_pattern.search(line_clean) or url_pattern.search(line_clean)):
                pii.append(line_clean)
            elif i < 3 and len(line_clean.split()) <= 5:
                # Likely Name
                pii.append(line_clean)
            else:
                body.append(line_clean)
                
        return {
            "personal_info": pii,
            "body": "\n".join(body)
        }
