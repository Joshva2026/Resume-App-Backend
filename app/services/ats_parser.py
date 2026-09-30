import pypdf
from docx import Document
from io import BytesIO
from typing import Dict, Any
import re

class AtsParser:
    def parse_pdf(self, file_bytes: bytes) -> str:
        pdf = pypdf.PdfReader(BytesIO(file_bytes))
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
        Extract sections using keyword matching.
        """
        lines = text.split("\n")
        
        sections = {
            "personal_info": [],
            "summary": [],
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "other": []
        }
        
        current_section = "personal_info"
        
        section_keywords = {
            "summary": ["summary", "profile", "objective", "about me"],
            "experience": ["experience", "employment", "work history", "professional experience"],
            "education": ["education", "academic background", "academic history"],
            "skills": ["skills", "technical skills", "core competencies", "technologies"],
            "projects": ["projects", "personal projects", "academic projects"],
            "certifications": ["certifications", "licenses", "courses"]
        }
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            line_lower = line_clean.lower()
            
            # Identify section headers
            is_header = False
            if len(line_clean.split()) <= 4:
                for sec, keywords in section_keywords.items():
                    if any(kw == line_lower for kw in keywords) or any(line_lower.startswith(kw) for kw in keywords):
                        current_section = sec
                        is_header = True
                        break
            
            if not is_header:
                # If we are in personal info and we exceed 15 lines, maybe default to "other" if not identified
                if current_section == "personal_info" and i > 15:
                    current_section = "other"
                sections[current_section].append(line_clean)
                
        # Join sections back to strings
        return {k: "\n".join(v) for k, v in sections.items()}
