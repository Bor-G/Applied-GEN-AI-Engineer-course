"""Document processing utilities for CV analysis."""
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
from typing import Dict, Any


class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, llm=None):
        """Initialize document processor with chunking parameters and LLM."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def generate_summary_with_llm(self, cv_text: str) -> str:
        """Generate CV summary using LLM."""
        if not self.llm:
            return self._extract_summary(cv_text)

        system_prompt = """You are a professional CV analyzer. Create a concise summary of the candidate's profile 
        based on their CV. Focus on their current role, key achievements, and main areas of expertise. 
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please create a professional summary based on this CV:\n\n{cv_text}"}
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating summary with LLM: {e}")
            return self._extract_summary(cv_text)

    def extract_skills_with_llm(self, cv_text: str) -> list:
        """Extract technical skills using LLM."""
        if not self.llm:
            return self._extract_skills(cv_text)

        system_prompt = """You are a professional CV analyzer. 
        Extract the key technical and professional skills from the CV.
        Return ONLY a JSON-formatted array of skills, limited to the top 10-15 most relevant skills.
        Example output: ["Python", "Machine Learning", "Project Management", "AWS", "Agile"]
        Focus on concrete, specific skills rather than general qualities."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract key skills from this CV:\n\n{cv_text}"}
        ]

        try:
            response = self.llm.invoke(messages)
            # Clean and parse the response
            skills_text = response.content.strip()
            # Remove any redundant formatting
            skills_text = skills_text.replace("```json", "").replace("```", "")
            # Convert string representation of list to actual list
            import ast
            skills = ast.literal_eval(skills_text)
            return skills[:10]  # Limit to top 10 skills
        except Exception as e:
            print(f"Error extracting skills with LLM: {e}")
            return self._extract_skills(cv_text)

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    def split_text(self, text: str) -> list:
        """Split text into chunks for processing."""
        return self.text_splitter.split_text(text)

    def extract_candidate_info(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract candidate information from CV text."""
        identifier = os.path.splitext(filename)[0]
        current_position = self._extract_current_position(text)
        experience_years = self._calculate_experience_years(text)
        skills = self.extract_skills_with_llm(text)  # Using LLM-based skill extraction
        summary = self.generate_summary_with_llm(text)

        return {
            "identifier": identifier,
            "current_position": current_position,
            "experience_years": experience_years,
            "key_skills": skills,
            "summary": summary
        }

    def _extract_current_position(self, text: str) -> str:
        """Extract details about the current/most recent role using LLM."""
        if self.llm:
            try:
                system_prompt = """You are a professional CV analyzer. Extract the current or most recent job 
                title/role from the CV. Return only the job title, without any additional text. If you can't 
                find a clear current role, return 'Not specified'.
                Example output formats:
                - Senior Software Engineer
                - IT Systems Administrator
                - Project Manager
                - DevOps Engineer"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract the current role from this CV:\n\n{text}"}
                ]

                response = self.llm.invoke(messages)
                if response.content and response.content.strip():
                    return response.content.strip()
            except Exception as e:
                print(f"Error extracting role with LLM: {e}")

        return "Role not specified"

    def _calculate_experience_years(self, text: str) -> int:
        """Calculate total years of experience using LLM."""
        if self.llm:
            try:
                system_prompt = """You are a professional CV analyzer. 
                Calculate the total years of professional experience from the CV.
                Return ONLY a number representing the total years of experience. Round to the nearest whole number.
                If you can't determine the exact number, estimate based on the work history.
                If you can't find any work history, return 0."""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Calculate the total years of experience from this CV:\n\n{text}"}
                ]

                response = self.llm.invoke(messages)
                try:
                    years = int(float(response.content.strip()))
                    return max(0, years)
                except (ValueError, TypeError):
                    print(f"Could not parse LLM response as integer: {response.content}")
                    return 0
            except Exception as e:
                print(f"Error calculating experience years with LLM: {e}")
                return 0

        return 0

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extract professional summary or profile (fallback method)."""
        summary_patterns = [
            r"(?i)(?:professional\s+)?summary:?(.*?)(?:\n\n|\Z)",
            r"(?i)(?:professional\s+)?profile:?(.*?)(?:\n\n|\Z)",
            r"(?i)about:?(.*?)(?:\n\n|\Z)"
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                return ' '.join(summary.split())[:300]

        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 100:
                return ' '.join(para.split())[:300]

        return "No summary available"

    @staticmethod
    def _extract_skills(text: str) -> list:
        """Extract technical skills from the CV (fallback method)."""
        skills_section_pattern = r"(?i)(?:skills|technical skills|core competencies):?(.*?)(?:\n\n|\Z)"
        match = re.search(skills_section_pattern, text, re.DOTALL)

        if match:
            skills_text = match.group(1)
            skills = re.split(r'[,•|]', skills_text)
            return [skill.strip() for skill in skills if skill.strip()][:10]
        return []
