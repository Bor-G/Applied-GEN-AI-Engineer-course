"""Configuration management for the CV analysis application."""
import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

# PostgreSQL configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cv_analysis")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Application settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
