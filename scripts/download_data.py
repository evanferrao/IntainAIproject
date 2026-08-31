"""Script to acquire Lending Club dataset or generate schema-compliant fallback prototype."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.download import download_lending_club_data
from src.utils.logger import logger


def main():
    logger.info("Starting dataset acquisition...")
    download_dir = download_lending_club_data()
    logger.info(f"Dataset acquisition completed. Files stored in: {download_dir}")


if __name__ == "__main__":
    main()
