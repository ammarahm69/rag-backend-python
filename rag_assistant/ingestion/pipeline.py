import json
import logging
from pathlib import Path
from typing import List
from .loaders import LoaderFactory
from .models import Document
from ..config.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class IngestionPipeline:
    def __init__(self, state_file: str = "ingestion_state.json"):
        self.state_file = Path(settings.PROCESSED_DATA_DIR) / state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run(self, input_dir: Path) -> List[Document]:
        documents = []
        files = [f for f in input_dir.glob("**/*") if f.suffix.lower() in settings.SUPPORTED_EXTENSIONS]

        logger.info(f"Starting ingestion of {len(files)} files")

        for file_path in files:
            try:
                loader = LoaderFactory.get_loader(file_path)
                doc = loader.load(file_path)

                rel_path = str(file_path.relative_to(input_dir))

                if self.state.get(rel_path) == doc.content_hash:
                    logger.info(f"Skipping unchanged file: {rel_path}")
                    continue

                documents.append(doc)
                self.state[rel_path] = doc.content_hash
                logger.info(f"Ingested file: {rel_path}")

            except Exception as e:
                logger.error(f"Failed to ingest file {file_path}: {str(e)}")

        self._save_state()
        return documents
