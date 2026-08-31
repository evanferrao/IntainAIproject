"""Data preparation pipeline: adapter transformation, monthly panel generation, and servicer updates."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    EXTERNAL_DATA_DIR, RAW_DATA_DIR, CANONICAL_DATA_DIR,
    DEFAULT_CONFIG, RANDOM_SEED
)
from src.adapters.lending_club_adapter import LendingClubAdapter
from src.ingestion.monthly_panel_generator import MonthlyPanelGenerator
from src.ingestion.servicer_simulator import ServicerUpdateSimulator
from src.features.time_splitter import TimeAwareSplitter
from src.utils.logger import logger


def prepare_all_data():
    logger.info("================ STARTING DATA PREPARATION ================")
    
    # 1. Locate raw files
    raw_files = list(EXTERNAL_DATA_DIR.glob("*.csv")) + list(EXTERNAL_DATA_DIR.glob("*.csv.gz")) + list(RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        logger.warning(f"No raw files found in {EXTERNAL_DATA_DIR}. Triggering download/generator...")
        from src.ingestion.download import download_lending_club_data
        download_lending_club_data()
        raw_files = list(EXTERNAL_DATA_DIR.glob("*.csv")) + list(EXTERNAL_DATA_DIR.glob("*.csv.gz"))

    target_file = raw_files[0]
    logger.info(f"Loading raw source file: {target_file}")
    
    # Read sample (up to configured max records for speed & memory safety)
    max_records = DEFAULT_CONFIG.max_sample_records
    raw_df = pd.read_csv(target_file, nrows=max_records, low_memory=False)
    logger.info(f"Loaded raw data: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns.")

    # 2. Transform via Adapter
    adapter = LendingClubAdapter(seed=RANDOM_SEED)
    static_df = adapter.transform(raw_df)
    
    static_out = CANONICAL_DATA_DIR / "loan_static_attributes.csv"
    static_df.to_csv(static_out, index=False)
    logger.info(f"Saved canonical static attributes to {static_out}")

    # 3. Generate Monthly Panel
    panel_gen = MonthlyPanelGenerator(seed=RANDOM_SEED)
    # Generate panel for subset of loans to keep performance optimal
    panel_df = panel_gen.generate_monthly_panel(
        static_df,
        max_horizon_months=DEFAULT_CONFIG.monthly_panel_months,
        max_loans=min(10_000, len(static_df))
    )

    # 4. Chronological Train/Test Split
    splitter = TimeAwareSplitter(test_months=DEFAULT_CONFIG.test_size_months)
    train_panel, test_panel, split_meta = splitter.split(panel_df)

    train_out = CANONICAL_DATA_DIR / "loan_monthly_performance_train.csv"
    test_out = CANONICAL_DATA_DIR / "loan_monthly_performance_test.csv"
    train_panel.to_csv(train_out, index=False)
    test_panel.to_csv(test_out, index=False)
    logger.info(f"Saved train monthly panel ({len(train_panel):,} rows) to {train_out}")
    logger.info(f"Saved test monthly panel ({len(test_panel):,} rows) to {test_out}")

    # 5. Generate Servicer Updates
    servicer_sim = ServicerUpdateSimulator(seed=RANDOM_SEED)
    servicer_df = servicer_sim.generate_servicer_updates(static_df, panel_df)
    servicer_out = CANONICAL_DATA_DIR / "servicer_updates.csv"
    servicer_df.to_csv(servicer_out, index=False)
    logger.info(f"Saved servicer updates ({len(servicer_df):,} rows) to {servicer_out}")

    logger.info("================ DATA PREPARATION COMPLETE ================")


if __name__ == "__main__":
    prepare_all_data()
