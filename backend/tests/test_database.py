import pytest
import os
import sys
import tempfile

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.database import (
    init_db,
    log_dataset_upload,
    get_active_dataset,
    save_prediction,
    get_prediction_history
)

def test_database_crud():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name
        
    try:
        init_db(db_path)
        
        # Test dataset logging
        ds_id = log_dataset_upload(
            filename="test.csv",
            filepath="/path/to/test.csv",
            row_count=100,
            column_count=10,
            columns=["Age", "MMSE", "Group"],
            target_column="Group",
            db_path=db_path
        )
        assert ds_id > 0
        
        active = get_active_dataset(db_path)
        assert active is not None
        assert active["filename"] == "test.csv"
        assert active["target_column"] == "Group"
        
        # Test prediction logging
        pred_id = save_prediction(
            model_name="Random Forest",
            predicted_class="1",
            probability=0.88,
            features={"Age": 75, "MMSE": 22},
            class_probabilities={"0": 0.12, "1": 0.88},
            patient_ref="TEST_PT_99",
            db_path=db_path
        )
        assert pred_id > 0
        
        history = get_prediction_history(limit=10, db_path=db_path)
        assert len(history) == 1
        assert history[0]["patient_ref"] == "TEST_PT_99"
        assert history[0]["probability"] == 0.88
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
