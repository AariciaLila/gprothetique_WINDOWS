import os
import pandas as pd
import numpy as np
from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner
from src.gprothetique.pipelines.data_processing import create_pipeline as create_data_processing_pipeline
from src.gprothetique.pipelines.data_science import create_pipeline as create_data_science_pipeline

def test_data_processing_pipeline(tmp_path):
    # Créer un fichier CSV de test temporaire
    data = {
        "before_exam_125_Hz": [79, 32, 22, 44],
        "before_exam_250_Hz": [42.17, 24, 30, 16.61],
        "before_exam_500_Hz": [63, 32, 50, 60],
        "before_exam_1000_Hz": [51, 36, 45, 55],
        "before_exam_2000_Hz": [12.87, 20, 73, 57],
        "before_exam_4000_Hz": [47, 40, 67, 56],
        "before_exam_8000_Hz": [65, 79, 62, 57],
        "after_exam_125_Hz": [42, 63, 38, 58],
        "after_exam_250_Hz": [27, 32, 24, 32],
        "after_exam_500_Hz": [36, 20, 40, 11],
        "after_exam_1000_Hz": [13, 0, 15, 12],
        "after_exam_2000_Hz": [0, 14, 0, 10],
        "after_exam_4000_Hz": [15, 12, 0, 14],
        "after_exam_8000_Hz": [3, 22, 30, 50]
    }
    df = pd.DataFrame(data)
    test_file_path = tmp_path / "test_tonal_exams.csv"
    df.to_csv(test_file_path, index=False)

    # Créer le catalogue de données
    data_catalog = DataCatalog({
        "raw_data": MemoryDataset(),
        "cleaned_data": MemoryDataset(),
        "normalized_data": MemoryDataset(),
        "train_data": MemoryDataset(),
        "test_data": MemoryDataset(),
        "params:raw_data_path": MemoryDataset(test_file_path.as_posix())
    })

    # Exécuter le pipeline
    pipeline = create_data_processing_pipeline()
    runner = SequentialRunner()
    runner.run(pipeline, data_catalog)

def test_data_science_pipeline():
    pipeline = create_data_science_pipeline()
    data_catalog = DataCatalog({
        "train_data": MemoryDataset(pd.DataFrame({
            'after_exam_125_Hz': np.random.rand(100),
            'after_exam_250_Hz': np.random.rand(100),
            'after_exam_500_Hz': np.random.rand(100),
            'after_exam_1000_Hz': np.random.rand(100),
            'after_exam_2000_Hz': np.random.rand(100),
            'after_exam_4000_Hz': np.random.rand(100),
            'after_exam_8000_Hz': np.random.rand(100),
            'before_exam_125_Hz': np.random.rand(100),
            'before_exam_250_Hz': np.random.rand(100),
            'before_exam_500_Hz': np.random.rand(100),
            'before_exam_1000_Hz': np.random.rand(100),
            'before_exam_2000_Hz': np.random.rand(100),
            'before_exam_4000_Hz': np.random.rand(100),
            'before_exam_8000_Hz': np.random.rand(100)
        })),
        "test_data": MemoryDataset(pd.DataFrame({
            'after_exam_125_Hz': np.random.rand(20),
            'after_exam_250_Hz': np.random.rand(20),
            'after_exam_500_Hz': np.random.rand(20),
            'after_exam_1000_Hz': np.random.rand(20),
            'after_exam_2000_Hz': np.random.rand(20),
            'after_exam_4000_Hz': np.random.rand(20),
            'after_exam_8000_Hz': np.random.rand(20),
            'before_exam_125_Hz': np.random.rand(20),
            'before_exam_250_Hz': np.random.rand(20),
            'before_exam_500_Hz': np.random.rand(20),
            'before_exam_1000_Hz': np.random.rand(20),
            'before_exam_2000_Hz': np.random.rand(20),
            'before_exam_4000_Hz': np.random.rand(20),
            'before_exam_8000_Hz': np.random.rand(20)
        })),
        "regressor": MemoryDataset()
    })

    runner = SequentialRunner()
    runner.run(pipeline, data_catalog)
