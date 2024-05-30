import pandas as pd
import numpy as np
from src.gprothetique.pipelines.data_processing.nodes import load_data, clean_data, normalize_data, split_data


def test_load_data(tmp_path):
    data = "before_exam_125_Hz,before_exam_250_Hz,before_exam_500_Hz,before_exam_1000_Hz,before_exam_2000_Hz,before_exam_4000_Hz,before_exam_8000_Hz,after_exam_125_Hz,after_exam_250_Hz,after_exam_500_Hz,after_exam_1000_Hz,after_exam_2000_Hz,after_exam_4000_Hz,after_exam_8000_Hz\n\
27,32,24,32,36,20,40,11,13,0,15,12,0,14"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(data)

    df = load_data(str(csv_file))
    assert not df.empty
    assert len(df) == 1


def test_clean_data():
    data = pd.DataFrame({
        'before_exam_125_Hz': ['E', '27', '3', '58'],
        'before_exam_250_Hz': [79, 32, 22, 44],
        'before_exam_500_Hz': [42.17, 24, 30, '16.61'],
        'before_exam_1000_Hz': [63, 32, 50, 60],
        'before_exam_2000_Hz': [51, 36, 45, 55],
        'before_exam_4000_Hz': [12.87, 20, 73, 57],
        'before_exam_8000_Hz': [47, 40, 67, 56],
    })
    cleaned_data = clean_data(data)
    assert cleaned_data.isnull().sum().sum() == 0
    assert cleaned_data.shape == (3, 7)  # Only three rows should remain


def test_split_data():
    data = pd.DataFrame(np.random.rand(100, 7))
    train, test = split_data(data)
    assert len(train) + len(test) == len(data)
    assert len(train) > len(test)  # 80-20 split
