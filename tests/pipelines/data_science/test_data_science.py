import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.gprothetique.pipelines.data_science.nodes import train_model, evaluate_model

def test_train_model():
    data = pd.DataFrame({
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
    })

    model = train_model(data)
    assert isinstance(model, LinearRegression)
