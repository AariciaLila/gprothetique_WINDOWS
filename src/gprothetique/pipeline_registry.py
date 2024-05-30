"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from gprothetique.pipelines import data_processing as dp
from gprothetique.pipelines import data_science as ds
from gprothetique.pipelines import data_prediction as pred

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    prediction = pred.create_pipeline()

    return {
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "predict": prediction,
        "__default__": data_processing_pipeline + data_science_pipeline
    }

