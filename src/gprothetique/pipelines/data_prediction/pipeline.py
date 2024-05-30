"""
This is a boilerplate pipeline 'data_prediction'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prediction,
            inputs=["params:run_id", "params:path_scaler", "data"],
            outputs="data_predict",
            name="prediction_node",
        ),
    ])
