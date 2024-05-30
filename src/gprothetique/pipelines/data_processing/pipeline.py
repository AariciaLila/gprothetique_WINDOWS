from kedro.pipeline import Pipeline, node
from .nodes import load_data, clean_data, normalize_data, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_data,
            inputs="params:raw_data_path",
            outputs="raw_data",
            name="load_raw_data"
        ),
        node(
            func=clean_data,
            inputs="raw_data",
            outputs="cleaned_data",
            name="clean_raw_data"
        ),
        node(
            func=normalize_data,
            inputs="cleaned_data",
            outputs=["normalized_data", "scaler"],
            name="normalize_data"
        ),
        node(
            func=split_data,
            inputs="normalized_data",
            outputs=["train_data", "test_data"],
            name="split_data"
        )
    ])
