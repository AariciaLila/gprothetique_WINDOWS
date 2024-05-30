"""
This is a boilerplate pipeline 'data_prediction'
generated using Kedro 0.19.6
"""
import pandas as pd
import mlflow


def prediction(run_id, path_scaler, data: pd.DataFrame) -> pd.DataFrame:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    import pickle

    # Charger le modèle
    with open(path_scaler, 'rb') as file:
        scaler = pickle.load(file)


    # zéro paddin


    data = zero_padding_after(data)
    data_scaler = scaler.transform(data)

    after_cols = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]
    data_scaler = data_scaler[:, :7]
    predict_data = pd.DataFrame(model.predict(data_scaler), columns=after_cols)

    predict_data = zero_padding_before(predict_data)
    predict_data_unscaler = scaler.inverse_transform(predict_data)
    result = predict_data_unscaler[:, :7]
    df_prediction = pd.DataFrame(result, columns=after_cols)
    return df_prediction


def zero_padding_after(df):
    # Ajouter les colonnes 'after_exam_*_Hz' avec des zéros
    columns_to_add = ['after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
                      'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz',
                      'after_exam_8000_Hz']

    for column in columns_to_add:
        df[column] = 0

    return df


def zero_padding_before(df):
    # Ajouter les colonnes 'after_exam_*_Hz' avec des zéros
    columns_to_add = ['before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
                      'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz',
                      'before_exam_8000_Hz']

    for column in columns_to_add:
        df[column] = 0

    return df




