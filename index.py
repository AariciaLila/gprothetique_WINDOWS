import mlflow
mlflow.set_tracking_uri("file:///home/kedro_docker/mlruns")

from flask import Flask, request, jsonify
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# Définir le chemin du projet Kedro
project_path = Path(__file__).resolve().parent

# Bootstrap the Kedro project
metadata = bootstrap_project(project_path)


# Route par défaut
@app.route("/")
def index():
    return "Bienvenue à l'API de prédiction audiométrique"


# Route pour l'entraînement (ou ré-entraînement) du modèle
@app.route("/train", methods=["GET"])
def train():
    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="__default__")
    return "Pipeline d'entraînement exécuté avec succès"


# Route pour la sauvegarde des données d'entrées
@app.route("/save_data", methods=["POST"])
def save_data():
    data = request.get_json()
    filepath = project_path / "data/01_raw/user_data.csv"
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    return "Données sauvegardées avec succès"


# Route pour obtenir des prédictions du modèle
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)

    nan_rows = df[df.isna().any(axis=1)].index
    df = df.dropna()
    message = "Format de donnees OK" if len(nan_rows.values) == 0 else f"Les lignes suivantes {nan_rows.values} contiennent des valeurs manquantes"

    df.to_csv(project_path/"data/04_prediction/data.csv", index=False)

    with KedroSession.create(project_path=project_path) as session:
        session.run(pipeline_name="predict")

    output = pd.read_csv(project_path/"data/04_prediction/data_predict.csv")
    prediction = output.to_json(orient='records')
    return jsonify({"predictions": prediction, "message": message})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
