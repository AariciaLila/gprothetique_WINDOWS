import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Gérer les valeurs manquantes et les conversions de type
    data = data.replace(r'[^\d.]+', np.nan, regex=True)  # Remplacer les non-numériques par NaN
    data = data.dropna()  # Supprimer les lignes avec NaN
    return data.astype(float)  # Convertir toutes les données en float


def normalize_data(data: pd.DataFrame) -> Tuple:
    scaler = StandardScaler()
    data_scaler = scaler.fit_transform(data)
    return pd.DataFrame(data_scaler, columns=data.columns), scaler


def split_data(data: pd.DataFrame) -> tuple:
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test
