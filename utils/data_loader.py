import csv
import numpy as np


def get_columns(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers


def load_selected_columns(csv_path, x_col, y_col):
    x_values = []
    y_values = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                x_values.append(float(row[x_col]))
                y_values.append(float(row[y_col]))
            except (ValueError, KeyError):
                continue

    X = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    if len(X) == 0:
        raise ValueError("No se pudieron cargar datos numéricos de esas columnas.")

    return X, y