import os
import matplotlib.pyplot as plt

from core.linear_regression import LinearRegressionGD
from core.metrics import mse, r2_score
from utils.data_loader import get_columns, load_selected_columns


def ask_column(headers, prompt_text):
    print("\nColumnas disponibles:")
    for i, col in enumerate(headers):
        print(f"{i}: {col}")

    while True:
        try:
            idx = int(input(prompt_text))
            if 0 <= idx < len(headers):
                return headers[idx]
            print("Índice fuera de rango.")
        except ValueError:
            print("Ingresa un número válido.")


def main():
    csv_path = input("Ruta del CSV (ej: data/dataset.csv): ").strip()

    if not os.path.exists(csv_path):
        print("El archivo no existe.")
        return

    headers = get_columns(csv_path)

    x_col = ask_column(headers, "\nSelecciona la columna X (número): ")
    y_col = ask_column(headers, "Selecciona la columna Y (número): ")

    X, y = load_selected_columns(csv_path, x_col, y_col)

    lr = input("\nLearning rate [0.01]: ").strip()
    epochs = input("Epochs [1000]: ").strip()

    learning_rate = float(lr) if lr else 0.01
    epochs = int(epochs) if epochs else 1000

    model = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs)
    model.fit(X, y)

    y_pred = model.predict(X)

    print("\n=== RESULTADOS ===")
    print(f"Columna X: {x_col}")
    print(f"Columna Y: {y_col}")
    print(f"m = {model.m:.6f}")
    print(f"b = {model.b:.6f}")
    print(f"MSE = {mse(y, y_pred):.6f}")
    print(f"R² = {r2_score(y, y_pred):.6f}")

    # Recta ordenada para la gráfica
    x_line = sorted(X)
    y_line = model.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Datos")
    plt.plot(x_line, y_line, label="Regresión lineal")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Regresión Lineal Simple")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Convergencia del descenso gradiente")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()