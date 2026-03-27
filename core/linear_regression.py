import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0.0
        self.b = 0.0
        self.loss_history = []

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.m * X + self.b

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de elementos.")

        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.predict(X)

            # Gradientes
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            # Actualización de parámetros
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

            # Guardar MSE
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)

        return self