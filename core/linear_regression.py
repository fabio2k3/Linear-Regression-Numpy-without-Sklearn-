import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0.0
        self.b = 0.0
        self.loss_history = []
        self.x_mean = 0.0
        self.x_std = 1.0

    def _normalize_X(self, X):
        return (X - self.x_mean) / self.x_std

    def fit(self, X, y):
        X = np.asarray(X, dtype=float).flatten()
        y = np.asarray(y, dtype=float).flatten()

        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener la misma cantidad de elementos.")

        self.x_mean = X.mean()
        self.x_std = X.std()
        if self.x_std == 0:
            self.x_std = 1.0

        Xn = self._normalize_X(X)

        m = 0.0
        b = 0.0
        n = len(Xn)

        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = m * Xn + b

            dm = (-2 / n) * np.sum(Xn * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            m -= self.learning_rate * dm
            b -= self.learning_rate * db

            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)

        self.m = m / self.x_std
        self.b = b - (m * self.x_mean / self.x_std)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float).flatten()
        return self.m * X + self.b