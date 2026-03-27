import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.linear_regression import LinearRegressionGD
from core.metrics import mse, r2_score
from utils.data_loader import get_columns, load_selected_columns


class DropArea(QFrame):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            QFrame {
                border: 2px dashed #888;
                border-radius: 12px;
                background: #fafafa;
            }
            """
        )

        layout = QVBoxLayout(self)
        self.label = QLabel("Arrastra y suelta aquí tu archivo CSV")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 16px; padding: 20px;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".csv"):
                event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        file_path = urls[0].toLocalFile()
        if file_path.lower().endswith(".csv"):
            self.fileDropped.emit(file_path)
        else:
            QMessageBox.warning(self, "Archivo inválido", "Solo se permiten archivos CSV.")


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regresión Lineal con Numpy")
        self.resize(1100, 700)

        self.csv_path = None
        self.columns = []

        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QGridLayout(root)

        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.load_csv)

        self.file_label = QLabel("Ningún archivo cargado")
        self.file_label.setWordWrap(True)

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.x_combo.setEnabled(False)
        self.y_combo.setEnabled(False)

        self.lr_input = QLineEdit("0.01")
        self.epochs_input = QLineEdit("1000")

        form = QFormLayout()
        form.addRow("Columna X:", self.x_combo)
        form.addRow("Columna Y:", self.y_combo)
        form.addRow("Learning rate:", self.lr_input)
        form.addRow("Epochs:", self.epochs_input)

        self.train_btn = QPushButton("Entrenar")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.train_model)

        self.m_label = QLabel("m: -")
        self.b_label = QLabel("b: -")
        self.mse_label = QLabel("MSE: -")
        self.r2_label = QLabel("R²: -")

        metrics_box = QVBoxLayout()
        metrics_box.addWidget(self.m_label)
        metrics_box.addWidget(self.b_label)
        metrics_box.addWidget(self.mse_label)
        metrics_box.addWidget(self.r2_label)

        self.canvas = MplCanvas()

        left_panel.addWidget(self.drop_area)
        left_panel.addWidget(self.file_label)
        left_panel.addLayout(form)
        left_panel.addWidget(self.train_btn)
        left_panel.addSpacing(10)
        left_panel.addWidget(QLabel("Resultados"))
        left_panel.addLayout(metrics_box)
        left_panel.addStretch()

        right_panel.addWidget(self.canvas)

        main_layout.addLayout(left_panel, 0, 0)
        main_layout.addLayout(right_panel, 0, 1)

        self.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                padding: 8px 12px;
                font-size: 14px;
            }
            QComboBox, QLineEdit {
                padding: 6px;
                font-size: 14px;
            }
            """
        )

    def load_csv(self, file_path):
        try:
            self.csv_path = file_path
            self.columns = get_columns(file_path)

            self.x_combo.clear()
            self.y_combo.clear()
            self.x_combo.addItems(self.columns)
            self.y_combo.addItems(self.columns)

            self.x_combo.setEnabled(True)
            self.y_combo.setEnabled(True)
            self.train_btn.setEnabled(True)

            self.file_label.setText(f"Archivo cargado: {os.path.basename(file_path)}")
            QMessageBox.information(self, "CSV cargado", "Columnas detectadas correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el CSV:\n{e}")

    def train_model(self):
        if not self.csv_path:
            QMessageBox.warning(self, "Sin archivo", "Primero carga un CSV.")
            return

        x_col = self.x_combo.currentText()
        y_col = self.y_combo.currentText()

        if not x_col or not y_col:
            QMessageBox.warning(self, "Columnas faltantes", "Selecciona X e Y.")
            return

        if x_col == y_col:
            QMessageBox.warning(self, "Selección inválida", "X e Y no deben ser la misma columna.")
            return

        try:
            learning_rate = float(self.lr_input.text().strip())
            epochs = int(self.epochs_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Parámetros inválidos", "Learning rate y epochs deben ser numéricos.")
            return

        try:
            X, y = load_selected_columns(self.csv_path, x_col, y_col)

            model = LinearRegressionGD(learning_rate=learning_rate, epochs=epochs)
            model.fit(X, y)

            y_pred = model.predict(X)

            current_mse = mse(y, y_pred)
            current_r2 = r2_score(y, y_pred)

            self.m_label.setText(f"m: {model.m:.6f}")
            self.b_label.setText(f"b: {model.b:.6f}")
            self.mse_label.setText(f"MSE: {current_mse:.6f}")
            self.r2_label.setText(f"R²: {current_r2:.6f}")

            self.plot_results(X, y, model, x_col, y_col)

        except Exception as e:
            QMessageBox.critical(self, "Error al entrenar", str(e))

    def plot_results(self, X, y, model, x_col, y_col):
        self.canvas.ax.clear()

        sorted_idx = np.argsort(X)
        X_sorted = X[sorted_idx]
        y_line = model.predict(X_sorted)

        self.canvas.ax.scatter(X, y, label="Datos")
        self.canvas.ax.plot(X_sorted, y_line, color="red", linewidth=2, label="Regresión lineal")
        self.canvas.ax.set_xlabel(x_col)
        self.canvas.ax.set_ylabel(y_col)
        self.canvas.ax.set_title("Regresión Lineal Simple")
        self.canvas.ax.grid(True)
        self.canvas.ax.legend()

        self.canvas.draw()