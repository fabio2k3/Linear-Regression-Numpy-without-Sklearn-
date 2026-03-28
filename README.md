# 📈 Linear Regression from Scratch (NumPy + GUI)

This project implements a **simple linear regression model from scratch** using only **NumPy**, without relying on machine learning libraries such as scikit-learn. It also includes a **graphical user interface (GUI)** built with PySide6, allowing users to interactively load datasets, select variables, and visualize results.

---

## 🚀 Features

- Linear regression implemented using **Gradient Descent**
- No use of `sklearn` or external ML libraries
- CSV dataset support (drag & drop or file browser)
- Interactive selection of:
  - Independent variable (X)
  - Dependent variable (Y)
- Real-time visualization using **Matplotlib**
- Displays key metrics:
  - Slope (`m`)
  - Intercept (`b`)
  - Mean Squared Error (MSE)
  - R² Score
- Embedded plot inside the GUI
- Clean and modular architecture

---

## 🧠 How It Works

The model learns the relationship between two variables using **Gradient Descent**, iteratively updating its parameters to minimize error.

**Linear model:**

```
y = mX + b
```

The goal is to minimize the **Mean Squared Error (MSE)** between predictions and real values:

```
MSE = (1/n) * Σ(y_i - ŷ_i)²
```

Parameters `m` (slope) and `b` (intercept) are updated at each step:

```
m = m - α * ∂MSE/∂m
b = b - α * ∂MSE/∂b
```

where `α` is the learning rate.

---

## 🗂️ Project Structure

```
project/
├── core/
│   ├── linear_regression.py   # Gradient Descent implementation
│   └── metrics.py             # MSE and R² calculations
│
├── utils/
│   └── data_loader.py         # CSV handling and column extraction
│
├── ui/
│   └── app.py                 # PySide6 graphical interface
│
├── data/
│   └── dataset.csv            # Sample dataset (optional)
│
├── main.py                    # Application entry point
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 🖥️ Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
python main.py
```

### 3. Use the GUI

1. **Load a dataset** — drag & drop a CSV file or click to browse
2. **Select columns:**
   - X column (independent variable)
   - Y column (dependent variable)
3. **Adjust parameters:**
   - Learning rate
   - Number of epochs
4. **Click Train**
5. **View results:**
   - Regression line plotted over the data
   - Metrics: slope, intercept, MSE, R²

---

## 📦 Requirements

| Package      | Purpose                        |
|--------------|--------------------------------|
| `numpy`      | Numerical computation          |
| `pandas`     | CSV loading and data handling  |
| `matplotlib` | Data and regression plotting   |
| `PySide6`    | Graphical user interface       |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Fabio Víctor Alonso Bañobre**
