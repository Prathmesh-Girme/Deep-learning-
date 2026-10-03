# DLOps Pipeline - Placement Prediction

An end-to-end deep learning pipeline that predicts whether a student will be placed, using a feedforward neural network (scikit-learn `MLPClassifier`).

## Dataset

`placement_data.csv`

| Feature | Description |
|---|---|
| Aptitude | Aptitude score |
| Coding | Coding score |
| Communication | Communication score |
| Academics | Academic score |
| Internship | Internship done (1) or not (0) |
| **Placed** | **Target: 1 = placed, 0 = not placed** |

## Pipeline steps

1. Read the data from CSV
2. Exploratory data analysis (columns, shape, statistics)
3. Preprocessing: split features and target
4. Train/test split (70/30, `random_state=42`)
5. Feature scaling with `StandardScaler`
6. FNN model training
7. Model evaluation: accuracy, confusion matrix, probabilities
8. Graphical representation
9. Save the model and scaler
10. Load the saved model and scaler
11. Predict for an unseen student

## Model

```
MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', solver='adam', max_iter=1000)
```

Input (5 features) -> Dense 8 (ReLU) -> Dense 4 (ReLU) -> Output (placed / not placed)

## Output files

| File | Description |
|---|---|
| `placement_fnn_model.pkl` | Trained model |
| `placement_fnn_scaler.pkl` | Fitted scaler (needed to scale new data) |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `loss_curve.png` | Training loss per iteration |
| `probability_distribution.png` | Predicted placement probabilities |
| `scaling_comparison.png` | Features before and after scaling |

## Run

```bash
pip install -r requirements.txt
python <your_script_name>.py
```

Close each graph window to let the script continue.

## Predict for a new student

```python
import joblib, pandas as pd

model = joblib.load("placement_fnn_model.pkl")
scaler = joblib.load("placement_fnn_scaler.pkl")

student = pd.DataFrame(
    [[70, 75, 80, 85, 1]],
    columns=["Aptitude", "Coding", "Communication", "Academics", "Internship"],
)

pred = model.predict(scaler.transform(student))
print("Placed" if pred[0] == 1 else "Not placed")
```

## Requirements

`pandas`, `numpy`, `matplotlib>=3.9`, `scikit-learn`, `joblib`
