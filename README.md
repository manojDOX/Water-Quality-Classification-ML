# Water Quality Classification Using Machine Learning

## Problem Statement

Water quality assessment is critical for determining the suitability of water for drinking, irrigation, industrial use, and ecological sustainability. Manual analysis of water quality parameters is time-consuming and prone to inconsistency.
This project aims to **automatically classify water sources into multiple use-based classes** using physicochemical, nutrient, and microbiological parameters with machine learning models.

The target variable is **Use Based Class**, which represents predefined water usage categories.

---

## Dataset

The dataset consists of water quality measurements collected from various water bodies such as rivers, lakes, dams, creeks, and seas.

### Key Characteristics

* **Total records:** ~500+ after cleaning
* **Total columns:** 55 (reduced to a selected subset for modeling)
* **Target variable:** `Use Based Class` (Multi-class classification: A, B, C, E)

### Feature Types

* **Physicochemical:** pH, Temperature, Turbidity, Conductivity, TDS
* **Nutrients:** Nitrate N, Ammonia N, Total Kjeldahl N, Phosphate
* **Organic pollution:** BOD, COD
* **Microbiological:** Total Coliform, Fecal Coliform
* **Contextual:** Type Water Body

Extensive data cleaning was performed, including:

* Handling missing values using skewness-based imputation
* Removing invalid string entries
* Converting object columns to numeric where applicable
* Addressing class imbalance

---

## Key Findings (EDA)

* The target variable **Use Based Class is imbalanced**, with Class A being dominant.
* **BOD, COD, Dissolved Oxygen, and microbial indicators** strongly influence water quality classification.
* Several features show **high correlation** (e.g., Conductivity and TDS), and redundant features were removed.
* Seasonal and water body type variations impact certain parameters such as Dissolved Oxygen.
* Tree-based models capture non-linear relationships better than linear models.

---

## Model Performance

Multiple models were trained and evaluated using **weighted F1-score** and confusion matrices.

| Model                        | Train F1 | Test F1  |
| ---------------------------- | -------- | -------- |
| Logistic Regression (Scaled) | 0.56     | 0.57     |
| SVM (Scaled)                 | 0.68     | 0.68     |
| Random Forest (Tuned)        | 0.99     | 0.84     |
| **LightGBM (Tuned)**         | **1.00** | **0.88** |

### Final Model Selection

* **LightGBM** was selected as the final model due to:

  * Highest test F1-score
  * Better handling of class imbalance
  * Strong performance on non-linear patterns

The trained LightGBM model was saved using **Joblib** for reuse.

---

## Run Instructions

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn lightgbm xgboost statsmodels joblib
```

### 2. Run the Notebook / Script

* Load the dataset
* Execute data cleaning and preprocessing steps
* Perform feature selection
* Train models and evaluate performance

### 3. Load Trained Model

```python
import joblib
model = joblib.load("water_quality_lgbm_model.joblib")
```

### 4. Make Predictions

```python
predictions = model.predict(X_test)
```

---

## Conclusion

This project demonstrates that machine learning, particularly **LightGBM**, can effectively classify water quality using environmental and chemical indicators. The approach reduces manual effort, improves consistency, and provides a scalable solution for water quality monitoring.
