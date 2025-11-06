# breast-cancer-ml-comparison

## Overview

This repository contains a Jupyter Notebook that builds, trains, and evaluates several machine learning classification models. It serves as the experimental foundation, where different algorithms are compared based on their performance metrics.

The notebook demonstrates the entire process of:

* Data loading and preprocessing
* Model training and timing
* Model evaluation with various metrics
* Visualization of performance results

The goal is to analyze and compare multiple classifiers under a uniform experimental setup.

---

## Models Implemented

The following models are included:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Random Forest
5. Gradient Boosting
6. Naive Bayes

Each model is evaluated on the same dataset with consistent preprocessing for fair comparison.

---

## Metrics Evaluated

For each model, the notebook computes:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* Training Time

Visualizations include:

* Bar charts comparing metrics across models
* ROC curves for all models
* Confusion matrices for detailed classification performance

---

## Dataset

By default, the notebook uses the **Breast Cancer Dataset** from `sklearn.datasets`.
You can replace it with your own dataset by modifying the data loading cell.

Example:

```python
# Replace this cell with your own dataset
data = pd.read_csv("your_dataset.csv")
X = data.drop('target', axis=1)
y = data['target']
```

---

## Requirements

Install the required Python libraries before running the notebook:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/ShloakN/breast-cancer-ml-comparison.git
   cd breast-cancer-ml-comparison
   ```
2. Open the notebook:

   ```bash
   jupyter notebook ml_model_comparison.ipynb
   ```
3. Run all cells to train and evaluate all models.

---

## Optional Extensions

The notebook also includes an example for **hyperparameter tuning** using `GridSearchCV` for SVM.
You can extend it to other models or add advanced algorithms like XGBoost or LightGBM.

---

## Results

After running the notebook, a summary table ranks all models by F1-score and visual results are displayed for performance comparison.

---

## License

This project is released under the MIT License. You are free to use, modify, and distribute it with proper attribution.

---

## Author

Developed by **Shloak Nioding** as part of the Honors project.
