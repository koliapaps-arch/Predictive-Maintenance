# Predictive Maintenance — Machine Learning Zoomcamp Project

An end-to-end machine learning project for predicting industrial equipment failures using the **AI4I 2020 Predictive Maintenance Dataset**.

The project covers exploratory data analysis, data preprocessing, model training and evaluation, classification threshold optimization, and deployment as a REST API using FastAPI and Docker.

## 1. Problem Description

Unexpected equipment failures can cause production downtime, increased maintenance costs, and operational disruptions.

The objective of this project is to predict whether a machine is likely to fail based on its operating conditions.

This is a **binary classification problem**:

- `0` — No machine failure
- `1` — Machine failure

The model uses operating measurements such as temperature, rotational speed, torque, tool wear, and product type.

The resulting prediction service can support maintenance decisions by identifying equipment operating under potentially problematic conditions.

**Note:** The dataset is synthetic. Predictions should not be interpreted as validated failure probabilities for real industrial equipment.

## 2. Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains:

- 10,000 observations
- 14 original columns
- No missing values
- A binary target variable: `Machine failure`

### Input Features

| Feature | Description |
|---|---|
| `Type` | Product quality type (L, M, H) |
| `Air temperature [K]` | Air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Machine rotational speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Tool wear |

The identifiers (`UDI`, `Product ID`) and failure-mode indicators (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from model training.

Failure-mode indicators are excluded to avoid target leakage.

### Target Distribution

| Class | Observations | Percentage |
|---|---:|---:|
| No failure (0) | 9,661 | 96.61% |
| Failure (1) | 339 | 3.39% |

The dataset is highly imbalanced. Therefore, accuracy alone is not sufficient to evaluate model performance.

## 3. Exploratory Data Analysis

Exploratory data analysis is available in:

`notebooks/01_eda.ipynb`

The analysis includes:

- Dataset structure and data quality checks
- Target class distribution
- Numerical feature distributions
- Boxplots comparing operating conditions
- Correlation analysis
- Failure rates by product type

### Key Findings

- Machine failures represent only 3.39% of observations.
- Air and process temperatures are strongly correlated.
- Rotational speed and torque have a strong negative correlation.
- Equipment operating conditions contain useful information for failure prediction.
- Class imbalance requires particular attention to precision, recall, F1-score, ROC-AUC, and PR-AUC.

## 4. Data Preparation

The data was divided into stratified training, validation, and test sets:

| Dataset | Percentage | Observations |
|---|---:|---:|
| Training | 60% | 6,000 |
| Validation | 20% | 2,000 |
| Test | 20% | 2,000 |

Preprocessing includes:

- StandardScaler for numerical features
- OneHotEncoder for the categorical `Type` feature
- ColumnTransformer and scikit-learn Pipeline

Preprocessing is fitted within the training pipeline to avoid data leakage.

## 5. Model Training and Evaluation

Three machine learning algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. Histogram-based Gradient Boosting

Class imbalance was addressed using class weights or balanced sample weights, depending on the algorithm.

### Validation Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1425 | 0.8235 | 0.2430 | 0.8788 | 0.3680 |
| Random Forest | 0.4700 | 0.6912 | 0.5595 | 0.9680 | 0.5555 |
| HistGradientBoosting | 0.4094 | 0.7647 | 0.5333 | 0.9748 | 0.7137 |

Histogram-based Gradient Boosting was selected because of its strong ROC-AUC and PR-AUC performance.

## 6. Classification Threshold Optimization

The default classification threshold of 0.5 was adjusted using the validation dataset.

The selected threshold was:

**0.87**

At this threshold, the validation metrics were:

| Metric | Value |
|---|---:|
| Precision | 0.7455 |
| Recall | 0.6029 |
| F1-score | 0.6667 |

Threshold optimization improved precision and F1-score, while reducing recall.

In a real maintenance system, the threshold should be selected according to the operational costs of missed failures and false alarms.

## 7. Final Test Results

The final Histogram-based Gradient Boosting pipeline was retrained using the combined training and validation data (8,000 observations).

The previously selected threshold of 0.87 was then applied to the held-out test dataset.

| Metric | Test Result |
|---|---:|
| Accuracy | 0.9855 |
| Precision | 0.8197 |
| Recall | 0.7353 |
| F1-score | 0.7752 |
| ROC-AUC | 0.9753 |
| PR-AUC | 0.8140 |

### Confusion Matrix

| | Predicted: No Failure | Predicted: Failure |
|---|---:|---:|
| Actual: No Failure | 1,921 | 11 |
| Actual: Failure | 18 | 50 |

The model correctly identified 50 out of 68 failures in the test set, while generating 11 false alarms.

These results are based on the synthetic AI4I dataset and should not be assumed to generalize directly to real industrial equipment.

## 8. Project Structure

```text
Predictive-Maintenance/
├── data/
│   └── ai4i2020.csv
├── notebooks/
│   └── 01_eda.ipynb
├── models/
│   └── model.joblib
├── train.py
├── predict.py
├── serve.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## 9. Running the Project Locally

### Clone the Repository

```bash
git clone https://github.com/koliapaps-arch/Predictive-Maintenance.git
cd Predictive-Maintenance
```

Replace `YOUR_USERNAME` with the actual GitHub username.

### Create a Virtual Environment

Python 3.13.5 is used in this project.

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

The training script saves the trained pipeline to:

`models/model.joblib`

The model artifact contains the trained pipeline, the classification threshold, and the feature names.

### Start the API

```bash
python -m uvicorn serve:app --host 0.0.0.0 --port 8000
```

Interactive API documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health

## 10. Running with Docker

Build the Docker image from the project root:

```bash
docker build -t predictive-maintenance .
```

Run the container:

```bash
docker run --rm -p 8001:8000 predictive-maintenance
```

The API will be available at:

http://127.0.0.1:8001/docs

The Docker image includes the trained model artifact.

## 11. Making Predictions

### API Endpoint

`POST /predict`

### Example Request

```json
{
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
```

### Example Response

```json
{
  "machine_failure": 0,
  "failure_probability": 0.0007,
  "threshold": 0.87,
  "status": "Normal operation"
}
```

The response includes:

- `machine_failure`: Binary prediction (0 or 1)
- `failure_probability`: Estimated probability from the classifier
- `threshold`: Classification threshold
- `status`: Human-readable prediction result

### Example Using curl

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
  }'
```

The example above assumes the API is running in Docker on port 8001.

## 12. Technologies Used

- Python 3.13.5
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn for exploratory analysis
- Jupyter Notebook
- joblib for model serialization
- FastAPI and Uvicorn
- Docker

## 13. Limitations and Future Improvements

The project demonstrates an end-to-end machine learning workflow, but several limitations remain:

- The dataset is synthetic rather than collected from operating industrial equipment.
- The data does not represent continuous equipment histories.
- The optimized threshold prioritizes F1-score rather than a real maintenance cost function.
- The reported test metrics come from a single held-out test split.
- Model performance and probability calibration would need additional validation before real-world deployment.

Potential improvements include:

- Cost-sensitive threshold selection
- Probability calibration
- Model explainability with permutation importance or SHAP
- Monitoring for data drift
- Evaluation on real equipment sensor data
- Automated testing and continuous integration

## 14. Conclusion

This project demonstrates the complete machine learning workflow for predictive maintenance, from exploratory data analysis to a containerized prediction API.

The final Histogram-based Gradient Boosting model achieved an F1-score of **0.7752** and ROC-AUC of **0.9753** on the held-out test dataset.

The trained model is exposed through a FastAPI REST endpoint and can be executed in a reproducible Docker environment.

Developed as a project for the **DataTalks.Club Machine Learning Zoomcamp**.
