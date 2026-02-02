# healthcare-cost-analysis
Healthcare cost prediction and analysis using Python and machine learning
# Healthcare Cost Prediction & Analysis

## Project Overview
This project focuses on analysing individual-level healthcare cost data to identify high-cost individuals using data analytics and machine learning techniques.  
The objective is to explore key drivers of healthcare costs and build predictive models that support cost control and decision-making.

## Dataset
The dataset contains individual-level healthcare and demographic information, including medical visits, health conditions, and associated costs.  
Publicly available healthcare cost data is used for educational and analytical purposes.

## Methodology
The project follows an end-to-end data analysis workflow:
- Data cleaning and preprocessing using Python (Pandas, NumPy)
- Exploratory Data Analysis (EDA) to understand cost distributions and key variables
- Feature engineering to construct meaningful predictors
- Model development using regression and tree-based models
- Model evaluation and interpretation

## Tools & Technologies
- Python (Pandas, NumPy, scikit-learn)
- SQL (exploratory querying)
- Jupyter Notebook
- Visual Studio Code
- Git

## Project Structure
healthcare-cost-analysis/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_modeling.ipynb
├── src/
├── sql/
├── results/
└── README.md

## Key Outcomes
- Identification of factors associated with high healthcare costs
- Comparison of regression and tree-based models
- Actionable insights for healthcare cost analysis

## Future Work
- Hyperparameter tuning and model optimisation
- Incorporation of additional socioeconomic features
- Extension to time-based or longitudinal analysis
- 
Feature Engineering 
* Applied binary encoding for gender and smoking status to preserve interpretability
* Used one-hot encoding for geographic regions to avoid imposing ordinal structure
* Engineered interaction features to capture non-linear cost drivers
* Log-transformed target variable to reduce skewness and stabilize variance



## Modelling

### Objective
The goal of the modelling stage is to evaluate whether the engineered features are informative in predicting individual healthcare costs, and to establish a robust baseline regression model with proper preprocessing.

The target variable is the log-transformed medical charges (`log_charges`), which helps reduce skewness and stabilise variance.

---

### Train–Test Split
The dataset is split into training and testing sets using an 80/20 split:

- Training set: 80%
- Test set: 20%
- Random seed fixed for reproducibility

This ensures model performance is evaluated on unseen data.

---

### Preprocessing Pipeline
A unified preprocessing pipeline is constructed using `ColumnTransformer` to ensure consistent feature transformations during both training and inference.

- **Numerical features** (`age`, `bmi`, `children`, `bmi_smoker`) are standardised using `StandardScaler`
- **Remaining features** (e.g. one-hot encoded categorical variables) are passed through unchanged

This approach avoids data leakage and keeps preprocessing tightly coupled with the model.

---

### Model Selection
Ridge Regression is used as the baseline model due to its ability to:

- Handle multicollinearity introduced by one-hot encoding
- Stabilise coefficient estimates via L2 regularisation
- Maintain interpretability of feature coefficients

The full modelling workflow is implemented using a `Pipeline` that combines preprocessing and model fitting into a single object.

---

### Model Training
The model is trained on the training dataset using the preprocessing + Ridge regression pipeline. This ensures all transformations are learned exclusively from the training data.

---

### Evaluation
Model performance is evaluated on the test set using standard regression metrics such as:

- R² (coefficient of determination)
- Root Mean Squared Error (RMSE)

These metrics assess both explanatory power and predictive accuracy.

---

### Interpretation
The Ridge model coefficients are examined to understand the direction and relative importance of engineered features, providing insights into key drivers of healthcare costs.
