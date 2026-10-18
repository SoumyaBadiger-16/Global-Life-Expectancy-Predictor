# Missing Value Imputation and Outlier Treatment

## 📌 Project Overview

This project focuses on data preprocessing techniques used to improve data quality before performing machine learning or statistical analysis. It uses a life expectancy dataset containing health, demographic, and socio-economic indicators across different countries and years.

The main objectives are to identify missing values, apply suitable imputation techniques, detect outliers using visualizations, and evaluate the effectiveness of preprocessing methods.

## 🎯 Objectives

- Identify and analyze missing values in the dataset.
- Apply appropriate imputation strategies to numerical features.
- Use mode imputation for selected vaccination-related features.
- Explore predictive imputation and interpolation techniques for complex features.
- Detect outliers using boxplots and histograms.
- Understand methods for handling outliers.
- Validate the dataset after preprocessing.

## 📂 Dataset Description

The dataset contains country-wise life expectancy and related health and socio-economic indicators.

**Key features include:**

- `Country` — Country name.
- `Year` — Observation year.
- `Status` — Developed or developing status.
- `Lifeexpectancy` — Life expectancy in years.
- `AdultMortality` — Adult mortality indicator.
- `infantdeaths` — Infant deaths.
- `Alcohol` — Alcohol consumption indicator.
- `HepatitisB`, `Polio`, `Diphtheria` — Vaccination-related indicators.
- `BMI` — Body mass index indicator.
- `GDP` — Gross domestic product.
- `Population` — Population.
- `Schooling` — Schooling indicator.
- `Incomecompositionofresources` — Income composition of resources indicator.

The dataset contains 22 columns covering health, education, population, and economic indicators.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## 🔍 Project Workflow

### 1. Missing Value Analysis

Missing values were identified using Pandas.

```python
df.isnull().sum()
```

This provides the number of missing values in each feature.

The column names and data types were also inspected to determine suitable preprocessing methods.

```python
df.columns
df.info()
```

### 2. Missing Value Imputation

Different imputation techniques were considered based on the characteristics of the features.

| Feature group | Proposed method | Reason |
|---|---|---|
| Alcohol, BMI, Totalexpenditure, thinness10-19years, thinness5-9years | Mean or median imputation | Fill missing numerical observations |
| HepatitisB, Polio, Diphtheria | Most-frequent value imputation | Preserve common observed vaccination values |
| Lifeexpectancy, AdultMortality | Predictive imputation | Estimate values using relationships with other features |
| GDP, Population | Time-series interpolation | Estimate missing values from surrounding years |

**Mean and median imputation**

Mean imputation replaces missing values with the average of the observed values. Median imputation uses the middle value and can be more robust when a feature contains outliers.

```python
from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy="mean")
median_imputer = SimpleImputer(strategy="median")
```

The appropriate imputer should be selected based on each feature's distribution.

**Most-frequent value imputation**

```python
mode_cols = ["HepatitisB", "Polio", "Diphtheria"]

mode_imputer = SimpleImputer(strategy="most_frequent")
df[mode_cols] = mode_imputer.fit_transform(df[mode_cols])
```

This fills missing values with the most frequently observed value in each selected column.

**Advanced imputation methods**

Predictive imputation and interpolation were identified as possible approaches for features such as life expectancy, adult mortality, GDP, and population. These methods require separate implementation and validation before they can be considered completed steps.

### 3. Data Visualization

Histograms and boxplots were used to explore numerical feature distributions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

for col in df.select_dtypes(include="number").columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    sns.histplot(data=df, x=col, kde=True, ax=axes[0])
    sns.boxplot(data=df, x=col, ax=axes[1])

    axes[0].set_title(f"Distribution of {col}")
    axes[1].set_title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()
```

Histograms help reveal skewness and distribution patterns, while boxplots help identify potential outliers.

### 4. Outlier Detection

Outliers are observations that differ substantially from the overall distribution of a feature.

Boxplots can highlight potential outliers using the interquartile range (IQR).

The IQR method defines:

- Q1: 25th percentile.
- Q3: 75th percentile.
- IQR = Q3 − Q1.
- Lower bound = Q1 − 1.5 × IQR.
- Upper bound = Q3 + 1.5 × IQR.

Values outside these bounds are flagged as potential outliers, not automatically classified as errors.

### 5. Outlier Treatment Strategies

The following approaches were considered:

- **Removal:** Remove observations only when they are confirmed errors or irrelevant to the analysis.
- **Capping:** Limit extreme values to selected lower and upper thresholds.
- **Transformation:** Apply suitable transformations, such as logarithmic transformations, when appropriate for the distribution.
- **Robust methods:** Use statistical techniques or models that are less sensitive to extreme observations.

The treatment method should depend on the feature, domain knowledge, and the effect on subsequent analysis.

### 6. Validation

After preprocessing, missing values should be checked again.

```python
print(df.isnull().sum())
```

Numerical distributions and boxplots should also be reviewed after outlier treatment.

For machine learning applications, imputation parameters should be learned from the training data only to avoid data leakage. Preprocessing decisions should be validated using appropriate evaluation methods.

## 📊 Expected Outcomes

- Improved understanding of missing-value patterns.
- A documented approach to selecting imputation techniques.
- Visual identification of potential outliers.
- Better understanding of numerical feature distributions.
- A cleaner dataset for subsequent analysis and modeling.

Actual improvements in data quality or model performance should be reported only after the preprocessing steps have been executed and evaluated.

## 🚀 How to Run the Project

1. Clone or download this repository.
2. Open the Jupyter notebook in Google Colab or a local Jupyter environment.
3. Upload the dataset or configure its file path.
4. Install the required Python libraries if necessary.
5. Run the notebook cells in sequence.
6. Review the missing-value summaries, visualizations, and preprocessing results.

## 📚 Key Learnings

- Missing-value analysis and feature-wise imputation.
- Mean, median, and mode imputation.
- Introduction to predictive imputation and interpolation.
- Outlier detection using histograms and boxplots.
- IQR-based outlier identification.
- Data validation and preprocessing best practices.

## 📄 License

This project is intended for educational and learning purposes. Add an appropriate open-source license if you intend to distribute the project under specific reuse terms.
