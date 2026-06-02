# COVID-19 Mortality Prediction in Cyprus (2020–2025)

Machine Learning project for predicting COVID-19 mortality using anonymized patient data from Cyprus.

## Overview

This project investigates the use of Machine Learning models to predict mortality outcomes in COVID-19 patients. The study evaluates multiple classifiers, applies feature selection techniques, performs statistical significance testing, and identifies the most effective predictive model.

## Dataset

* **Source:** Anonymized COVID-19 patient records from Cyprus
* **Period:** 2020–2025
* **Original size:** ~73,000 records
* **Final dataset:** 1,800 patients (900 deceased, 900 recovered)
* **Features:** 24 demographic, clinical, and hospitalization variables
* **Target:** Mortality outcome (Death / Recovery)

### Main Features

* Age
* Sex
* City
* Symptoms
* Comorbidities
* Hospitalization
* ICU Admission
* Ventilation
* Intubation
* Vaccination
* Reinfection
* Various COVID-19 symptoms

## Methodology

### Feature Selection

* Sequential Forward Floating Selection (SFFS)
* 5-Fold Cross Validation
* ROC-AUC used as optimization metric

### Machine Learning Models

* Logistic Regression (LR)
* Random Forest (RF)
* Support Vector Machine (Linear)
* Support Vector Machine (Polynomial)
* Support Vector Machine (RBF)
* Multi-Layer Perceptron (MLP)

### Statistical Analysis

* Wilcoxon Signed-Rank Test
* Friedman Test
* Mann-Whitney U Test

### Hyperparameter Optimization

* GridSearchCV
* 3-Fold Cross Validation

## Results

| Model               | Best ROC-AUC |
| ------------------- | ------------ |
| Random Forest       | 0.9903       |
| MLP                 | 0.9878       |
| Logistic Regression | 0.9867       |
| SVM (Linear)        | 0.9862       |
| SVM (RBF)           | 0.9748       |
| SVM (Polynomial)    | 0.9736       |

### Best Model

Although Random Forest achieved the highest ROC-AUC, the **MLP** was selected as the optimal model because it achieved comparable performance using **12 features** instead of **20**, resulting in a simpler and more interpretable solution.

### Optimal MLP Configuration

```python
activation = "tanh"
alpha = 0.0001
hidden_layer_sizes = (20, 10, 10)
learning_rate = "constant"
solver = "adam"
```

### Final MLP Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 95.67% |
| Precision | 96.18% |
| Recall    | 95.11% |
| F1 Score  | 95.64% |
| ROC-AUC   | 95.67% |

## Important Features

The most influential features identified by SFFS and statistical analysis were:

* Age
* Symptoms
* Myalgia
* Diarrhea
* Running Nose
* Shortness of Breath
* Weakness
* Hospitalization
* Comorbidities
* ICU Admission
* Vaccination

The **Reinfection** feature was not statistically significant and was removed.

## Technologies

* Python
* Scikit-Learn
* NumPy
* Pandas
* SciPy
* mlxtend
* Matplotlib
* Seaborn

## Conclusion

The combination of SFFS feature selection and machine learning models proved highly effective for COVID-19 mortality prediction. The MLP achieved near state-of-the-art performance while requiring fewer features than Random Forest, making it a strong candidate for interpretable clinical decision-support systems.
