# TabPFN for Decision Support

A small, reproducible benchmark comparing **TabPFN** with two established machine-learning baselines on structured classification data.

The project uses the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn and compares:

- Logistic Regression
- Random Forest
- TabPFN

The goal is not only to compare aggregate predictive performance, but also to examine **malignant recall** and individual misclassification patterns.

> This project is an exploratory machine-learning benchmark and is **not intended for clinical use or medical decision-making**.

## Research question

**How does TabPFN compare with established machine-learning baselines on a small tabular classification task, and what can we learn from the models' misclassifications?**

## Dataset

The analysis uses the **Wisconsin Diagnostic Breast Cancer** dataset included with scikit-learn.

- 569 observations
- 30 numerical features
- Target classes:
  - `0` = malignant
  - `1` = benign
- Stratified 75/25 train-test split
- `random_state=42`

## Models

### Logistic Regression
A standardized Logistic Regression is used as a simple linear baseline.

### Random Forest
A Random Forest with 300 estimators provides a non-linear ensemble baseline.

### TabPFN
TabPFN is evaluated on exactly the same train-test split using the `tabpfn-client`.

## Evaluation

The models are compared using:

- Accuracy
- F1-score
- ROC-AUC
- Malignant Recall

### Results

| Model | Accuracy | F1-Score | ROC-AUC | Malignant Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.986 | 0.989 | 0.998 | 0.981 |
| TabPFN | 0.979 | 0.983 | 0.997 | 0.962 |
| Random Forest | 0.958 | 0.967 | 0.995 | 0.925 |

On this particular train-test split, Logistic Regression achieved the strongest overall performance. TabPFN performed very closely and outperformed the Random Forest baseline on accuracy, F1-score and malignant recall.

The experiment illustrates why model selection should remain **empirical and context-dependent** rather than being based on model complexity alone.

## Error analysis

Seven test observations were misclassified by at least one model.

Among the malignant cases:

- Logistic Regression missed 1 case.
- TabPFN missed 2 cases.
- Random Forest missed 4 cases.
- One malignant observation (case 73) was misclassified as benign by all three models.

An exploratory comparison of selected features showed that several measurements of case 73 were closer to the benign group averages than to the malignant group averages. This provides context for why the observation may have been challenging, but it is **not a formal explanation of the models' predictions**.

## Key takeaways

- TabPFN achieved strong predictive performance with minimal model-specific setup.
- A comparatively simple Logistic Regression remained highly competitive.
- Model complexity alone did not determine predictive performance.
- Class-specific metrics added useful information beyond overall accuracy.
- Examining individual errors revealed observations that were difficult across different modelling approaches.

## Limitations

This is an exploratory benchmark based on a small public dataset and a single stratified train-test split.

The results should not be interpreted as evidence that one model is generally superior to another. Performance may vary with different datasets, splits, preprocessing choices, hyperparameters and evaluation strategies.

The case-level error analysis is descriptive and should not be interpreted as a causal or model-specific explanation.

## Run the notebook

The easiest way to reproduce the analysis is in **Google Colab**.

Install the TabPFN client:

```python
!pip install -q tabpfn-client
```

Then run the notebook from top to bottom.

The first TabPFN call may require authentication with the TabPFN service.

## Project structure

```text
tabpfn-decision-support/
├── TabPFN_for_Decision_Support.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Tech stack

- Python
- pandas
- scikit-learn
- matplotlib
- TabPFN / tabpfn-client

## Development note

This project was developed iteratively with AI assistance. The research question, experimental design, model comparison, metric selection, error analysis and interpretation were reviewed and refined throughout the process.
