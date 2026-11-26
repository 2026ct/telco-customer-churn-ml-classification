# Telco Customer Churn Classification

## 1. Problem statement

The objective is to predict whether a telecommunications customer will leave the service (`Churn = Yes`) using customer account, service and billing information. Identifying likely churners can help a telecom provider prioritize retention activities. In this assignment, five classification algorithms are trained and compared using the same stratified 80:20 train-test split.

## 2. Dataset description

- **Dataset:** IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Rows:** 7,043
- **Columns in source file:** 21 (including customer identifier and target)
- **Predictor features:** 19
- **Target:** `Churn` (`No` / `Yes`)
- **Positive class for evaluation:** `Yes` (encoded as 1)
- **Feature types:** 4 numerical (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) and 15 categorical.
- **Class distribution:** 5,174 No-churn records (73.46%) and 1,869 churn records (26.54%).

The dataset satisfies the assignment's minimum size and feature-count requirements. It is useful for demonstrating a binary classification task with mixed numerical and categorical predictors and a moderately imbalanced target.

**Dataset source URL:** The executed notebook identifies the dataset by filename and IBM Telco Customer Churn name, but does not record the exact download URL. Insert the exact URL from which you obtained the CSV before publishing the repository.

## 3. GitHub Repository Link

**Repository:** `https://github.com/2026ct/telco-customer-churn-ml-classification`

The repository should include:
- `README.md`
- `requirements.txt`
- `test_data.csv`
- the executed notebook / source code
- saved model pipeline files (if included in your final project)

## 4. Preprocessing

1. Converted whitespace-only entries to missing values.
2. Converted `TotalCharges` from text to numeric; 11 blank values became missing.
3. Checked duplicate rows and duplicate customer IDs. No duplicates were found; all 7,043 rows were retained.
4. Removed `customerID` from model inputs because it is an identifier, not a behavioural predictor.
5. Encoded the target as `No = 0` and `Yes = 1`.
6. Split the data into 80% training (5,634 records) and 20% testing (1,409 records), using stratification and `random_state=42`.
7. Used median imputation and standard scaling for numerical features.
8. Used most-frequent imputation and one-hot encoding for categorical features.
9. Kept preprocessing inside each model's scikit-learn pipeline, so imputation, scaling and encoding are fitted on training data only. Unseen test categories are ignored by the encoder.

The train and test churn proportions were preserved at approximately 26.54%.

## 5. Models used and evaluation results

All five models used the same split and preprocessing approach. AUC was calculated from predicted probabilities. Accuracy, precision, recall, F1 and MCC were calculated from each model's predicted class labels.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.8413 | 0.5043 | 0.7834 | 0.6136 | 0.4531 |
| Decision Tree | 0.7282 | 0.6496 | 0.4878 | 0.4813 | 0.4845 | 0.3000 |
| K-Nearest Neighbors | 0.7637 | 0.7895 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Gaussian Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Random Forest (Ensemble) | 0.7814 | 0.8217 | 0.6170 | 0.4652 | 0.5305 | 0.3983 |

### Metric leaders

- **Accuracy:** Random Forest (0.7814)
- **AUC:** Logistic Regression (0.8413)
- **Precision:** Random Forest (0.6170)
- **Recall:** Gaussian Naive Bayes (0.8369)
- **F1:** Logistic Regression (0.6136)
- **MCC:** Logistic Regression (0.4531)

## 6. Model-wise observations

### Logistic Regression
Logistic Regression gives the highest AUC (0.8413), F1 (0.6136) and MCC (0.4531). Its churn recall is also strong at 0.7834, meaning it identifies a comparatively large share of actual churners. Precision is 0.5043, so some customers flagged as churners will not actually churn. It offers the strongest overall balance in these results and is the recommended overall model when balancing churn detection and classification quality is important.

### Decision Tree
The Decision Tree has accuracy of 0.7282, AUC of 0.6496, F1 of 0.4845 and MCC of 0.3000. Its churn recall is 0.4813, the lowest alongside the Random Forest being slightly lower. It is the weakest model overall on AUC, F1 and MCC in this experiment. A single train-test result is not enough to conclude that it overfits; a separate training-versus-validation comparison would be needed.

### K-Nearest Neighbors
KNN achieves accuracy of 0.7637 and precision of 0.5527, with recall of 0.5749 and F1 of 0.5636. Standard scaling is appropriate for this distance-based algorithm because feature magnitudes can otherwise dominate distance calculations. Its results are relatively balanced, but its churn recall and AUC (0.7895) are below Logistic Regression and Gaussian Naive Bayes.

### Gaussian Naive Bayes
Gaussian Naive Bayes obtains the highest churn recall (0.8369), meaning it catches the largest proportion of actual churners among the five models. However, it has the lowest accuracy (0.6948) and precision (0.4589), so it also produces more false-positive churn alerts. Its F1 (0.5928) and AUC (0.8074) are reasonably competitive. It may be useful when missing a likely churner is especially costly and the business can tolerate more retention outreach.

### Random Forest
Random Forest has the highest accuracy (0.7814) and precision (0.6170), indicating the best overall correctness rate and the strongest reliability among positive churn predictions in this test. However, its churn recall is 0.4652 and F1 is 0.5305, so it misses more than half of the actual churners. Despite outperforming the single Decision Tree on accuracy, AUC, precision, recall, F1 and MCC, it is not the strongest choice when churn detection is the main objective.

### Overall winner
**Logistic Regression is the overall winner for this experiment**, using F1 as the primary selection criterion and AUC, recall and MCC as supporting measures. It has the best F1 (0.6136), AUC (0.8413) and MCC (0.4531), while maintaining high churn recall (0.7834). Random Forest leads in accuracy and precision, and Gaussian Naive Bayes leads in recall, so the final choice should reflect business costs. If the primary goal is to minimize missed churners, Gaussian Naive Bayes deserves consideration; if the goal is a balanced model, Logistic Regression is the strongest measured option here.

## 7. Visualizations

The executed notebook generates:
- a combined bar chart comparing all six evaluation metrics across the five models;
- an individual comparison chart for each metric;
- confusion matrices for all five classifiers; and
- an ROC curve comparison.

Include the generated graph images in the repository or capture them in the final report. The combined metric chart and ROC curve are particularly useful for explaining the overall comparison.

## 8. Execution instructions

1. Clone or download the repository.
2. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project directory (or update `DATA_PATH` in the notebook).
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook in Jupyter:

   ```bash
   jupyter notebook
   ```

5. Run all cells from top to bottom.
6. The notebook writes evaluation outputs, plots and fitted model pipelines under `assignment_outputs/`.

**Important:** The notebook's test-data export contains the held-out 1,409 test records and their true `Churn` labels. Use that generated file as `test_data.csv` in the repository. Do not use the held-out test data for model training or tuning.

## 9. Limitations and reproducibility notes

- The notebook output reports one stratified hold-out evaluation with `random_state=42`; results may vary with a different split or configuration.
- The models use the stated initial configurations and are not reported as hyperparameter-tuned.
- KNN and Gaussian Naive Bayes do not use the `class_weight` setting used by Logistic Regression, Decision Tree and Random Forest.
- The dataset source's exact URL and access date must be added by the repository owner because they are not recorded in the executed notebook.
- Add the working GitHub URL after creating the repository.


