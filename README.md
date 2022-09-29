# Credit_Risk_Analysis
### Supervised Machine Learning and Credit Risk

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Different techniques are used to train and evaluate models with unbalanced classes. Imbalanced-learn and scikit-learn libraries are employed to build and evaluate models using resampling.

Using the 2019 Q1 credit card credit dataset from LendingClub, a peer-to-peer lending services company, I will oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. And lastly, I will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Purpose
- Explain how a machine learning algorithm is used in data analytics.
- Create training and test groups from a given data set.
- Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
- Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
- Compare the advantages and disadvantages of each supervised learning algorithm.
- Determine which supervised learning algorithm is best used for a given data set or scenario.
- Use ensemble and resampling techniques to improve model performance.

## Results
*The results for the six machine learning models including their respective balanced accuracy, precision, and recall scores are as follows:*

As mentioned in the overview, we use Machine Learning to resample the dataset from LendingClub using Python libraries: scikit-learn and imbalanced-learn. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining were grouped as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

![datacount](https://user-images.githubusercontent.com/68654746/193049108-17f5bb90-53dc-470f-8ffa-e7efd189b520.png)

### Naive Random Oversampling
**RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal.**

*The results classified 51,352 records each as High Risk and Low Risk.*

![Naive Random Oversampling](https://user-images.githubusercontent.com/68654746/193042361-9e62f113-3241-4424-9046-db14dae216d1.jpg)
- Balanced Accuracy: 0.643860465491054 or 64%
- Precision: The precision is low of 1% for High-risk loans and is high of 100% for Low-risk loans.
- Recall: High risk 61% / Low risk 68%

### SMOTE Oversampling
**SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.**

![SMOTE Oversampling](https://user-images.githubusercontent.com/68654746/193043547-23144147-7a28-44bc-896a-c99a8214980b.jpg)
- Balanced Accuracy: 0.6521789928729992 or 65% 
- Precision: The precision is low of 1% for High-risk loans and is high of 100% for Low-risk loans.
- Recall: High risk 67% / Low risk 64%

### Undersampling
**ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters.**

*The model classified 260 records each as High Risk and Low Risk.*
![Undersampling](https://user-images.githubusercontent.com/68654746/193044155-9a69c7e2-80ee-452b-9955-8be04200e52c.jpg)
- Balanced Accuracy: 0.6521789928729992 or 65%
- Precision: The precision is low of 1% for High-risk loans and is high of 100% for Low-risk loans.
- Recall: High risk 57% / Low risk 46%

### Combination Under-Over Sampling
**SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling.**

*The model classified 68,460 records as High Risk and 62,011 as Low Risk.*
![Combination Under-Over Sampling](https://user-images.githubusercontent.com/68654746/193044697-3bfdbeb5-1e3a-4e72-85be-77cff9bd9f92.jpg)
- Balanced Accuracy: 0.5184288770441278 or 52%
- Precision: The precision is low of 1% for High-risk loans and is high of 100% for Low-risk loans.
- Recall: High risk 70% / Low risk 58%

### Balanced Random Forest Classifier
**BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.**
![Balanced Random Forest Classifier](https://user-images.githubusercontent.com/68654746/193045293-4a44c455-e050-4c76-8c0e-4f304b93a4e1.jpg)
- Balanced Accuracy: 0.7877672625306695 or 79%
- Precision: The precision is low of 4% for High-risk loans and is high of 100% for Low-risk loans.
- Recall: High risk 67% / Low risk 91%
- total_rec_prncp is 7.38% of the total

### Easy Ensemble AdaBoost Classifier
**EasyEnsembleClassifier Model, a set of classifiers where individual decisions are combined to classify new examples.**
![Easy Ensemble AdaBoost Classifier](https://user-images.githubusercontent.com/68654746/193046597-79f1782a-1804-491a-a920-6d0380254c70.jpg)
- Balanced Accuracy: 0.925427358175101 or 93%
- Precision: The precision increased to 7% for High-risk loans and remains at 100% for Low-risk loans.
- Recall: High risk 91% / Low risk 94%

## Summary
Ranking of models in descending order based on "High Risk" results:

- EasyEnsembleClassifer: 93% accuracy, 7% precision, 91% recall, and 14% F1 Score
- BalancedRandomForestClassifer: 79% accuracy, 4% precision, 67% recall and 7% F1 Score
- SMOTE: 65% accuracy, 1% precision, 67% recall and 2% F1 Score
- SMOTEENN: 52% accuracy, 1% precision, 70% recall and 2% F1 Score
- RandomOverSampler: 64% accuracy, 1% precision, 61% recall and 2% F1 Score
- ClusterCentroids: 65% accuracy, 1% precision, 57% recall and 1% F1 Score

After reviewing all six models, the EasyEnsembleClassifer model yields the best results with the highest balanced accuracy rate of 92.5% and a 7% precision rate when predicting "High Risk" candidates. The sensitivity rate (aka recall score) was also the highest at 91% compared to the findings in other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, I would recommend using the Easy Ensemble AdaBoost Classifier as a clear choice between all the machine learning models for further credit card analysis.

However, it may be important to consider that the original dataset had 99% of the applications classified as "Low Risk" and only 1% were classified in the "High Risk" category. This may skew the results as there is a risk that the Machine Learning algorithms are creating clusters because it is drawing from a much smallier dataset of actual "High Risk" applications. Therefore, this margin of risk might not be something that banks would be comfortable accepting.
