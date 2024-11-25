![](UTA-DataScience-Logo.png)

# Human Resources/People Analytics 

* This repository leverages a straightforward machine learning model to predict employee termination status while incorporating statistical testing to address key business questions and uncover insights into organizational health.
The dataset used is Kaggle's "Human Resources Dataset" [(https://www.kaggle.com/datasets/rhuebner/human-resources-data-set)], a rich source of information containing diverse features that can be used to gather meaningful analysis and insight.

## Overview

* People analytics is very important for organizations and shareholders because it provides actionable insights into workforce dynamics, that enable leaders to make data-driven decisions for business success.
* The Kaggle task involves identifying key performance indicators (KPIs) that can benefit a company while developing a predictive model to determine whether an employee will be terminated. This project explores several critical business questions, including:

- Turnover and Retention: What are the turnover and retention rates? What is the company's overall net growth?
- Employee Satisfaction and Performance: Is there a relationship between employee satisfaction (engagement) and their performance?
- Engagement Analysis: How does employee engagement vary by age, gender, and race?
- Manager Impact: Is there a correlation between who an employee works for and their performance or satisfaction?
- Diversity: What is the organization's overall diversity profile?
- Recruiting Sources: Which recruiting sources are most effective for ensuring diversity?
- Pay Equity: Are there areas where pay is inequitable within the company?
- Predictive Modeling: Can we predict whether an employee will terminate, and how accurate can the prediction be?
  
* This repository approaches the problem as a binary classification task, leveraging the Random Forest model. The model achieved an impressive ~98% accuracy in predicting whether an employee would be terminated.

## Summary of Work Done

### Data

* Data:
  * Type: Binary Classification
    * Input: CSV file: HR_Datasetv14.csv
    * Output: success or failure based on whether or not the employee will be terminated or not -> target col = 'termid'
  * Size: 76 KB; 311 rows & 36 features

#### Preprocessing / Clean up

- Standardize Data Columns
- Confirming data types were appropriate
- Trim leading spaces
- Resolve duplicates values
- Binned categries to reduce number of values
- Feature engineered columns
  - Age
  - Tenure
  - Diversity fair performance
  - Salary/department average ratio
  - Performance score/department perf. average
  - Days since last performance review

#### Data Visualization

TBD

### Problem Formulation

* The features were anonymized so not much domain knowledge could be used, however there were some hints as to what columns were such as job titles, salary, states, and cities. The features were used in the model to help the company, Springleaf better connect with their current clientale and bring in potential customers.
  * Models
    * Catboost; chosen for it's built-in methods, predictive power and great results without the need for parameter tuning, and robustness.
  * Some in-depth fine-tuning or optimization to the model was performed such as hypyerparameters and feature importance. 

### Training

* Describe the training:
  * Training was done on a Surface Pro 9 using Python via jupyter notebook.
  * Training did not take long to process, with the longest training time to be approximately a minute.
  * Concluded training when results were satisfactory and plenty of evaluation metrics for comparison observed fairly decent results.

### Performance Comparison

* Key performance metrics were imported from sklearn and consist of:
  * classification_report().
  * accuracy_score().

### Conclusions

Since only one model was used, there are no comparisons between models to be made. The RandomForest model performed great, and was further utilized as a predictive model to apply a risk score upon employees.

### Future Work

I want to work on adding a Tableau dashboard along with practicing with unsupervised learning to explore clusters.

## How to reproduce results

* The notebooks are well organized and include further explanation; a summary is provided below:
* Download the original data file (HR_Datasetv14) from Kaggle or directly through the current repository.
* Install the necessary libraries
* Run the notebooks attached
* As long as a platform that can provide Python, such as Collab, Anaconda, etc, is used, results can be replicated.

### Overview of files in repository

* The repository includes 4 files in total.
  * EDA.ipynb:  provides my intitial walkthrough of trying to understand the data such as class distributions, features and missing values. Feature engineered columns to set up for exploration of KPI's.
  * df_preprocess.ipynb: transforms the dataset appropriately. Diverges into two datasets, one used for modeling in RandomForest, the other for statistical testing.
  * baseline_model.ipynb: baseline model of RandomForest that was tested on the raw data to be used for comparison for more advanced/tuned models.
  * model_RandomForest.ipynb: The "advanced" or final model that was used as the predictive model.
  * analysis.ipynb: notebook that explores business questions in detail. Utilized statistical testing.
  * HR_Datasetv14: The official dataset from Kaggle (subject to change as the author may update the dataset--this is version 14)

### Software Setup
* Required Packages:
  * Numpy
  * Pandas
  * Sklearn
  * Seaborn
  * Matplotlib.pyplot
  * Math
  * Scipy
  * Tabulate
* Installlation Proccess:
  * Installed through Linux subsystem for Windows
  * Installed via Ubuntu
  * pip3 install numpy
  * pip3 install pandas
  * pip3 install -U scikit-learn

### Data

* Data can be downloaded through the official Kaggle website through the link stated above. Or through Kaggle's API interface. Can also be downloaded directly through the datasets provided in this directory.

## Citations
- Official SciKit-Learn website; used to learn about RandomForest and other potential models: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
