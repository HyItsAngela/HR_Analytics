![](UTA-DataScience-Logo.png)

# Springleaf Marketing Response

* This repository applies machine learning techniques and models to springlife marketing response to predict customer responses to the direct mail (DM) that springleaf sends out to connect with their current and potential clientale.
From Kaggle's "Springleaf Marketing Response" [(https://www.kaggle.com/competitions/springleaf-marketing-response/overview)]. Kaggle provides a high-dimensional dataset that included anonymized customer information/features.

## Overview

* Springleaf is a financial services company that provides customers with personal and auto loans. Direct mail is Springleaf's primary communication source to connect with their current customers and reach out to potential and target customers.
* The task, as defined by the Kaggle challenge is to develop a model to "determine whether to send a direct mail peice to a customer". This repository approaches this problem as a binary classification task, using model CatBoost. The model was able to determine whether a customer succesfully responded to a DM and hence should be futher contacted via DM scored at ~72% accuracy. At the time of this writing, the best performance on the Kaggle leaderboards of this metric is 80%.

## Summary of Work Done

### Data

* Data:
  * Type: Binary Classification
    * Input: CSV file: train.csv, test.csv; described customer response
    * Output: success or failure based on whether or not the customer responded or not -> target col = 'target'
  * Size: Original training and testing datasets together was 1,931 MB (training: 145,231 rows & 1934 features (966 MB); test: 145,232 rows & 1934 features (965 MB). After cleaning and preprocessing, both datasets was about 882 MB.

#### Preprocessing / Clean up

Before proceeding, my machine did not have the memory to compute the entire dataset so I proceeded with a subset/chunk of the data (24,205 rows & 1934 features).
- Dropped features that had one unique & constant values
- Dropped features with over 50% NA values
- Operationalizing NA values
- Confirming data types were appropriate
- Dropped outliers
- Transformed numerical data
- Encoded categorical data

#### Data Visualization

Visualization of some categorical and numerical features that compare the targets.
![image](https://github.com/user-attachments/assets/a75618c1-a765-40ff-81de-8cdcb6c4b31e)

![image](https://github.com/user-attachments/assets/e41a6421-a645-480f-b419-a9cb631c580f)

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
  * Played around with the hyperparameters such as the learning rate, depth, and early round stopping while also optimizing the number of important features to use.

### Performance Comparison

* Key performance metrics were imported from sklearn and consist of:
  * classification_report().
  * accuracy_score().

### Conclusions

Since only one model was used, there are no comparisons between models to be made. The CatBoost model that was used did not perform up to my standards. To increase accuracy, more hyperparameter tuning may be needed.

### Future Work

I would like to mess around more with the preprocessing with this dataset. I would also like to train and compare more models to find a better model, as well as study the learning curves to tune the base model better.

## How to reproduce results

* The notebooks are well organized and include further explanation; a summary is provided below:
* Download the original data files ('train.csv', 'test.csv') from Kaggle or directly through the current repository along with the processed data files.
* Install the necessary libraries
* Run the notebooks attached
* As long as a platform that can provide Python, such as Collab, Anaconda, etc, is used, results can be replicated.

### Overview of files in repository

* The repository includes 4 files in total.
  * data_understanding.ipynb:  provides my intitial walkthrough of trying to understand the data such as class distributions, features and missing values.
  * preprocess_data.ipynb: transforms and encodes the dtypes appropriately.
  * data_clean.ipynb: explores the data further by dealing with the missing values and duplicates, visualizes the data
  * model_prediction.ipynb: trains the model, CatBoost on the preprocessed data.

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

### Training

* Models can be trained by first splitting the testing dataset into two datasets to be trained and validated. Choose the model you wish to train and fit the data and validation variables. Look below in citations to research official websites to find parameters of the model functions to tune to your liking.

#### Performance Evaluation

* Evaluation metrics are imported such as the accuracy score and classification score.
* Run the notebooks.


## Citations
- Official CatBoost website; used to learn about the CatBoost model and parameters it has to offer: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier_eval-metrics
- Official SciKit-Learn website; used to learn about RandomForest and other potential models: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
