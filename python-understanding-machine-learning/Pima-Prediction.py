
import pandas as pd                     # pandas is a data frame library
import matplotlib.pyplot as plt         # matplotlib.pyplot plots data
import numpy as np                      # numpy provides N-dim object support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


df = pd.read_csv("./data/pima-data.csv")
# print(df.shape)
# print(df.head(5))
#
# print(df.tail(5))
#
# print(df.isnull().values.any())

corr = df.corr()
# fig, ax = plt.subplots(figsize=(11, 11))
# ax.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns)
# plt.yticks(range(len(corr.columns)), corr.columns)

# print(df.corr())
# print("column deleted")
del df['skin']
# print(df.corr())

diabetes_map = {True : 1, False : 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)
# print(df.head())

# Spliting data
# 70% for traning, 30% for testing

feature_col_name = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_name = ['diabetes']

x = df[feature_col_name].values
y = df[predicted_class_name].values
split_test_size = 0.30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=45) # test_size = 0.30 is 30%, 42 is the answer to everything

print("{0:2.2f}% in traning set".format((len(x_train)/len(df.index))*100))
print("{0:2.2f}% in test set".format((len(x_test)/len(df.index))*100))


num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Original true cases: {0} ({1:2.2f}%)".format(num_true, ((num_true*100) / len(df.index))))
print("Original false cases: {0} ({1:2.2f}%)".format(num_false, ((num_false*100) / len(df.index))))


num_true = len(y_train[[y_train[:] == 1]])
num_false = len(y_train[[y_train[:] == 0]])
print("Training true cases: {0} ({1:2.2f}%)".format(num_true, ((num_true*100) / len(y_train))))
print("Training false cases: {0} ({1:2.2f}%)".format(num_false, ((num_false*100) / len(y_train))))


num_true = len(y_test[[y_test[:] == 1]])
num_false = len(y_test[[y_test[:] == 0]])
print("Test true cases: {0} ({1:2.2f}%)".format(num_true, ((num_true*100) / len(y_test))))
print("Test false cases: {0} ({1:2.2f}%)".format(num_false, ((num_false*100) / len(y_test))))


# Hidden missing vaues

# print(df.head())
#
# print("")
# print("# of rows in dataframe {0}".format(len(df)))
# print("# of missing values in num_preg {0}".format(len(df.loc[df['num_preg'] == 0])))
# print("# of missing values in glucose_conc {0}".format(len(df.loc[df['glucose_conc'] == 0])))
# print("# of missing values in diastolic_bp {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
# print("# of missing values in thickness {0}".format(len(df.loc[df['thickness'] == 0])))
# print("# of missing values in insulin {0}".format(len(df.loc[df['insulin'] == 0])))
# print("# of missing values in bmi {0}".format(len(df.loc[df['bmi'] == 0])))
# print("# of missing values in diab_pred {0}".format(len(df.loc[df['diab_pred'] == 0])))
# print("# of missing values in age {0}".format(len(df.loc[df['age'] == 0])))

# Impute with mean all 0 readings
fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)


# Training initial Algorithm - Naive Bayes

# Create Gaussian Naive Bayes model object and train it with the data
# nb_model = GaussianNB()
# nb_model.fit(x_train, y_train.ravel())
# # Performance on training data
# nb_predict_train = nb_model.predict(x_train)
# # Accuracy
# print("Accuracy train data: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
# # Performance on testing data
# nb_predict_test = nb_model.predict(x_test)
# # Accuracy
# print("Accuracy test data: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
#
# # Metrics
#
# print("")
# print("Confusion Metrics")
# print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
#
# print("")
# print("Classification Metrics")
# print(metrics.classification_report(y_test, nb_predict_test))
#
#
# # Random Forest Algorithm
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(x_train, y_train.ravel())
# rf_predict_train = rf_model.predict(x_train)
# print("Accuracy  RandomForestClassifier train data {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
# rf_predict_test = rf_model.predict(x_test)
#
#
# print("")
# print("Confusion Metrics")
# print("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test)))
# print("")
# print("Classification Metrics")
# print(metrics.classification_report(y_test, rf_predict_test))
#
#
# # Logistic Regression
#
# lr_model = LogisticRegression(C=0.7, random_state=42)
# lr_model.fit(x_train, y_train.ravel())
# lr_predict_test = lr_model.predict(x_test)
# # Training metrics
# print("Accuracy  Logistic Regression test data {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
# print("")
# print("Confusion Metrics")
# print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test)))
# print("")
# print("Classification Metrics")
# print(metrics.classification_report(y_test, lr_predict_test))
#
# C_start = 0.1
# C_end = 0.5
# C_inc = 0.1
#
# C_values, recall_scores = [], []
#
# C_val = C_start
# best_recall_score = 0
# while (C_val < C_end):
#     C_values.append(C_val)
#     lr_model_loop = LogisticRegression(C=C_val, random_state=42)
#     lr_model_loop.fit(x_train, y_train.ravel())
#     lr_predict_loop_test = lr_model_loop.predict(x_test)
#     recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
#     recall_scores.append(recall_score)
#     if (recall_score > best_recall_score):
#         best_recall_score = recall_score
#         best_lr_predict_test = lr_predict_loop_test
#
#     C_val = C_val + C_inc
#
# best_recall_score_C_val = C_values[recall_scores.index(best_recall_score)]
# print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_recall_score_C_val))
#
#
# # Logistic Regression with class_weight parameter
#
# C_start = 0.1
# C_end = 0.5
# C_inc = 0.1
#
# C_values, recall_scores = [], []
#
# C_val = C_start
# best_recall_score = 0
# while (C_val < C_end):
#     C_values.append(C_val)
#     lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42)
#     lr_model_loop.fit(x_train, y_train.ravel())
#     lr_predict_loop_test = lr_model_loop.predict(x_test)
#     recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
#     recall_scores.append(recall_score)
#     if (recall_score > best_recall_score):
#         best_recall_score = recall_score
#         best_lr_predict_test = lr_predict_loop_test
#
#     C_val = C_val + C_inc
#
# best_recall_score_C_val = C_values[recall_scores.index(best_recall_score)]
# print("with class weight -- 1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_recall_score_C_val))
#
#
# lr_model = LogisticRegression(C=best_recall_score_C_val, class_weight="balanced", random_state=42)
# lr_model.fit(x_train, y_train.ravel())
# lr_predict_test = lr_model.predict(x_test)
# # Training metrics
# print("Accuracy  Logistic Regression test data {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
# print("")
# print("Confusion Metrics")
# print("{0}".format(metrics.confusion_matrix(y_test, lr_predict_test)))
# print("")
# print("Classification Metrics")
# print(metrics.classification_report(y_test, lr_predict_test))

# LogisticRegressionCV Algo
def lrcvmodel():
    lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced")
    lr_cv_model.fit(x_train, y_train.ravel())
    lr_cv_predict_test = lr_cv_model.predict(x_test)
    # Training metrics
    print("Accuracy  Logistic RegressionCV test data {0:.4f}".format(metrics.accuracy_score(y_test, lr_cv_predict_test)))
    print("")
    print("Confusion Metrics")
    print("{0}".format(metrics.confusion_matrix(y_test, lr_cv_predict_test)))
    print("")
    print("Classification Metrics")
    print(metrics.classification_report(y_test, lr_cv_predict_test))



if __name__ == '__main__':
    lrcvmodel()