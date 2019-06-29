import csv
import numpy as np
import os
import sklearn
from sklearn import datasets
from sklearn import neighbors, datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.multioutput import MultiOutputClassifier
from tpot import TPOTClassifier

def T_Pot(X_train, X_test, y_train, y_test):
    pipeline_optimizer = TPOTClassifier(generations=5, population_size=50, cv=5,
                                        random_state=42, verbosity=2, early_stop=3, n_jobs=-1)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('pipelineOutput.py')

def main():
    Trainfiles = 7868
    TrainList = np.zeros((7868, 76))
    for x in range(Trainfiles):
        filename = "/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/train_feature_files/" + str(
            x) + ".npz"
        data = np.load(filename)
        TrainList[x] = data['summary']

    X = TrainList
    X = np.nan_to_num(X)

    # tesetfile = 2705
    # testList = np.zeros((2705, 76))
    # for x in range(tesetfile):
    #     filename = "/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/test_feature_files/" + str(
    #         x) + ".npz"
    #     data = np.load(filename)
    #     testList[x] = data['summary']
    # xtest = testList
    # xtest= np.nan_to_num(xtest)

    file = '/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/cal10k_train_data2.csv'
    y = np.array(list(csv.reader(open(file, "r"), delimiter=","))).astype("float")

    # print(y.shape)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    print("data load done")
    rows,cols = y_test.shape
    for x in range(cols):
        Y_train = y_train[:, x]
        Y_test = y_test[:, x]
        T_Pot( X_train, X_test, Y_train, Y_test)
main()
