import csv
import numpy as np
import os
from sklearn import datasets, svm
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.multioutput import MultiOutputClassifier

output = np.zeros((7868, 147))
def SVM(X, Y):
    clf = svm.SVC(C=1.0, cache_size=200, class_weight=None,
                  coef0=0.0, decision_function_shape='ovr', degree=3,
                  gamma='scale', kernel='rbf', max_iter=-1, probability=False,
                  random_state=None, shrinking=True, tol=0.001, verbose=False)

    clf.fit(X, Y)
    res = clf.predict_proba(X, Y)
    output = np.concatenate(res)

def main():
    file = '/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/cal10k_train_data.csv'
    y = np.array(list(csv.reader(open(file, "r"), delimiter=","))).astype("float")

    Trainfiles = 7868
    TrainList = np.zeros((7868, 76))
    for x in range(Trainfiles):
        filename = "/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/train_feature_files/" + str(
            x) + ".npz"
        data = np.load(filename)
        temp = data['summary']
        temp = np.nan_to_num(temp)
        y_train = y[x:x+1,1: ]
        # temp=temp.T
        # print(temp.shape)
        # print(y_train.shape)
        temp = temp.reshape((1, 76))
        SVM(temp,y_train)
    print(output)
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


    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print(y_train.shape)
    KNN(X_train, X_test, y_train, y_test)
main()