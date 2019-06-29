import csv

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

'''
Xtrain:  (5901, 76)
Xtest:  (1967, 76)
ytrain:  (5901, 148)
ytest:  (1967, 148)

'''
def pipeline(training_features, training_target, testing_features, testing_target):
    # NOTE: Make sure that the class is labeled 'target' in the data file
    # tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
    # features = tpot_data.drop('target', axis=1).values
    # training_features, testing_features, training_target, testing_target = \
    #             train_test_split(features, tpot_data['target'].values, random_state=42)
    output = np.zeros((1967, 147))
    for x in range(147):
        print(x)
        Y_train = training_target[:, x]

        KN = KNeighborsClassifier(n_neighbors=100, p=2, weights="distance")
        KN.fit(training_features, Y_train)

        results = KN.predict_proba(testing_features)
        results = results[:, 1]
        results = results.reshape(1, -1)
        output[:, x] = results
        # print(results)

    # print(output)
    score = KN.score(output,testing_target)
    print(score)
    # np.savetxt("topSecret.csv", output, delimiter=",")


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

    tesetfile = 2705
    testList = np.zeros((2705, 76))
    for x in range(tesetfile):
        filename = "/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/test_feature_files/" + str(
            x) + ".npz"
        data = np.load(filename)
        testList[x] = data['summary']
    xtest = testList
    xtest= np.nan_to_num(xtest)

    file = '/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/cal10k_train_data2.csv'
    y = np.array(list(csv.reader(open(file, "r"), delimiter=","))).astype("float")

    # print(y.shape)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    print("data load done")
    # rows,cols = y_test.shape
    pipeline(X_train,y_train,X_test, y_test)
    # for x in range(cols):
    #     Y_train = y_train[:, x]
    #     Y_test = y_test[:, x]
    #     pipeline(X_train, y_train, X_test)
    #     # T_Pot( X_train, X_test, Y_train, Y_test)
main()
