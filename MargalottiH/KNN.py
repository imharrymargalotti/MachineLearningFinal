import csv
import numpy as np
import os
from sklearn import datasets
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.multioutput import MultiOutputClassifier


def KNN(X_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(algorithm='auto', metric='minkowski', metric_params=None, n_jobs=-1,
                         n_neighbors=int(np.sqrt(147)), p=2, weights='distance')

    knn.fit(X_train, y_train)
    classifier = MultiOutputClassifier(knn, n_jobs=-1)
    classifier.fit(X_train, y_train)
    y_predict = (classifier.predict_proba(x_test))
    output = np.zeros((1967,147)) #2597
    for x in range(1967):
        for y in range(147):
            output[x][y] = y_predict[y][x][1]
    # print(output)
    # np.savetxt("sub.csv", output, delimiter=",")
    print(classifier.score(output,y_test))


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

    file = '/Users/harrymargalotti/MLfinal/MachineLearningFinal/Kaggle_Final/cal10k_train_data.csv'
    y = np.array(list(csv.reader(open(file, "r"), delimiter=","))).astype("float")

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    # print(y_train.shape)
    KNN(X_train, X_test, y_train, y_test)
main()










'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

create a data matrix X_train from the 74-dimensional summary audio feature vectors for each of the 7868 training tracks: done
Load the cal10k_train_date matrix Y_train for the 147 genres and 7868 training tracks
Using scikit-learn, train 147 logistic regression classifiers with one for each genre
Iterate through the list of test npz files and create a data matrix X_test from the 74-dimensional summary audio feature vectors for each of the 2705 test tracks
Predict the probability of each test track and each genre. This should be a 2705-by-147 dimensional matrix called Y_predict
Format Y_predict so that it match the file format that is given in the cal10k_test_random_submission.csv
Upload your submission csv to Kaggle and check out the leaderboard
------------------------------------------------------------------------------------------------------------------------------
The training set is a subset of the data set used to train a model.

x_train is the training data set.
y_train is the set of labels to all the data in x_train.
The test set is a subset of the data set that you use to test your model after the model has gone through initial vetting by the validation set.

x_test is the test data set.
y_test is the set of labels to all the data in x_test.



'''