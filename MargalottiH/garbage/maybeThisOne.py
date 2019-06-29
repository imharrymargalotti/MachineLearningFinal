''' ExtraTreesClassifier(BernoulliNB(PCA(input_matrix, iterated_power=9, svd_solver=randomized), alpha=0.001, fit_prior=True), bootstrap=False, criterion=gini, max_features=0.25, min_samples_leaf=5, min_samples_split=16, n_estimators=100)'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import csv
# # NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1).values
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'].values, random_state=42)
#
# # Average CV score on the training set was:0.8607003544826993
# exported_pipeline = make_pipeline(
#     PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
#     LogisticRegression(C=25.0, dual=False, penalty="l2")
# )
#
# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator


def runRealData(training_features, training_target, testing_features):
    rows, cols = testing_features.shape
    pca = PCA(n_components=5)
    pca.fit(training_features)
    output = np.zeros((rows, 147))
    for x in range(147):
        print(x)
        Y_train = training_target[:, x]
        exported_pipeline = LogisticRegression(C=25.0, dual=False, penalty="l2")

        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)
        results = results[:, 1]
        output[:, x] = results
        # print(results)
    np.savetxt("basic.csv", output, delimiter=",")

def pipelineTest(training_features, training_target, testing_features, testing_target):
    rows, cols = testing_features.shape
    pca = PCA(n_components=20)
    pca.fit(training_features)
    output = np.zeros((rows, 147))
    for x in range(147):
        print(x)
        Y_train = training_target[:, x]
        exported_pipeline = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features=0.6000000000000001, min_samples_leaf=12, min_samples_split=12, n_estimators=100)

        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)
        results = results[:, 1]
        output[:, x] = results
        # print(results)

    print(roc_auc_score(testing_target, output))


def pipeline(training_features, training_target, testing_features, testing_target): #this gets me an 88
    rows, cols = testing_features.shape
    pca = PCA(n_components=20)
    #was 20 and got 88%
    # at 40 i got 8831460858300355
    #at 10 i got 8792561959963644
    #at 2 i got 8797151130382569
    output = np.zeros((rows, 147))
    for x in range(147):
        pca.fit(training_features)
        print(x)
        Y_train = training_target[:, x]
        pca.fit(testing_features)
        exported_pipeline = LogisticRegression(C=25.0, dual=False, penalty="l2", solver='lbfgs', n_jobs=-1, max_iter=1500)

        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)
        results = results[:, 1]
        output[:, x] = results
        # print(results)

    print(roc_auc_score(testing_target, output))


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
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("data load done")
    pipeline(X_train,y_train,X_test, y_test)
    # runRealData(X,y,xtest)
    # for x in range(cols):
    #     Y_train = y[:, x]
    #     Y_test = y_test[:, x]
        # pipeline(X, Y_train, xtest)
main()
