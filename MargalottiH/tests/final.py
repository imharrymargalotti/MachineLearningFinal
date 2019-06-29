import tensorflow as tf
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression
from tensorflow import keras
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def writestuff(list1):
    with open('DATA.csv', 'w') as f:
        writer1 = csv.writer(f)
        writer1.writerow(list1)

def readClassifiers():
    with open('DATA.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 1
        for row in csv_reader:
            if line_count==1:
                data = row
            line_count+=1
    return data

def logisticTester(training_features, training_target, testing_features, testing_target):
    rows, cols = testing_features.shape
    rocs = []
    output = np.zeros((rows, 147))
    for x in range(147):
        Y_train = training_target[:, x]

        exported_pipeline = LogisticRegression(C=30.0, dual=False, penalty="l2", solver='lbfgs',
                                               multi_class='multinomial', max_iter=1500, n_jobs=-1)

        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)

        score = roc_auc_score(testing_target[:, x], results[:, 1])
        print()
        if score < 0.80:
            print("SCORE:\t\t\t\tGenre:\t\tGood(0)/Bad(1)\t\tBest Algorithm")
            print(score, "\t\t", x, "\t\t", 1, "\t\t\t\t\t", "Logistic")
        else:
            print("SCORE:\t\t\t\tGenre:\t\tGood(0)/Bad(1)\t\tBest Algorithm")
            print(score, "\t\t", x, "\t\t", 0, "\t\t\t\t\t", "Logistic")

        rocs.append(score)
        results = results[:, 1]
        output[:, x] = results
    print()
    print("average: ", np.mean(rocs) - .04)


def pipelineKaggle(training_features, training_target, testing_features, algsToUse): #this gets me an 88
    rows, cols = testing_features.shape
    algs = ["SVM", "Logistic", "KNN", "Random Forest"]
    output = np.zeros((rows, 147))

    for x in range(147):
        Y_train = training_target[:, x]
        if (int(algsToUse[x]) == 0):
            exported_pipeline = svm.SVC(C=1.0, cache_size=200, class_weight=None,
                                        coef0=0.0, decision_function_shape='ovr', degree=3,
                                        gamma='scale', kernel='rbf', max_iter=-1, probability=True,
                                        random_state=None, shrinking=True, tol=0.001, verbose=False)
        elif (int(algsToUse[x])==1):
            exported_pipeline = LogisticRegression(C=30.0, dual=False, penalty="l2", solver='lbfgs', multi_class='multinomial', max_iter=1500, n_jobs=-1)

        elif (int(algsToUse[x])==2):
            exported_pipeline=KNeighborsClassifier(algorithm='auto', metric='minkowski', metric_params=None, n_jobs=-1,
                                                   n_neighbors=147, p=2, weights='distance')
        else:
            exported_pipeline = RandomForestClassifier(n_estimators=100, max_depth=60,
                                                       random_state=0)

        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)

        print("Genre:\t\tAlgorithm Used")
        print(x,"\t\t\t", algs[int(algsToUse[x])])
        print('\n')
        results = results[:, 1]
        output[:, x] = results

    np.savetxt("firstPlace.csv", output, delimiter=",")


def getData(training_features, training_target, testing_features, testing_target, algsToUse): #this gets me an 88
    rows, cols = testing_features.shape
    algs = ["SVM", "Logistic", "KNN", "Random Forest"]
    allscores = []
    output = np.zeros((rows, 147))
    for x in range(147):
        print(x+1)
        Y_train = training_target[:, x]
        if int(algsToUse[x]) == 0:
            exported_pipeline = svm.SVC(C=1.0, cache_size=200, class_weight=None,
                                        coef0=0.0, decision_function_shape='ovr', degree=3,
                                        gamma='scale', kernel='rbf', max_iter=-1, probability=True,
                                        random_state=None, shrinking=True, tol=0.001, verbose=False)
        elif (int(algsToUse[x])==1):
            exported_pipeline = LogisticRegression(C=30.0, dual=False, penalty="l2", solver='lbfgs', multi_class='multinomial', max_iter=1500, n_jobs=-1)

        elif (int(algsToUse[x])==2):
            exported_pipeline=KNeighborsClassifier(algorithm='auto', metric='minkowski', metric_params=None, n_jobs=-1,
                                                   n_neighbors=147, p=2, weights='distance')
        else:
            exported_pipeline = RandomForestClassifier(n_estimators=100, max_depth=60,
                                                       random_state=0)
        exported_pipeline.fit(training_features, Y_train)
        results = exported_pipeline.predict_proba(testing_features)

        score = roc_auc_score(testing_target[:, x], results[:, 1])
        print()
        allscores.append(score)
        results = results[:, 1]
        output[:, x] = results

    # print()
    average = np.mean(allscores) - .04

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(147):
        fpr[i], tpr[i], _ = roc_curve(testing_target[:, i], output[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testing_target.ravel(), output.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # fpr["macro"], tpr["macro"], _ = roc_curve(testing_target.ravel(), output.ravel())
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))

    for i in range(0,146,20):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi Classifier')
    plt.legend(loc="lower right")
    plt.show()


def pipeline(training_features, training_target, testing_features, testing_target): #this gets me an 88
    classifiers=list(range(147))
    rows, cols = testing_features.shape
    algs = ["SVM", "Logistic", "KNN", "Random Forest"]
    rocs = []
    output = np.zeros((rows, 147))
    badcols=[]
    badScores = []
    bestScore = 0
    bestAlg = ""
    for x in range(147):
        Y_train = training_target[:, x]
        bestScore=0
        for i in range(4):
            if (i == 0):
                exported_pipeline = svm.SVC(C=1.0, cache_size=200, class_weight=None,
                      coef0=0.0, decision_function_shape='ovr', degree=3,
                      gamma='scale', kernel='rbf', max_iter=-1, probability=True,
                      random_state=None, shrinking=True, tol=0.001, verbose=False)
            elif (i==1):
                exported_pipeline = LogisticRegression(C=30.0, dual=False, penalty="l2", solver='saga', multi_class='multinomial', max_iter=1500, n_jobs=-1)

            elif (i==2):
                exported_pipeline=KNeighborsClassifier(algorithm='auto', metric='minkowski', metric_params=None, n_jobs=-1,
                             n_neighbors=147, p=2, weights='distance')
            else:
                exported_pipeline = RandomForestClassifier(n_estimators=100, max_depth=60,
                                         random_state=0)

            exported_pipeline.fit(training_features, Y_train)
            results = exported_pipeline.predict_proba(testing_features)

            score = roc_auc_score(testing_target[:, x], results[:, 1])
            # print("scores for: ",x,": ",score, end="\t")

            if(score>bestScore):
                bestScore=score
                bestAlg=algs[i]
                classifiers[x] = i
        print()
        print()
        if bestScore < 0.80:
            print("SCORE:\t\t\t\tGenre:\t\tGood(0)/Bad(1)\t\tBest Algorithm")
            print(bestScore, "\t\t", x, "\t\t", 1, "\t\t\t\t\t", bestAlg)
            badScores.append(bestScore)
            badcols.append(x)
        else:
            print("SCORE:\t\t\t\tGenre:\t\tGood(0)/Bad(1)\t\tBest Algorithm")
            print(bestScore,"\t\t",x,"\t\t",0, "\t\t\t\t\t", bestAlg)

        rocs.append(bestScore)
        results = results[:, 1]
        output[:, x] = results
        # print(results)
    # writestuff(classifiers)
    # np.savetxt("multiAlg.csv", output, delimiter=",")
    print()
    print("average: ", np.mean(rocs) -.04)


def main():
    if not sys.warnoptions:
        import os, warnings
        warnings.simplefilter("ignore")  # Change the filter in this process
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

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

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # xnew = SelectKBest(k=2).fit_transform(X_train, y_train)
    # xnew2 = SelectKBest(k=2).fit_transform(X_test, y_test)


    print("data load done")
    classifiers = readClassifiers()
    # pipelineKaggle(X, y, xtest, classifiers)
    # pipeline(X_train,y_train,X_test, y_test)
    getData(X_train,y_train,X_test, y_test, classifiers)
    # logisticTester(xnew,y_train,xnew2, y_test) #8398812263597718
main()
