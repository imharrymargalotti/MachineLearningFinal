from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA
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


def writestuff(list1, list2):
    with open('DATA.csv', 'w') as f:
        writer1 = csv.writer(f)

        writer1.writerow(list1)
        writer1.writerow(list2)


def pipeline(training_features, training_target, testing_features, testing_target): #this gets me an 88

    # Training Parameters
    learning_rate = 0.001
    num_steps = 200
    batch_size = 128
    display_step = 10

    rows, cols = training_features.shape
    # Network Parameters
    num_input = rows*cols  # MNIST data input (img shape: 28*28)
    num_classes = 147  # MNIST total classes (0-9 digits)
    dropout = 0.75  # Dropout, probability to keep units

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(x, weights, biases, dropout):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for 256 MNIST test images
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256],
                                            keep_prob: 1.0}))








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

    # print(X.shape)
    # print(y.shape)

    # X, y = make_classification(n_samples=7868, n_features=20,
    #                            n_informative=10, n_classes=147,
    #                            class_sep=2.0,
    #                            random_state=0, shuffle=False)
    # print(X.shape)
    # print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("data load done")
    pipeline(X_train,y_train,X_test, y_test)
    # pipeline(X,y,xtest, None)
    # runRealData(X,y,xtest)
    # for x in range(cols):
    #     Y_train = y[:, x]
    #     Y_test = y_test[:, x]
        # pipeline(X, Y_train, xtest)
main()
