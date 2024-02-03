'''
Usage-

To run classification using your perceptron implementation:
    python classification.py

To run classification using our KNN implementation:
    python classification.py -knn (or python classification.py -k)

'''
from sklearn import preprocessing
from knn import KNNClassifier
from perceptron import Perceptron
from cross_validate import cross_validate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def classification(method, train_size=0.9):
    """
    Classifies data using the model type specified in method. Default is the
    perceptron.

    Returns the model accuracy on the test data.
    """

    # Load the data. Each row represents a datapoint, consisting of all the
    # feature values and a label, which indicates the class to which the point
    # belongs. Here, each row is a patient. The features are calculated
    # from a digital image of a fine needle aspirate (FNA) of a breast mass, and
    # the label represents the patient's diagnosis, i.e. malignant or benign.
    all_data = pd.read_csv("breast_cancer_diagnostic.csv")

    # Uncomment to visualize the first 5 entries and get dataset information,
    # such as the number of entries and column names.
    # print(all_data.head)
    # all_data.info()

    # Remove the id and Unnamed:32 columns. They are not necessary for prediction.
    all_data = all_data.drop(['Unnamed: 32', 'id'], axis = 1)

    # Convert the diagnosis values M and B to numeric values, such that
    # M (malignant) = 1 and B (benign) = 0
    def convert_diagnosis(diagnosis):
        if diagnosis == "B":
            return 0
        else:
            return 1
    all_data["diagnosis"] = all_data["diagnosis"].apply(convert_diagnosis)

    # Store the features of the data
    X = np.array(all_data.iloc[:, 1:])
    # Store the labels of the data
    y = np.array(all_data["diagnosis"])

    # How much of the data do you want to use for training? The rest will be used
    # as test data for cross validation.
    train_size = 0.9
    X_train, X_test, y_train, y_test = cross_validate(X, y, train_size)

    print("-" * 30)
    if method == "KNN":
        # Set the number of neighboring points to compare each datapoint to.
        # (You will want to adjust this to optimize the accuracy of your
        # KNN Classifier. Implement optimal_k to figure out the optimal k.)
        k = 1

        # Normalize the feature data, so that values are between [0,1]. This allows
        # us to use euclidean distance as a meaningful metric across features.
        X_train = preprocessing.normalize(X_train)
        X_test = preprocessing.normalize(X_test)

        # For KNN, we want the feature function to return the value of the
        # given feature.
        def feature_func(x):
            return x

        # Initialize the KNN Classifier
        classifier = KNNClassifier([feature_func], k)

        # optimal_k explores which value of k is optimal,
        # where k is the number of neighbors used in the KNN classifier.

        # optimal_k(feature_func, X_train, X_test, y_train, y_test)
        
    else:
        is_classifier = True
        learning_rate = 0.1
        def feature_func(x):
            return x
        classifier = Perceptron([feature_func], learning_rate, is_classifier)
    
        # optimal_learning_rate(feature_func, X_train, X_test, y_train, y_test, is_classifier)
    
    # Fit the data on the train set
    print("Training {} Classifier".format(method))
    classifier.train(X_train, y_train)

    # Evaluate the model's accuracy (between 0 and 1) on the test set
    print("Testing {} Classifier".format(method))
    accuracy = classifier.evaluate(X_test, y_test)

    print("{} Model Accuracy: {:.2f}%".format(method, accuracy*100))

        
    print("-" * 30)
    return accuracy
    
def optimal_learning_rate(feature_func, X_train, X_test, y_train, y_test, is_classifier):
    optimal_learning_rate = 0
    max_accuracy = 0
    learning_rates = []
    accuracies = []
    for learning_rate in np.arange(0.1, 1, 0.1):
        learning_rates.append(learning_rate)
        classifier = Perceptron([feature_func], learning_rate, is_classifier)
        classifier.train(X_train, y_train)
        accuracy = classifier.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_learning_rate = learning_rate
    print("The optimal learning rate is {}".format(optimal_learning_rate))
    print("Model Accuracy with learning_rate={} is {:.2f}".format(optimal_learning_rate, max_accuracy))
    plt.plot(learning_rates, accuracies)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate vs Accuracy")
    plt.show()

def optimal_k(feature_func, X_train, X_test, y_train, y_test):
    """
    1) Finds the optimal value of k, where k is the number of neighbors being
    looked at during KNN.
    2) Plots the accuracy values returned by performing cross validation on
    the KNN model, with k values in the range [1, 50).
    """
    optimal_k = 0
    max_accuracy = 0
    neighbors = []
    accuracies = []
    for k in range(1, 50):
        neighbors.append(k)
        classifier = KNNClassifier([feature_func], k)
        classifier.train(X_train, y_train)
        accuracy = classifier.evaluate(X_test, y_test)
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            optimal_k = k
    print("The optimal number of neighbors is {}".format(optimal_k))
    print("Model Accuracy with k={} is {:.2f}%".format(optimal_k, max_accuracy))
    plt.figure(figsize = (10, 6))
    plt.plot(neighbors, np.multiply(accuracies,100))
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy (% Correct)')
    plt.show()
    return optimal_k

    # Printing the optimal_k and max_accuracy:
        #print("The optimal number of neighbors is {}".format(optimal_k))
        #print("Model Accuracy with k={} is {:.2f}%".format(optimal_k, max_accuracy))

    # Visualizing number of neighbors vs accuracy, if neighbors is a list of
    # your values of k and accuracies is a list of the same size,
    # corresponding to the cross validation accuracy returned for a given
    # value of k:
        # plt.figure(figsize = (10, 6))
        # plt.plot(neighbors, np.multiply(accuracies,100))
        # plt.xlabel('Number of Neighbors')
        # plt.ylabel('Accuracy (% Correct)')
        # plt.show()
        # quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Supervised Learning - Classification")
    parser.add_argument("-k", "--knn", help="Indicates to use KNN model. Otherwise, uses perceptron.",
        action="store_true")
    args = vars(parser.parse_args())
    if args["knn"]:
        method = "KNN"
    else:
        method = "Perceptron"
    classification(method)
