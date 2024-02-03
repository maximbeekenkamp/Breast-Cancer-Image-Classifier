from sklearn.model_selection import train_test_split

def cross_validate(X, y, train_size):
    """
    Separates data into training and testing sets for cross validation.

    :param X: a numpy array containing data features
    :param y: a numpy array containing data labels
    :param train_size: a double in the range (0,1) representing the portion
                       of the data to be used as training data

    Returns:
        X_train: a numpy array containing training data features
        X_test: a numpy array containing testing data features
        y_train: a numpy array containing training data labels
        y_test: a numpy array containing testing data labels

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size = (1 - train_size), random_state = 42)
        
    return X_train, X_test, y_train, y_test
