import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def display_confusion(c_matrix):
    """
    Displays the confusion matrix using matrix show
    Args:
        c_matrix: square confusion matrix, shape (num_classes, num_classes)
    """
    _, ax = plt.subplots()
    ax.matshow(c_matrix, cmap=plt.cm.Blues)  # type: ignore
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[0]):
            ax.text(i, j, str(c_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    plt.show()


def logreg(x, y):
    """
    Logistic Regression Model and classification report.

    :param x: X 
    :param y: Description
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(class_weight='balanced')
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)

    # Metrics
    print('Confusion matrix')
    display_confusion(confusion_matrix(y_test, y_pred))

    print("Accuracy")
    print(accuracy_score(y_test, y_pred))

    print("Classification report")
    print(classification_report(y_test, y_pred))

    feature_names = x_train.columns.to_list()
    coef = lr.coef_[0]

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    })

    return coef_df
