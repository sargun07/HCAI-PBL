from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def get_classification_model(name, params):

    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "svm":
        return LinearSVC(**params)
    elif name == "random_forest_classifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

    
