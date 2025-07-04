from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

def get_model_instance(model_name, hyperparameters):
    if model_name == "logistic_regression":
        return LogisticRegression(**hyperparameters)
    elif model_name == "svm":
        return SVC(**hyperparameters)
    elif model_name == "random_forest_classifier":
        return RandomForestClassifier(**hyperparameters)
    elif model_name == "linear_regression":
        return LinearRegression(**hyperparameters)
    elif model_name == "random_forest_regressor":
        return RandomForestRegressor(**hyperparameters)
    elif model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
