from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from palmerpenguins import load_penguins
import pandas as pd


def train_logistic_with_lambda(X_train, y_train, X_test, y_test, lambda_value):
    C_value = 1 / lambda_value if lambda_value > 0 else 1e12  # Avoid division by zero
    
    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',  # supports L1
        C=C_value,
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    non_zero_features = np.sum(model.coef_ != 0)

    return acc, non_zero_features, model



def train_gosdt_with_lambda(X_train, y_train, X_test, y_test, lambda_value=0.01):
    
    from gosdt import GOSDT
    from gosdt.model import Model

    config = {
        "regularization": lambda_value,
        "depth_budget": 5,
        "time_limit": 10,
        "precision": 1e-6,
        "threads": 4
    }

    model = GOSDT(config)
    model.fit(X_train.values, y_train.values)

    prediction = model.predict(X_test.values)
    acc = accuracy_score(y_test, prediction)

    model_info = model.export()
    num_leaves = model_info["nodes"].count("leaf")

    return acc, num_leaves, model_info

def DecisionTreeClassifier(max_depth_n):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from palmerpenguins import load_penguins
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    import os

    # loading the dataset
    df = load_penguins().dropna()

    # Features and target
    X = pd.get_dummies(df.drop(columns=['species']))
    y = df['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth = max_depth_n, random_state=0)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    num_leaves = clf.get_n_leaves()

    print(f"Accuracy: {acc:.3f}")
    print(f"Number of Leaves: {num_leaves}")

     # Plot tree and save image
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, ax=ax)
    
    img_path = 'media/tree.png'
    fig.savefig(img_path)
    plt.close(fig)
    print(f"[INFO] Decision tree saved at: {img_path}")


    # gosdt
    lambda_value = 0.01
    acc, leaves, info = train_gosdt_with_lambda(X_train, y_train, X_test, y_test, lambda_value)
    
    print(f"Lambda: {lambda_value}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of leaves: {leaves}")

    #logistic regression
    from sklearn.linear_model import LogisticRegression
    lambda_value = 0.1
    acc, used_features, model = train_logistic_with_lambda(X_train, y_train, X_test, y_test, lambda_value)

    print(f"Lambda: {lambda_value}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Number of used features: {used_features}")
    return


def index(request):
    
    DecisionTreeClassifier(3)
    return HttpResponse("working on the console")
    
