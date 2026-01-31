import joblib

# Iris class mapping (standard order in sklearn iris dataset)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def predict_data(X):
    """
    Predict class labels for the input data and return both class id + probabilities.
    Args:
        X: list or numpy array with shape [n_samples, 4]
    Returns:
        dict with:
          - response (int): predicted class id
          - predicted_class_name (str)
          - probabilities (dict): per-class probability
    """
    # DO NOT CHANGE THIS PATH (as per your request)
    model = joblib.load("./model/iris_model.pkl")

    # class id prediction
    y_pred = model.predict(X)
    class_id = int(y_pred[0])

    # probabilities (DecisionTreeClassifier supports this)
    proba = model.predict_proba(X)[0]  # array like [p0, p1, p2]

    return {
        "response": class_id,
        "predicted_class_name": CLASS_NAMES[class_id],
        "probabilities": {
            "setosa": float(proba[0]),
            "versicolor": float(proba[1]),
            "virginica": float(proba[2]),
        }
    }
