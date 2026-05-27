import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, path="models/model.pkl"):
    joblib.dump({"model": model, "scaler": scaler}, path)
