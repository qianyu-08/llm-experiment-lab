from src.data_loader import load_data
from src.preprocess import split_and_scale
from src.model import train_model, save_model
from src.evaluate import evaluate_model

def main():
    # 1. データ読み込み
    X, y = load_data()

    # 2. 前処理
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 3. モデル学習
    model = train_model(X_train, y_train)

    # 4. 保存
    save_model(model, scaler)

    # 5. 評価
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
