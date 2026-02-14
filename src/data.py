import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH
from src.preprocess import clean_text


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_temp, y_train, y_temp = train_test_split(
        df["review"], df["label"], test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
