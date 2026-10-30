import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_and_split_data(file_path='adhdata.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(file_path).dropna()

    if 'Class' not in df.columns or 'ID' not in df.columns:
        raise ValueError(f"Brak kolumny 'Class' lub 'ID'. Dostępne kolumny: {df.columns}")

    X = df.drop(columns=['Class', 'ID'])
    y = df['Class']
    groups = df['ID']

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test