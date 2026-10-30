import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_evaluate(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=20, max_depth=15, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\nWyniki Modeli:")
    print(f"Dokładność (Decision Tree): {accuracy_score(y_test, dt_pred):.2f}")
    print(f"Dokładność (Random Forest): {accuracy_score(y_test, rf_pred):.2f}")

    print("\n-------- Raport klasyfikacji (Random Forest) --------")
    print(classification_report(y_test, rf_pred))

    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=rf_model.classes_,
                yticklabels=rf_model.classes_)
    plt.title('Confusion Matrix - Random Forest (ADHD)')
    plt.xlabel('Przewidywana klasa')
    plt.ylabel('Rzeczywista klasa')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    return rf_model