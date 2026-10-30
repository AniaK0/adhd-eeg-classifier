import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path='adhdata.csv'):
    df = pd.read_csv(file_path).dropna()

    class_counts = df['Class'].value_counts()
    print("Balans klas:")
    print(class_counts)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Class', hue='Class', palette='Set2', legend=False)
    plt.title('Balans klas w zbiorze danych (ADHD vs Control)')
    plt.xlabel('Klasa (Diagnoza)')
    plt.ylabel('Liczba próbek (Sygnałów)')
    plt.tight_layout()
    plt.savefig('eda_class_balance.png')