from eda import perform_eda
from loader import load_and_split_data
from train import train_and_evaluate

def main():
    perform_eda('adhdata.csv')
    X_train, X_test, y_train, y_test = load_and_split_data('adhdata.csv')
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()