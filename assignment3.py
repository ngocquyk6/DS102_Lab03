import numpy as np
from Linear_regression import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
def organize_data(data: pd.DataFrame):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.array(X)
    y = np.array(y)

    return (X, y)

def convert_month2int(month: str) -> int:
    month_dict = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
    }
    return month_dict[month]

def convert_day2int(day: str) -> int:
    day_dict = {
        'sun': 1,
        'mon': 2,
        'tue': 3,
        'wed': 4,
        'thu': 5,
        'fri': 6,
        'sat': 7,
    }
    return day_dict[day]

def preprocess_data(df: pd.DataFrame):
    df['month'] = df['month'].map(convert_month2int)
    df['day'] = df['day'].map(convert_day2int)

    return df



if __name__ == "__main__":
    data = pd.read_csv("D:/DS102/TH/Lab03/data/forestfires.csv")
    data = preprocess_data(data)
    data = data.sample(frac=1, random_state=42, ignore_index=True)

    train_data = data.iloc[:414]
    test_data = data.iloc[414:]

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test).ravel()
    rmse = lr_model.evaluate(y_test, y_predicted)

    print("The original RMSE:", rmse)

    corr_matrix = train_data.iloc[:, :-1].corr()
    print("DMC correlations:")
    print(corr_matrix["DMC"].drop("DMC").abs().sort_values(ascending=False))

    train_data_new = train_data.drop(columns=['DMC'])
    test_data_new = test_data.drop(columns=['DMC'])

    X_train_new, y_train_new = organize_data(train_data_new)
    X_test_new, y_test_new = organize_data(test_data_new)

    lr_model_new = LinearRegression()
    lr_model_new.fit(X_train_new, y_train_new)

    y_predicted_new = lr_model_new.predict(X_test_new).ravel()
    rmse_new = lr_model_new.evaluate(y_test_new, y_predicted_new)

    print("The preprocessed RMSE:", rmse_new)
    print("RMSE difference:", abs(rmse - rmse_new))
