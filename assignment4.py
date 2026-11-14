
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearregression
from Linear_regression import LinearRegression


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

    model_1 = LinearRegression()
    model_1.fit(X_train, y_train)

    y_predicted_1 = model_1.predict(X_test).ravel()
    rmse_1 = model_1.evaluate(y_test, y_predicted_1)

    model_2 = SklearnLinearregression(fit_intercept=False)
    model_2.fit(X_train, y_train)

    y_predicted_2 = model_2.predict(X_test)
    rmse_2 = np.sqrt(((y_test - y_predicted_2) ** 2).mean())

    print("RMSE - NumPy:", rmse_1)
    print("RMSE - Scikit-learn:", rmse_2)
    print("Difference:", abs(rmse_1 - rmse_2))

