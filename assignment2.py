from Linear_regression import LinearRegression
import pandas as pd
import numpy as np

def organize_data(data: pd.DataFrame):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = np.array(X)
    y = np.array(y)

    return (X,y)

def convert_month2int(month: str) -> int:
    month_dict = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12
    }
    return month_dict[month]

def convert_date2int(date: str):
    date_dict = {
        "sun": 1,
        "mon": 2,
        "tue": 3,
        "wed": 4,
        "thu": 5,
        "fri": 6,
        "sat": 7 
    }
    return date_dict[date]

def preprocess_data(df: pd.DataFrame):
    df["month"] = df["month"].map(convert_month2int) 
    df["day"] = df["day"].map(convert_date2int)

    continuous_variables = ["FFMC", "DMC", "DC", "ISI", "temp", "wind", "RH", "rain"]
    for variable in continuous_variables:
        df[variable] = (df[variable] - df[variable].mean()) / df[variable].std()
    return df

if __name__ == "__main__":
    data = pd.read_csv("D:/DS102/TH/Lab03/data/forestfires.csv")
    data = preprocess_data(data)
    data = data.sample(frac = 1,random_state=42, ignore_index=True)

    train_data = data.iloc[:414]
    test_data = data.iloc[414:]

    X_train, y_train = organize_data(train_data)
    X_test, y_test = organize_data(test_data)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_predicted = lr_model.predict(X_test).ravel()
    rmse = lr_model.evaluate(y_test, y_predicted)

    print("The rmse is: ", rmse)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    lr_model_std = LinearRegression()
    lr_model_std.fit(X_train_std, y_train)

    y_predicted_std = lr_model_std.predict(X_test_std).ravel()
    rmse_std = lr_model_std.evaluate(y_test, y_predicted_std)

    print("The standardized rmse is: ", rmse_std)
    print("Difference: ", abs(rmse - rmse_std))

    print("Mean: ", X_train_std.mean(axis=0))
    print("Variance: ", X_train_std.var(axis=0))
