import mandatory_libraries as ml

csv_path = ml.Path('Resources/Stocks - Sheet1.csv')
apple_earning_path = ml.Path('Resources/apple_earning - Sheet1.csv')
apple_launch_path = ml.Path('Resources/iphone_launch - Sheet1.csv')

stocks = ml.pd.read_csv(csv_path, header=0, index_col=0).dropna(how='all', axis=1)
apple_earning = ml.pd.read_csv(apple_earning_path, index_col=0)
iphone_launch = ml.pd.read_csv(apple_launch_path, index_col=0)

def set_apple_info():
    apple_info = apple_earning.join(iphone_launch, how='outer')
    apple_info.drop(['Quarter End','Estimated EPS','Actual EPS'], axis=1, inplace=True)
    return apple_info

def set_stocks():
    stocks_df = ml.pd.DataFrame()
    for i in stocks.index:
        stocks_df[i] = ml.get_data(i)['close']
    stocks_df.index = ml.pd.to_datetime(stocks_df.index)
    stocks_df.dropna(inplace=True)
    return stocks_df

def get_stocks_pct_change():
    result = set_stocks().pct_change()
    result.dropna(inplace=True)
    return result

def get_stocks_list():
    return stocks.index.tolist()

def set_lags():
    stocks_df = set_stocks()
    for stock in get_stocks_list():
        stocks_df[stock+' Lag'] = stocks_df[stock].shift()
        if (stock!='AAPL'):
            stocks_df.drop(columns=stock, inplace=True)

    stocks_df.dropna(inplace=True)
    return stocks_df

def combine_lags_appleInfo():
    combined_df = set_lags()
    combined_df = combined_df.join(set_apple_info(), how='outer')
    combined_df = combined_df[combined_df['AAPL'].notna()]
    combined_df['iphone'].fillna(0, inplace=True)
    combined_df['Result'].fillna(1, inplace=True)
    return combined_df

def window_data(df, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i: (i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return ml.np.array(X), ml.np.array(y).reshape(-1, 1)

def getSampleValues_stocks():
    # Creating the features (X) and target (y) data using the window_data() function.
    window_size = 5
    feature_column = 0
    target_column = 0
    X, y = window_data(combine_lags_appleInfo(), window_size, feature_column, target_column)
    return X,y

def getTestingData_stocks(percent_training=70/100):
    # Use 70% of the data for training and the remainder for testing
    X, y = getSampleValues_stocks()
    split = int(percent_training * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test

def scale_TestingData_stocks():
    X, y = getSampleValues_stocks()
    X_train, X_test, y_train, y_test = getTestingData_stocks()
    scaler = ml.MinMaxScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(y)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    return X_train, X_test, y_train, y_test, scaler

def reshapeFeatures_stocks():
    X_train, X_test, y_train, y_test, scaler = scale_TestingData_stocks()
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # print(f"X_train sample values:\n{X_train[:5]} \n")
    # print(f"X_test sample values:\n{X_test[:5]}")
    return X_train, X_test, y_train, y_test, scaler

def set_LSTM_RNN_stocks_model():
    X_train, X_test, y_train, y_test, scaler = reshapeFeatures_stocks()
    model = ml.Sequential()

    number_units = 5
    #     dropout_fraction = 0.2

    # Layer 1
    model.add(ml.LSTM(
        units=number_units,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1))
    )
    #     model.add(ml.Dropout(dropout_fraction))
    #     # Layer 2
    #     model.add(ml.LSTM(units=number_units, return_sequences=True))
    #     model.add(ml.Dropout(dropout_fraction))
    #     # Layer 3
    #     model.add(ml.LSTM(units=number_units))
    #     model.add(ml.Dropout(dropout_fraction))
    # Output layer
    model.add(ml.Flatten())
    model.add(ml.Dense(1))

    print('step 1. compile the model')
    #     model.compile(optimizer="adam", loss="mean_absolute_percentage_error"), X_train, X_test, y_train, y_test, scaler
    model.compile(optimizer="adam", loss="mean_squared_error"), X_train, X_test, y_train, y_test, scaler

    print('step 2. model summary')
    model.summary()

    print('step 3. Train the model')
    model.fit(X_train, y_train, epochs=35, shuffle=False, batch_size=1, verbose=1)

    print('step 4. Evaluate the model')
    model.evaluate(X_test, y_test)
    return model, X_train, X_test, y_train, y_test, scaler

def get_predictions_stocks():
    model, X_train, X_test, y_train, y_test, scaler = set_LSTM_RNN_stocks_model()
    # Make some predictions
    predicted = model.predict(X_test)

    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    # Create a DataFrame of Real and Predicted values
    stocks_predict = ml.pd.DataFrame({
        "Real": real_prices.ravel(),
        "Predicted": predicted_prices.ravel()
    }, index=set_stocks().index[-len(real_prices):])

    return stocks_predict