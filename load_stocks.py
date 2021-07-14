import mandatory_libraries as ml
import hvplot.pandas

csv_path = ml.Path('Resources/2018_stocks_data.csv')
apple_earning_path = ml.Path('Resources/apple_earning - Sheet1.csv')
apple_launch_path = ml.Path('Resources/iphone_launch - Sheet1.csv')

stocks = ml.pd.read_csv(csv_path, header=0, index_col=0).dropna(how='all', axis=1)
apple_earning = ml.pd.read_csv(apple_earning_path, index_col=0)
iphone_launch = ml.pd.read_csv(apple_launch_path, index_col=0)

def set_stocks():
    return stocks

def set_apple_info():
    apple_info = apple_earning.join(iphone_launch, how='outer')
    apple_info.drop(['Quarter End','Estimated EPS','Actual EPS'], axis=1, inplace=True)
    return apple_info

def get_appl_data(start_date = "2016-01-01"):
    aapl_df = ml.yf.download("AAPL", start=start_date,progress=False)
    aapl_df.reset_index(level=0, inplace=True)
    return aapl_df

def get_aapl_signals(short_window = 50, long_window = 100):
    aapl_df = get_appl_data()
    # Grab just the `date` and `close` from the dataset
    signals_df = aapl_df.loc[:, ["Date", "Close"]].copy()
    # Set the `date` column as the index
    signals_df = signals_df.set_index("Date", drop=True)
    # Generate the short and long moving averages (50 and 100 days, respectively)
    signals_df["SMA50"] = signals_df["Close"].rolling(window=short_window).mean()
    signals_df["SMA100"] = signals_df["Close"].rolling(window=long_window).mean()
    signals_df["Signal"] = 0.0
    signals_df = signals_df.dropna()
    # Generate the trading signal 0 or 1,
    # where 0 is when the SMA50 is under the SMA100, and
    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
    signals_df["Signal"][short_window:] = ml.np.where(
        signals_df["SMA50"][short_window:] > signals_df["SMA100"][short_window:], 1.0, 0.0
    )

    # Calculate the points in time at which a position should be taken, 1 or -1
    signals_df["Entry/Exit"] = signals_df["Signal"].diff()
    return signals_df

def aapl_signals_plot():
    signals_df = get_aapl_signals().copy()

    # Visualize exit position relative to close price
    exit = signals_df[signals_df['Entry/Exit'] == -1.0]['Close'].hvplot.scatter(
        color='red',
        marker='v',
        size=200,
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize entry position relative to close price
    entry = signals_df[signals_df['Entry/Exit'] == 1.0]['Close'].hvplot.scatter(
        color='green',
        marker='^',
        size=200,
        legend=False,
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize close price for the investment
    security_close = signals_df[['Close']].hvplot(
        line_color='lightgray',
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Visualize moving averages
    moving_avgs = signals_df[['SMA50', 'SMA100']].hvplot(
        ylabel='Price in $',
        width=1000,
        height=400
    )

    # Overlay plots
    entry_exit_plot = security_close * moving_avgs * entry * exit
    return entry_exit_plot.opts(xaxis=None)

def analyze_aapl_signals():
    signals_df = get_aapl_signals().copy()
    # Set initial capital
    initial_capital = float(100000)

    # Set the share size
    share_size = 500

    # Take a 500 share position where the dual moving average crossover is 1 (SMA50 is greater than SMA100)
    signals_df['Position'] = share_size * signals_df['Signal']

    # Find the points in time where a 500 share position is bought or sold
    signals_df['Entry/Exit Position'] = signals_df['Position'].diff()

    # Multiply share price by entry/exit positions and get the cumulatively sum
    signals_df['Portfolio Holdings'] = signals_df['Close'] * signals_df['Entry/Exit Position'].cumsum()

    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    signals_df['Portfolio Cash'] = initial_capital - (signals_df['Close'] * signals_df['Entry/Exit Position']).cumsum()

    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']

    # Calculate the portfolio daily returns
    signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()

    # Calculate the cumulative returns
    signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1
    return signals_df

def analyze_aapl_portfolio():
    signals_df = analyze_aapl_signals()
    # Visualize exit position relative to total portfolio value
    exit = signals_df[signals_df['Entry/Exit'] == -1.0]['Portfolio Total'].hvplot.scatter(
        color='red',
        legend=False,
        ylabel='Total Portfolio Value',
        width=1000,
        height=400
    )

    # Visualize entry position relative to total portfolio value
    entry = signals_df[signals_df['Entry/Exit'] == 1.0]['Portfolio Total'].hvplot.scatter(
        color='green',
        legend=False,
        ylabel='Total Portfolio Value',
        width=1000,
        height=400
    )

    # Visualize total portoflio value for the investment
    total_portfolio_value = signals_df[['Portfolio Total']].hvplot(
        line_color='lightgray',
        ylabel='Total Portfolio Value',
        width=1000,
        height=400
    )

    # Overlay plots
    portfolio_entry_exit_plot = total_portfolio_value * entry * exit
    return portfolio_entry_exit_plot.opts(xaxis=None)

def get_aapl_portfolio_evaluation():
    signals_df = analyze_aapl_signals()
    # Prepare DataFrame for metrics
    metrics = [
        'Annual Return',
        'Cumulative Returns',
        'Annual Volatility',
        'Sharpe Ratio',
        'Sortino Ratio']

    columns = ['Backtest']

    # Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
    portfolio_evaluation_df = ml.pd.DataFrame(index=metrics, columns=columns)
    # Calculate cumulative return
    portfolio_evaluation_df.loc['Cumulative Returns'] = signals_df['Portfolio Cumulative Returns'][-1]
    # Calculate annualized return
    portfolio_evaluation_df.loc['Annual Return'] = (
            signals_df['Portfolio Daily Returns'].mean() * 252
    )
    # Calculate annual volatility
    portfolio_evaluation_df.loc['Annual Volatility'] = (
            signals_df['Portfolio Daily Returns'].std() * ml.np.sqrt(252)
    )
    # Calculate Sharpe Ratio
    portfolio_evaluation_df.loc['Sharpe Ratio'] = (
                                                          signals_df['Portfolio Daily Returns'].mean() * 252) / (
                                                          signals_df['Portfolio Daily Returns'].std() * ml.np.sqrt(252)
                                                  )
    # Calculate Downside Return
    sortino_ratio_df = signals_df[['Portfolio Daily Returns']].copy()
    sortino_ratio_df.loc[:, 'Downside Returns'] = 0
    # signal_df[signal_df['Close '] > 100]
    target = 0
    mask = sortino_ratio_df['Portfolio Daily Returns'] < target
    sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Portfolio Daily Returns'] ** 2

    # Calculate Sortino Ratio
    down_stdev = ml.np.sqrt(sortino_ratio_df['Downside Returns'].std()) * ml.np.sqrt(252)
    expected_return = sortino_ratio_df['Portfolio Daily Returns'].mean() * 252
    sortino_ratio = expected_return / down_stdev

    portfolio_evaluation_df.loc['Sortino Ratio'] = sortino_ratio
    return portfolio_evaluation_df

def aapl_trade_evaluation():
    signals_df = analyze_aapl_signals()
    # Initialize trade evaluation DataFrame with columns
    trade_evaluation_df = ml.pd.DataFrame(
        columns=[
            'Stock',
            'Entry Date',
            'Exit Date',
            'Shares',
            'Entry Share Price',
            'Exit Share Price',
            'Entry Portfolio Holding',
            'Exit Portfolio Holding',
            'Profit/Loss']
    )
    # Initialize iterative variables
    entry_date = ''
    exit_date = ''
    entry_portfolio_holding = 0
    exit_portfolio_holding = 0
    share_size = 0
    entry_share_price = 0
    exit_share_price = 0

    # Loop through signal DataFrame
    # If `Entry/Exit` is 1, set entry trade metrics
    # Else if `Entry/Exit` is -1, set exit trade metrics and calculate profit,
    # Then append the record to the trade evaluation DataFrame
    for index, row in signals_df.iterrows():
        if row['Entry/Exit'] == 1:
            entry_date = index
            entry_portfolio_holding = row['Portfolio Holdings']
            share_size = row['Entry/Exit Position']
            entry_share_price = row['Close']

        elif row['Entry/Exit'] == -1:
            exit_date = index
            exit_portfolio_holding = abs(row['Close'] * row['Entry/Exit Position'])
            exit_share_price = row['Close']
            profit_loss = exit_portfolio_holding - entry_portfolio_holding
            trade_evaluation_df = trade_evaluation_df.append(
                {
                    'Stock': 'AAPL',
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Shares': share_size,
                    'Entry Share Price': entry_share_price,
                    'Exit Share Price': exit_share_price,
                    'Entry Portfolio Holding': entry_portfolio_holding,
                    'Exit Portfolio Holding': exit_portfolio_holding,
                    'Profit/Loss': profit_loss
                },
                ignore_index=True)
    return trade_evaluation_df

def plot_closing_prices():
    signals_df = get_aapl_signals()
    price_df = signals_df[['Close', 'SMA50', 'SMA100']]
    price_chart = price_df.hvplot.line(width=1000, height=500)
    price_chart.opts(xaxis=None)
    return price_chart

def plot_portfolio_evaluation():
    result = get_aapl_portfolio_evaluation()
    result.reset_index(inplace=True)
    result_set = result.hvplot.table()
    return result_set

def plot_trade_evaluation():
    trade_evaluation_df = aapl_trade_evaluation()
    trade_evaluation_table = trade_evaluation_df.hvplot.table()
    return trade_evaluation_table

def get_stocks_pct_change():
    result = set_stocks().pct_change()
    result.dropna(inplace=True)
    return result

def get_stocks_list():
    return stocks.columns.tolist()

def set_lags():
    stocks_df = set_stocks().copy()

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

def plot_predictions_stocks():
    return get_predictions_stocks().hvplot(width=1000, height=500)

def plot_correlations(param_ax=None):
    correlation = (get_stocks_pct_change()).corr()
    # Display de correlation matrix
    ml.plt.rcParams['figure.figsize'] = (25.0, 10.0)
    ml.plt.rcParams['font.family'] = "consolas"
    ml.plt.rcParams.update({'font.size': 20})

    return ml.sns.heatmap(correlation, annot=True, vmin=-1, vmax=1, annot_kws={'size': 25}, ax=param_ax)

def listOfStocks_ToString(s=get_stocks_list()):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))

def make_interpretations():
    aapl_pct_change = ml.pd.DataFrame()
    aapl_pct_change['AAPL'] = (get_stocks_pct_change())['AAPL']
    for index, row in aapl_pct_change.iterrows():
        if(row['AAPL']>=0):
            aapl_pct_change.at[index,'Positive']=1
            aapl_pct_change.at[index,'Negative']=0
        else:
            aapl_pct_change.at[index,'Negative']=1
            aapl_pct_change.at[index,'Positive']=0

    return aapl_pct_change

def make_model_interpretations(predictions):
    predictions_pct_change=predictions.pct_change()
    predictions_pct_change.dropna(inplace=True)

    return predictions_pct_change