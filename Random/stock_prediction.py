import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

"""
    //  Function Name : download_data       
    //  Description :  Downloads historical stock data for the given ticker.
    //  Input :        ticker (str), start (str), end (str) 
    //  Output :       DataFrame containing stock data
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    
    data = yf.download(ticker, start=start, end=end)
    return data

"""
    //  Function Name : preprocess_data       
    //  Description :  Preprocess the stock data by filling missing values and normalizing.
    //  Input :        data (DataFrame) 
    //  Output :       DataFrame after preprocessing
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    
    data.fillna(method='ffill', inplace=True)
    
    # Feature normalization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaled_data

    return data

"""
    //  Function Name : feature_engineering       
    //  Description :  Adds technical indicators as features to the stock data.
    //  Input :        data (DataFrame) 
    //  Output :       DataFrame with additional features
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    
    # Moving Average as a feature
    data['Moving_Average'] = data['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)

    return data

"""
    //  Function Name : train_model       
    //  Description :  Trains a Random Forest model using the provided features and target.
    //  Input :        X (DataFrame), y (Series)
    //  Output :       Trained model
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def train_model(X: pd.DataFrame, y: pd.Series):

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

"""
    //  Function Name : merge_sort       
    //  Description :  Implements the merge sort algorithm to sort an array.
    //  Input :        arr (list) 
    //  Output :       None (sorted in place)
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def merge_sort(arr):
    
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

"""
    //  Function Name : evaluate_model       
    //  Description :  Evaluates the trained model using Mean Squared Error and Mean Absolute Error.
    //  Input :        model, X_test (DataFrame), y_test (Series)
    //  Output :       None (prints evaluation metrics and shows an enhanced graph)
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    
    # Set up a cleaner visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot actual prices
    ax1.plot(y_test.values, color='blue', label='Actual Prices', lw=2)
    ax1.set_title('Actual Stock Prices', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.grid(True)
    
    # Plot predicted prices
    ax2.plot(y_pred, color='green', label='Predicted Prices', lw=2)
    ax2.set_title('Predicted Stock Prices', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.grid(True)
    
    # Display both subplots
    plt.tight_layout()
    plt.show()

"""
    //  Function Name : main       
    //  Description :  Entry point of the application. Calls other functions to execute the stock prediction workflow.
    //  Input :        None 
    //  Output :       None
    //  Author :       Rushikesh Ratnakar Waghmare 
    //  Date :         05/10/2024
"""
def main():

    # Step 1: Download data
    ticker = 'AAPL'
    data = download_data(ticker, start='2020-01-01', end='2024-01-01')

    # Step 2: Preprocess data
    data = preprocess_data(data)

    # Step 3: Feature engineering
    data = feature_engineering(data)

    # Step 4: Prepare data for model training
    X = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'Moving_Average']]
    y = data['Close']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Train model
    model = train_model(X_train, y_train)

    # Step 6: Evaluate model
    evaluate_model(model, X_test, y_test)

    # Step 7: Sort data for demonstration using merge sort
    sample_data = [3, 6, 1, 5, 2, 4]
    merge_sort(sample_data)
    print("Sorted data:", sample_data)


if __name__ == "__main__":
    main()
