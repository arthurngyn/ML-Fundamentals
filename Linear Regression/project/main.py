from numpy import *
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    #code to get the specific columns 
    csv_file = r'M:\Downloads\Python Projects\ML Fundamentals\Linear Regression\project\2010.csv'
    pd.set_option('display.max_rows', None)
    data = pd.read_csv(csv_file)
    selected_columns = data [['dnce', 'pop']]

    X = selected_columns['dnce'].values.reshape(-1,1)
    Y = selected_columns['pop'].values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)
    regressor = LinearRegression().fit(X_train, Y_train)    

    y_pred = regressor.predict(X_test)

    m = regressor.coef_[0]  
    b = regressor.intercept_

    print(f"Equation of best fit: y = {m:.2f}x + {b:.2f}")
    print(f"Mean squared error: {mean_squared_error(Y_test, y_pred):.2f}")
    print(f"Coefficient of determination: {r2_score(Y_test, y_pred):.2f}")

    fig, ax = plt.subplots(ncols=2, figsize=(10,5), sharex=True, sharey=True)
    ax[0].scatter(X_train, Y_train, label = "Train data points")
    ax[0].plot(
    X_train,
    regressor.predict(X_train),
    linewidth=3,
    color="tab:orange",
    label="Model predictions",
    )
    ax[0].set(xlabel="Danceability", ylabel="Popularity", title="Train set")
    ax[0].legend()

    ax[1].scatter(X_test, Y_test, label="Test data points")
    ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
    ax[1].set(xlabel="Danceability", ylabel="Popularity", title="Test set")
    ax[1].legend()

    fig.suptitle("Linear Regression")

    plt.show()




    







