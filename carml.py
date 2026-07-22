import pandas as pd
import numpy as np

df = pd.read_csv("car_price_prediction_.csv")
x  = df[['Year','Engine Size', 'Mileage']].values
x = (x - x.mean(axis=0))/(x.std(axis=0))
y = df['Price'].values
def cost(x, y, w, b):
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y
    return np.sum(errors ** 2) / (2 * m)
def grad(x, y, w, b):
    m = x.shape[0]
    predictions = np.dot(x, w) + b
    errors = predictions - y
    dj_dw = np.dot(x.T, errors) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db
def grad_loop(x, y, w, b , alpha, nb_iter):
    J_history = []
    for i in range (nb_iter):
        dj_dw, dj_db = grad(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost(x, y, w, b))
        if (i+1) % 200 == 0:
            print(f"Iteration : {i+1}, Cost function : {J_history[-1]}")
    return w, b, J_history
def main():
    split = int(0.99 * x.shape[0])
    x_train = x[:split]
    y_train = y[:split]
    x_predict = x[split:]
    y_predict = y[split:]
    alpha = 1e-2
    w_in = np.zeros(x.shape[1])
    b_in = 0
    print(f"Initial cost: {cost(x_train, y_train, w_in, b_in)}") 
    nb = 5000
    w,b,J = grad_loop(x_train, y_train, w_in, b_in, alpha, nb)
    print("====================================")
    print("Comparison between prediction and real value")
    m = x_predict.shape[0]
    for i in range (m):
        price = np.dot(w,x_predict[i]) + b
        print(f"Actual price {y_predict[i]}, Prediction {price:.2f}")
if __name__ == "__main__":
    main()








