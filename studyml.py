import numpy as np
import matplotlib.pyplot as plt

def gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        y_bar = w * x[i] + b
        dj_db += (y_bar - y[i])
        dj_dw += (y_bar - y[i]) * x[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def cost_func(x, y, w, b):
    cost = 0 
    m = x.shape[0]
    for i in range(m):
        f_x = w * x[i] + b 
        cost += (f_x - y[i]) ** 2 
    cost /= (2*m)
    return cost

def gradient_loop(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        if i<100000:      
            J_history.append(cost_func(x, y, w , b))
            p_history.append([w,b])
        if i % 1000 == 0:
            print(f"iteration nb: {i}, cost function {J_history[-1]:0.3e}, w = {w:0.3e}, b = {b:0.3e} ")
    return w, b
def main():
    x_train = np.array([1.5, 2.0, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0])
    y_train = np.array([42, 48, 55, 63, 68, 74, 80, 85, 90, 95])
    alpha = 1e-2
    num_iters = 10000
    w_in = 0
    b_in = 1
    w, b = gradient_loop(x_train, y_train, w_in, b_in, alpha, num_iters)
    print(f"w = {w} , b = {b}")
    x = float(input("how many hours did you study? "))
    y = w * x + b 
    print(f"Your predicted score is {int(y)}")
    plt.scatter(x_train, y_train, label ='Training data')  # this plots your data points
    plt.plot(x_train, w*x_train + b, label = 'Regression line')
    plt.xlabel('Hours studied')
    plt.ylabel('Score', rotation = 90, ha = 'right')
    plt.title('ML predictions of final grade based on hours studied')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()










    

        


