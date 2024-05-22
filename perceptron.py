from random import random

# INPUTS  [x0][x1][x2]
x_input = [[1, 1, 1],
           [1, 1, -1],
           [1, -1, 1],
           [1, -1, -1]]

# yD = x1 AND x2
yD_input = [-1, 1, 1, -1]

# WEIGHTS       w0       w1         w2
w_weights = [random(), random(), random()] 

alpha = 0.5

print("Initial weights: " + str(w_weights))

def activationFunction(y):
    if y < 0:
        return -1
    else:
        return 1

def train_perceptron():
    global w_weights
    all_errors = True
    
    while all_errors:
        all_errors = False
        for i in range(len(x_input)):
            print("\nCALCULATION FOR ROW: " + str(i))
            sum = 0
            for x, w in zip(x_input[i], w_weights):
                sum += x * w
            y = activationFunction(sum)
            error = yD_input[i] - y
            if error != 0:
                all_errors = True
                for j in range(len(w_weights)):
                    w_weights[j] = w_weights[j] + (alpha * error * x_input[i][j])
                print(f"Updated weights after row {i}: {w_weights}")
            else:
                print(f"No error for row {i}. Weights unchanged: {w_weights}")

train_perceptron()
print("\nFinal weights: " + str(w_weights))

def check_results():
    print("\nCHECKING FINAL RESULTS:")
    for i in range(len(x_input)):
        sum = 0
        for x, w in zip(x_input[i], w_weights):
            sum += x * w
        y = activationFunction(sum)
        print(f"Row {i} input: {x_input[i]} => Output: {y}, Expected: {yD_input[i]}")

check_results()






