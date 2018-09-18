'''
This script implements a simple linear regression algorithm from scratch without using
modules like sklearn. The idea is to predict a variable y using a straight line with the
formula y = mx+b. Therefore the script must be capable of calculating the slope m and the
constant b after the input dataset x is loaded.
'''
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#This is our input data and it is defined on a numpy array of floats
#x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#y = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

#This function creates a random dataset of xs and ys
#it recieves the number of instances
#the variance represents how variable the points in y are
#the step represents how separate the points in y are
#the correlation defines if the line has a positive or negative slope
#the x are the number of instances assign to y in ascending order
def create_dataset(data_points_number, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(data_points_number):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation =='neg':
            val -= step
    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

#this plots the data with matplotlib and the scatter() function
#shows it as dots and not a line.
    #plt.scatter(x,y)
    #plt.show()

#This function calculates the best slope m
#and the best fit line (constant b)
def best_fit_slope_and_fit_line(x,y):
    m = ( ((mean(x)*mean(y))-(mean(x*y))) /
          ((mean(x)**2) - (mean(x**2))) )
    b = mean(y) - (m*mean(x))
    return m, b

#This function calculates the squared error of the current line by
#obtaining the difference between the test line and the original points in Y
def squared_error(y_original, y_line):
    return sum((y_line-y_original)**2)

#This functiion calculates the squared error mean
def coefficient_of_determination(y_original, y_line):
    #print('starting calculation')
    y_mean_line = mean(y_original)
    #print('calculating sq error regr')
    squared_error_regr = squared_error(y_original, y_line)
    #print('calculating sq errir y mean')
    squared_error_y_mean = squared_error(y_original, y_mean_line)
    #print('Returning calculation')
    return 1 - (squared_error_regr/squared_error_y_mean)
    

#if the variance is lower the squared error should increase
#since the data points present a more linear behavior
x, y = create_dataset(40, 10, 2, correlation='pos')

m,b = best_fit_slope_and_fit_line(x,y)


print('Best slope is: ',m,'\nBest fit line - Constant is: ',b)

#Now, since we have calculated all the variables needed for the Y=mx+b equation
#We need to read the input values in a for since it is a list of values
regression_line = np.array([(m*xs)+b for xs in x], dtype=np.float64)

#If we want to predict a new value of x, all we have to do is apply the model's equation
predict_x = 15
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(y, regression_line)
print('Squared error mean: ',r_squared)

print('When x = ', predict_x,'y = ', predict_y)

#Graph the points of the dataset
plt.scatter(x,y)
#Graph the new point to predict
plt.scatter(predict_x, predict_y, color='r')
#Graph the regression line obtained
plt.plot(x, regression_line)
plt.show()

