"""
Linear regression needs to have relationship between y and x axis, if there is no clear relation ship between them, it will be hard to find the best fit line between them, so linear regression will not be very beneficial there. For example if both y and x values are getting higher with each data point, we can clearly see that there is relationship between them.

y = mx + b, we usually have x values so we can plug it right in, but we do not have m or b values. Whate are they?

m = slope of the line, b = y intercept                           

m = (mean(x) * mean(y) - mean(xy)) / (mean(x)^2 - mean(x^2))
b = mean(y) - m*mean(x)

y = (mean(x) * mean(y) - mean(xy)) / (mean(x)^2 - mean(x^2)) * x + mean(y) - m*mean(x)

It works on 2d data.

# linear regression can do classification problems if the data is linear, for example if we have 2 groups of linear data we can draw BFL through first group and then throu second one, so that we have two BFL. Then we if we want to predict group label for specfic point we can measure its distance (R^2 error) to one of the BFLs and assign it that way.
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

#xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# we can also use np function mean()


def best_fit_slope_intercept(xs, ys):
    numerator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denumerator = mean(xs)**2 - mean(xs**2)
    m = numerator/denumerator

    b = mean(ys) - m*mean(xs)
    return m, b


def best_fit_line(xs, ys):
    m, b = best_fit_slope_intercept(xs, ys)
    return [(m*x) + b for x in xs]


# best fit line name is misleading because it does not mean that the line is good, how we determine accuracy? by rsquared we want to check our Coefficient of determination - Współczynnik determinacji. The error is the distance between datapoint and best fit line, we square it mainly because the distance sometimes may be negative and we want to deal with positive values and we also do it to penalize outliers
# R^2 = 1 - (SEy^)/SEmean(y) | (y^ = best fit line, sometimes called y^ hat, SE = square error, so basically 1 - square error of best fit line divided by square error of mean(y) )
# we want R^2 to be high

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coef_of_det(ys_orig, ys_line):
    y_mean_line = mean(ys_orig)
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_y_mean)

# anything above 0 means that regression line is more accurate


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1  # first value for y
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit_slope_intercept(xs, ys)

regression_line = best_fit_line(xs, ys)

r_squared = coef_of_det(ys, regression_line)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
print(xs, ys)
# as we decrease variance the r squared is getting higher, which is correct because the higher the variance the less linear the data gets. The lesser the variance the more linear it is, so our r squared is getting higher because error SEy^ is lesser than error SEmean(y). 1 - 2/10 = 0.8  etc. if we turn off correlation completly we can see that the data is not linear at all and our R^2 is nearly 0. This tell us that the data is not linear so linear regression will not work here.
