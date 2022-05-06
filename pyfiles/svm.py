import sys
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

# scuffed version of SVM
# pretty hard ngl


class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):  # The data is the data we intend to train against / optimize with
        # train, find vector w and b value
        # svm is all about optimization
        self.data = data
        # { ||w||: [w, b] }
        #  When we're all done optimizing, we'll choose the values of w and b for whichever one in the dictionary has the lowest key value (which is ||w||)
        opt_dict = {}
        # our intention there is to make sure we check every version of the vector possible.
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        # the closer our support vectors are to 1 the better
        # 1.02, 1.1 etc is still valid in some cases

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense
                      self.max_feature_value * 0.001]

        # super expensive
        # make larger steps for b than we use for w, since we care far more about w precision than b
        b_range_multiple = 5
        b_multiple = 5
        # we can use the same stepping idea for b but we are not doing it right now
        latest_optimum = self.max_feature_value * 10
        # As we step down our w vector, we'll test that vector in our constraint function, finding the largest b
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # convex problem
            optimized = False
            while not optimized:
                # we can thread it, but we cant thread the steps themselves
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation  # we are applying transform np array to original w np array
                        found_option = True
                        # weakest link in the SVM fundemantally, we have to run this function in all data to make sure it fits
                        # SMO attempts to fix this a bit
                        # yi(xi . w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    found_option = False

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('optimized step')
                else:
                    # w = [5,5], step = 1, w - step = [4, 4] because of how np.arrays work
                    w = w - step
            # sorting the keys (magnitudes)
            # { ||w||: [w, b] }
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # { ||w||: [w, b] }, the w in [w , b] is a vector like [5, 5]
            # so we take for example [ [5,5], 7 ] [0] for the w vector
            latest_optimum = opt_choice[0][0] + step * 2
            # and [0] for first value of w vector

    def predict(self, features):
        # sign (x_i . w + b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,
                            marker='*', c=self.colors[classification])
        return classification

    # this has no effect on SVM at all, it is only for us to visualize the hyperlane etc. The svm does not really care about hyperlane, it only needs the sign of the x_i . w + b to make predictions
    def visualize(self):
        for i in data_dict:
            for x in data_dict[i]:
                self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])

        #hyperplane is x.w+b
        # v = x . w + b
        # psv = 1 (positive support vectors)
        # nsv = -1 (negative support vectors)
        # dec = 0 (decision boundary)

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9,
                     self.max_feature_value * 1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # w.x+b =1
        # positive support hyperlane
        # it is gonna be a scalar value, it is going to be our "y" on x, y 2d graph
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # w.x+b = -1
        # negative support hyperlane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # w.x+b =1
        # decision boundry
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 5]]),
             1: np.array([[5, 1], [5, -1], [7, 3]])}


clf = SVM()

clf.fit(data_dict)

predict = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]

for p in predict:
    clf.predict(p)


clf.visualize()
