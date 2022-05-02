'''
The distance between poinst are calculated with Euclidean Distance.

the formula is in jupyter notebook
'''
from math import sqrt
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings

style.use('fivethirtyeight')


#p1 = [1, 3]
#p2 = [2, 5]

#euc_distance = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
# we use numpy because it is faster

data = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[5, 6], [6, 5], [7, 8]]}

new_features = [6, 6]

for d in data:
    for dd in data[d]:
        plt.scatter(dd[0], dd[1], s=100, color=d)

plt.scatter(new_features[0], new_features[1], s=100)
plt.show()


def knb(data, predict, k=3):
    if len(data) >= 3:
        warnings.warn('K number is set to lesser than the number of groups')

    distances = []

    for group in data:  # for every data group in data, like 'r' group, 'k' group
        # for every set of points (features) of that data group
        for features in data[group]:
            euc_distance = np.linalg.norm(
                np.array(features) - np.array(predict))  # calculate euc distance with linealgnorm, reminder that we can subtract np arrays, it works like vector subtraction, the result is np array.
            distances.append([euc_distance, group])  # appending to distances

    # we are intrested only in group labels, we sort the distances by values, so the lower values appear at the front of list, :k because we are intrested in 3 votes( 3 nearest points/features), which have the smallest distance to predicting feature
    votes = [i[1] for i in sorted(distances)[:k]]
    # using counter to count apperances, most common 1
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = knb(data, new_features, k=3)


print(np.linalg.norm(np.array([1, 2]) - np.array([4, 5])))
