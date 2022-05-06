'''
The distance between poinst are calculated with Euclidean Distance.

the formula is in jupyter notebook
'''
from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

#p1 = [1, 3]
#p2 = [2, 5]

#euc_distance = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
# we use numpy because it is faster


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
    # counter returns for example [('r', 12)]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


df = pd.read_csv('./data/breast-cancer-wisconsin.data', names=['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
                                                               'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei', 'bland_chrom', 'norm_nucleoli', 'mistoses', 'class'])


df.replace('?', -99999, inplace=True)
df.drop('id', axis=1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]  # up to last 20%
test_data = full_data[-int(test_size*len(full_data)):]  # last 20%

for i in train_data:
    train_set[i[-1]].append(i[:-1])  # train set is a dictionary with keys either 2 or 4 which is the same name as the cancer class, so train_set[i[-1]], the i[-1] is the last index of our dataset which is exactly cancer class (same as key of our dictionary) - train_set[2] gives us empty list -> [].append(). This way we get value for specific key and this value is our empty list which we are populating by .append() and passing every value till the last :-1 index - our cancer class which we used as a key for a dictionary.

for i in test_data:
    test_set[i[-1]].append(i[:-1])


correct = 0
total = 0

for group in test_set:  # for every group of test_set (group is either 2 or 4)
    # for data (lists including or features/values/datapoints) in that test_group (2 or 4)
    for data in test_set[group]:
        # knb with with train_set and predict as data (list)
        # result from our knb algorithm which returns in this situation 2 or 4 depending on features
        vote, confidence = knb(train_set, predict=data, k=5)
        if group == vote:  # if the group matches our return from knb we got 1 point correct
            correct += 1
        else:
            print(
                f'Percent of votes in {group, vote} that were incorrect: {confidence * 100} %')
        total += 1  # total is always counting up 1

print('accuracy', correct/total)
# knb can give us not only accuracy but also confidence which is related to votes
# knb works on linear and non-linear data
