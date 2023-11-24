import pickle
import numpy as np
import copy
import math

# load BeerAdvocate data
with open('cleaned data/ba_reviews_dict.pkl', 'rb') as file:
    reviews_dict = pickle.load(file)
with open('cleaned data/ba_beer_dict.pkl', 'rb') as file:
    beer_dict = pickle.load(file)

# remove reviewers with <4 or >30 favourites
new_dict = copy.deepcopy(reviews_dict)
for key in new_dict:
    if len(new_dict[key]) < 4 or len(new_dict[key]) > 30:
        del reviews_dict[key]

# check outputs
print(len(new_dict)-len(reviews_dict), 'deleted')
print(len(reviews_dict), 'remaining')

# create blank one-hot vector
data = np.zeros((len(reviews_dict), max(beer_dict)), dtype=bool)
    
# populate array with data from reviews_dict
for index, key in enumerate(reviews_dict):
    favourites = reviews_dict[key]
    for beer_id in favourites:
        data[index][beer_id] = 1
    
# separate 500 randomly chosen rows for testing dataset
indices_take = np.random.choice(data.shape[0], 500, replace=False)
indices_keep = [index for index in range(0,data.shape[0]) if index not in indices_take]
test_data = data[indices_take]
data = data[indices_keep]
print('test:', test_data.shape)
print('train:', data.shape)

# create training dataset
x_train = copy.deepcopy(data)
y_train = np.zeros_like(x_train)
for row, values in enumerate(x_train):
    # select a small portion of reviewers favourite beers
    hits = [index for index, value in enumerate(values) if value == 1]
    num_to_pick = math.ceil(len(hits)/6.66)
    indices = np.random.choice(hits, num_to_pick, replace=False)
    # remove selected beers from x_data and add to y_data
    x_train[row][indices] = 0
    y_train[row][indices] = 1
print('train x,y:', x_train.shape, y_train.shape)

# augment training dataset with repeat sampling
MULTIPLES = 3
for mult in range(MULTIPLES):
    # clone original data
    x = copy.deepcopy(data)
    y = np.zeros_like(x)
    for row, values in enumerate(x):
        # select a different small portion of reviewers favourite beers
        hits = [index for index, value in enumerate(values) if value == 1]
        num_to_pick = math.ceil(len(hits)/6.66)
        indices = np.random.choice(hits, num_to_pick, replace=False)
        # remove selected beers from x_data and add to y_data
        x[row][indices] = 0
        y[row][indices] = 1
    # add to previous training data
    x_train = np.concatenate((x_train, x))
    y_train = np.concatenate((y_train, y))
    print('train x,y:', x_train.shape, y_train.shape)

# augment training dataset with extra sampling of frequent reviewers
LIMIT = 12
MULTIPLES = 5
# clone original data
x = copy.deepcopy(data)
y = np.zeros_like(x)
indices_keep = []
for row_index, row_values in enumerate(x):
    # find reviewers with more than LIMIT favourite beers
    favourites = [value for value in row_values if value == 1]
    if len(favourites) >= LIMIT:
        indices_keep.append(row_index)
# remove reviewers with less than LIMIT favourite beers from working dataset
x = x[indices_keep]
y = y[indices_keep]
print(len(x), 'reviewers with >', LIMIT, 'samples found')
# repeat random sampling of working dataset several times
for mult in range(MULTIPLES):
    for row, values in enumerate(x):
        # select a small portion of reviewers favourite beers
        hits = [index for index, value in enumerate(values) if value == 1]
        num_to_pick = math.ceil(len(hits)/6.66)
        indices = np.random.choice(hits, num_to_pick, replace=False)
        # remove selected beers from x_data and add to y_data
        x[row][indices] = 0
        y[row][indices] = 1
    # add to previous training data
    x_train = np.concatenate((x_train, x))
    y_train = np.concatenate((y_train, y))
    print('train x,y:', x_train.shape, y_train.shape)

# add reviewers with only three favourites to training dataset
with open('cleaned data/ba_reviews_dict.pkl', 'rb') as file:
    reviews_dict = pickle.load(file)
new_dict = copy.deepcopy(reviews_dict)
for key in new_dict:
    if len(new_dict[key]) != 3:
        del reviews_dict[key]
print(len(reviews_dict), 'reviewers added')
# create array add populate with data from reviews_dict
xx = np.zeros((len(reviews_dict), max(beer_dict)), dtype=bool)
for index, key in enumerate(reviews_dict):
    favourites = reviews_dict[key]
    for beer_id in favourites:
        xx[index][beer_id] = 1
# create dataset by removing one beer, repeated twice
for mult in range(2):
    x = copy.deepcopy(xx)
    y = np.zeros_like(x)
    for row, values in enumerate(x):
        # select one favourite beer
        hits = [index for index, value in enumerate(values) if value == 1]
        num_to_pick = 1
        indices = np.random.choice(hits, num_to_pick, replace=False)
        # remove selected beer from x_data and add to y_data
        x[row][indices] = 0
        y[row][indices] = 1
    # add to previous training data
    x_train = np.concatenate((x_train, x))
    y_train = np.concatenate((y_train, y))
    print('train x,y:', x_train.shape, y_train.shape)


# create testing dataset
x_test = test_data
y_test = np.zeros_like(x_test)
for row, values in enumerate(x_test):
    # select a different small portion of reviewers favourite beers
    hits = [index for index, value in enumerate(values) if value == 1]
    num_to_pick = math.ceil(len(hits)/6.66)
    indices = np.random.choice(hits, num_to_pick, replace=False)
    # remove selected beers from x_data and add to y_data
    x_test[row][indices] = 0
    y_test[row][indices] = 1
print('test x,y:', x_test.shape, y_test.shape)

# save datasets
with open('training data/x_train.pkl', 'wb') as file:
    pickle.dump(x_train, file)
with open('training data/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)
with open('training data/x_test.pkl', 'wb') as file:
    pickle.dump(x_test, file)
with open('training data/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)