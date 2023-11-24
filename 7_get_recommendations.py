import pickle
import difflib
import numpy as np
import keras


def get_key(value, dict):
    """ searches a dictionary for a matching value and returns the key """
    for key, val in dict.items():
        if value == val:
            
            return key
    
    return "key doesn't exist"


def fuzzy_name_match(my_beers, beer_dict):
    """ searches a dictionary for best approximate matches with the values, then returns a list of keys """
    indices =[]
    all_beers = list(beer_dict.values())
    
    for beer_name in my_beers.split(','): 
        # search for closest match, with increasing flexibility if nothing found
        for step in range(100,0,-1):
            matches_list = difflib.get_close_matches(word = beer_name, 
                                                     possibilities = all_beers, 
                                                     n = 1,
                                                     cutoff = step/100)
            if matches_list:
                break
        
        # lookup index of best match and add to list of indices
        index = get_key(matches_list[0], beer_dict)
        beer_dict_name = beer_dict[index]
        indices.append(index)
        print(beer_dict_name)
    
    return indices


def load_svd_model(pkl):
    """ loads a singlular value decomposition model """
    with open(pkl, 'rb') as file:
        svd = pickle.load(file)
    
    return svd


def get_highest_scoring(array, top_k):
    """ returns a list of indices of the top_k largest values in a numpy array """
    indices = []
    for k in range(top_k):
        index = np.argmax(array)
        indices.append(index)
        array[:,index] = 0
    print(indices)
    
    return indices


def get_beer_names(indices, beer_dict):
    """ returns a list of beer name values from a dictionary using their indices as the key """
    beer_names = []
    for index in indices:
        try:
            beer_names.append(beer_dict[index])
        except:
            continue

    return beer_names


# load beer_dict
with open('cleaned data/ba_beer_dict.pkl', 'rb') as file:
    beer_dict = pickle.load(file)

# get user inputs
my_beers = 'beer1, beer2, beer3'

# find matching beers in beer_dict
indices = [600,850,400,550]    # for testing only
indices = fuzzy_name_match(my_beers, beer_dict)

# create blank one-hot vector
one_hot_vector = np.zeros((1, max(beer_dict)), dtype=int)
print(one_hot_vector)

# populate vector
one_hot_vector[:, indices] = 1  

# decompose vector
svd_object = load_svd_model('models/svd_384.pkl')
decomposed = svd_object.transform(one_hot_vector)
decomposed = np.array(decomposed)
# decomposed = np.squeeze(decomposed)
print(decomposed.shape)

# # predict missing favourites / get recommendations
# model = keras.models.load_model('models/recom1')
# y_predict = model(decomposed)
y_predict = decomposed    # for testing only

# recompose vector
prob_vec = svd_object.inverse_transform(y_predict)
print(prob_vec.shape)

# display recommended beer names
indices = get_highest_scoring(prob_vec, 4)
names = get_beer_names(indices, beer_dict)
print(names)


