from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
import numpy as np
import pickle
import copy

# load beeradvocate data
with open('cleaned data/ba_reviews_dict.pkl', 'rb') as file:
    reviews_dict = pickle.load(file)
with open('cleaned data/ba_beer_dict.pkl', 'rb') as file:
    beer_dict = pickle.load(file)

# remove reviewers with <3 or >30, favourites
new_dict = copy.deepcopy(reviews_dict)
for key in new_dict:
    if len(new_dict[key]) < 3 or len(new_dict[key]) > 30:
        del reviews_dict[key]

# create blank array
data = np.zeros((len(reviews_dict), max(beer_dict)), dtype=bool)
print(data.shape)
    
# populate array with data from reviews_dict
for index, key in enumerate(reviews_dict):
    favourites = reviews_dict[key]
    for beer_id in favourites:
        data[index][beer_id] = 1

print(data.shape)


# define singular value decomposition object and fit on dataset
svd = TruncatedSVD(n_components=1024)
svd.fit(data)

# save svd model
with open('models/svd_1024.pkl', 'wb') as file:
    pickle.dump(svd, file)


# create umapper object then fit on dataset
umapper = umap.UMAP(n_neighbors=15,          
                    n_components=384,        
                    set_op_mix_ratio=1,             
                    metric='euclidean',
                    n_epochs=500,
                    low_memory=True)     
umapper.fit(data)

# save umap model
with open('models/umap_384_15nn.pkl', 'wb') as file:
    pickle.dump(umapper, file)


# https://umap-learn.readthedocs.io/en/latest/parametric_umap.html#

""" @article{sainburg2021parametric,
    title={Parametric UMAP Embeddings for Representation and Semisupervised Learning},
    author={Sainburg, Tim and McInnes, Leland and Gentner, Timothy Q},
    journal={Neural Computation},
    volume={33},
    number={11},
    pages={2881--2907},
    year={2021},
    publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
    } """

dims = (max(beer_dict),)
n_components = 384

encoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=dims),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=2000, activation="relu"),
    tf.keras.layers.Dense(units=1000, activation="relu"),
    tf.keras.layers.Dense(units=n_components),])

decoder = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_components)),
    tf.keras.layers.Dense(units=1000, activation="relu"),
    tf.keras.layers.Dense(units=2000, activation="relu"),
    tf.keras.layers.Dense(units=max(beer_dict), activation="relu"),])

encoder.summary()
decoder.summary()

callbacks = {"callbacks": [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                        min_delta=0.02,
                                                        patience=1,
                                                        verbose=1,)]}

para_umapper = ParametricUMAP(encoder=encoder,
                              decoder=decoder,
                              dims=dims,
                              n_components = n_components,
                              parametric_reconstruction=True,
                              autoencoder_loss=True,
                              keras_fit_kwargs=callbacks,
                              loss_report_frequency=10,
                              n_training_epochs=3,
                              batch_size=1000,
                              verbose=True)
para_umapper.fit(data)
para_umapper.save('drive/MyDrive/p_umap_384')

print(para_umapper._history)
