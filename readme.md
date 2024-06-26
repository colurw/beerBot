# beerBot

Is a pintelligent* recommendation alegorithm* that uses data scraped from BeerAdvocate.com to suggest novel beers based on a user's favourites.

An experiment / work in progress...

*<sub><sup>Nice one ChatGPT, but I'll stick with 'BeerBot' thank you.</sub></sup>

## 1_parse_json.py

Parses json data scraped from BeerAdvocate.com and RateBeer.com.  Each user's highest-scoring beers are added to a list in a dictionary, with the user's name as its key.

Only the index number of each beer is saved to the dictionary, so a second dictionary is created, which maps the index numbers to their names.

## 2_analyse_data.py

Discovers statistics from the data, to help choose which dataset to use.  BeerAdvocate's was deemed most suitable.

## 3_exploratory_factor_analysis.py

Uses Kernel Principal Component Analysis to produce a scree plot showing the cumulative variance explained by each extra dimension in the data.

There are 77,000 beers in the dataset, which is too many dimensions for a neural network (running locally) to reasonably handle.  By using matrix decomposition techniques, we can map the data to a lower-dimension space - if we can estimate the number of dimensions needed to hold a reasonably accurate representation.

The scree plot shows 384 dimensions are enough to capture 95% of the variance in the data.

## 4_train_decomposers.py

Trains a Truncated Singular Value Decomposition (T-SVD) model to reduce the number of dimensions in the data from ~77,000 to 384.  T-SVD is suitable for decomposing sparse data - but as a linear method, roughly analogous to matrix factorisation, it may be possible to improve on its performance with more advanced techniques.

Uniform Manifold Approximation and Projection (UMAP) can be used as a non-linear method of dimensionality reduction. It tries to recreate a topological representation of the data in fewer dimensions by maintaining its nearest neighbour graph.  It assumes the data must be spread uniformly over its manifold, and so stretches/shrinks regions of dense/sparse data to achieve this.  This allows a better capture of the fine details in the high-dimension data, in the low-dimension representation.

An issue with UMAP is the large amount of stored data required to transform newly recieved data between these two states.  The trained T-SVD model requires 226 MB of space, whilst the UMAP model requires 17.5 GB!  With a commensurate processing time for matrix operations when transforming fresh data.

Parametric-UMAP aims to solve this problem by combining the graph approach of UMAP with a Keras autoencoder neural network.  It samples the data, based on nearest neighbour edge probabilities, and learns how to decompose it into a parametric embedding - and how to recompose it back into _almost_ its original state.  It requires substantial computational resources to train this autoencoder model, but once finished, the model has a similar size to T-SVD, but with many of the non-linear and topological representations of UMAP - in theory.  

In reality, the decoder learns that just by guessing 'zero' for each one-hot feature, it can _almost_ perfectly recompose the original extremely sparse vector.  Meaning no beers get recommended.  Implementing a better-suited loss function (based on cosine distance or the Sørensen–Dice coefficient) may improve performance.

## 5_create_training_data.py

Turns the dictionary of each user's highest-scoring beers into a one-hot array (x_data) then randomly removes some beers to use as the targets that the neural network will attempt to learn (y_data).

This process is repeated several times, with different beers being selected each time.  This multiplies the amount of training data available.

## 6_train_recommender.py

Defines a neural network and trains it to recognise the missing favourite beers of each user.

A work in progress. The (awful) training graph plot shows the model is under-fitting and requires more training or possibly a wider/deeper neural network. It also might benefit from a higher learning rate, and more training data...  Or, more likely, a reworking of the algorithm to one based on natural language processing.

![image](https://github.com/colurw/beerBot/assets/66322644/c8e2c78a-9086-413a-9f8d-fd0830b22298)

## 7_get_recommendations.py

Searches for the best matches in the 'beer_dict' dictionary with the user's favourite beers.  Uses the returned indices to create a one-hot vector, which is then decomposed using T-SVD, then transformed by the neural network into a probability distribution describing the 'missing' favourite beers.  This vector is recomposed back into its original high-dimension space, then the matching names are looked up in 'beer_dict' dictionary.
