import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
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

# assess explained variance using principle component analysis
pca = PCA().fit(data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('BeerAdvocate Explained Variance')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.savefig('data analysis/ba_explained_variance')

# >>> 80% of variance is _alledgedly_ explained by 1400 dimensions... But PCA is not suited to sparse data!


# assess explained variance ratio using kernel principle component analysis
pca = KernelPCA(n_components=2000, kernel='linear', eigen_solver='arpack')  # kernel='rbf'
pca.fit(data)
var_values = pca.eigenvalues_ / sum(pca.eigenvalues_)
print(sum(var_values))
plt.plot(np.arange(1, pca.n_components + 1), var_values, "+", linewidth=2)
plt.title('BeerAdvocate Explained Variance Ratio - KernelPCA')
plt.xlabel('number of components')
plt.ylabel('explained variance ratio')
plt.savefig('data analysis/ba_kpca_scree_linear')

# >>> 90% of variance is explained by 250 dimensions, and 95% by 384 dimensions