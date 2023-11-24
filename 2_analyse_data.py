import pickle
import matplotlib.pyplot as plt

# load BeerAdvocate data
with open('cleaned data/ba_reviews_dict.pkl', 'rb') as file:
    reviews_dict = pickle.load(file)
with open('cleaned data/ba_beer_dict.pkl', 'rb') as file:
    beer_dict = pickle.load(file)

# display stats
print(len(beer_dict))
print(max(beer_dict))
print(len(reviews_dict))

# create histogram
lengths = [len(reviews_dict[key]) for key in reviews_dict]
bins = [num+0.5 for num in range(3,20)]
plt.hist(lengths, bins, histtype='bar', rwidth=0.8)
plt.xlabel('Number Of Favourites / Reviewer')
plt.ylabel('Frequency')
plt.title('BeerAdvocate')
plt.show()


# load RateBeer data
with open('cleaned data/rb_reviews_dict.pkl', 'rb') as file:
    reviews_dict = pickle.load(file)
with open('cleaned data/rb_beer_dict.pkl', 'rb') as file:
    beer_dict = pickle.load(file)

# display stats
print(len(beer_dict))
print(max(beer_dict))
print(len(reviews_dict))

# create histogram
lengths = [len(reviews_dict[key]) for key in reviews_dict]
bins = [num+0.5 for num in range(3,20)]
plt.hist(lengths, bins, histtype='bar', rwidth=0.8)
plt.xlabel('Number Of Favourites / Reviewer')
plt.ylabel('Frequency')
plt.title('RateBeer')
plt.show()