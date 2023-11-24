import pickle

## RateBeer

# create empty dictionaries
beer_dict = {}
reviews_dict = {}

# parse json file
with open('raw data/ratebeer2.json', 'r', encoding='utf-8') as file:
    for json_obj in file:
        # convert json string to dict
        data = eval(json_obj)
        # update dictionary of beer_ids with beer_name
        beer_dict[int(data['beer/beerId'])] = data['beer/name']
        
        # if reviewer scores beer highly
        if int(data['review/overall'].split('/')[0]) >= 17:
            # if reviewer not seen before
            if data['review/profileName'] not in reviews_dict:   
                # create new key for reviewer and start new list of favourites
                reviews_dict[data['review/profileName']] = [int(data['beer/beerId'])]
            else:
                # append beer_id to reviewers list of favourites
                reviews_dict[data['review/profileName']].append(int(data['beer/beerId']))

# save outputs
with open('cleaned data/rb_beer_dict.pkl', 'wb') as file:
    pickle.dump(beer_dict, file)
with open('cleaned data/rb_reviews_dict.pkl', 'wb') as file:
    pickle.dump(reviews_dict, file)


## BeerAdvocate

# create empty dictionaries
beer_dict = {}
reviews_dict = {}

# parse beeradvocate json file
with open('raw data/beeradvocate2.json', 'r', encoding='utf-8') as file:
    for json_obj in file:
        # convert json string to dict
        data = eval(json_obj)
        # update dictionary of beer_ids with beer_name
        beer_dict[int(data['beer/beerId'])] = data['beer/name']
        
        # if reviewer scores beer highly
        if float(data['review/overall']) >= 4:
            
            # if reviewer not already in reviews_dict
            if data['review/profileName'] not in reviews_dict:   
                # create new key for reviewer and start new list of favourites
                reviews_dict[data['review/profileName']] = [int(data['beer/beerId'])]
            else:
                # append beer_id to reviewers list of favourites
                reviews_dict[data['review/profileName']].append(int(data['beer/beerId']))

# save outputs
with open('cleaned data/ba_beer_dict.pkl', 'wb') as file:
    pickle.dump(beer_dict, file)
with open('cleaned data/ba_reviews_dict.pkl', 'wb') as file:
    pickle.dump(reviews_dict, file)

