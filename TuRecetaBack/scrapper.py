from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import requests
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn

search_term = "cream cheese"

recipe_key = "https://www.allrecipes.com/recipe/"
recipe_end_key = "\""

ingredient_key = "itemprop=\"recipeIngredient\""
ingredient_end_key = "<"

direction_key = "class=\"recipe-directions__list--item\">"
direction_end_key = "<"

units_of_measure = ["tablespoon", "cup", "ounce", "pound", "package",
                    "teaspoon", "fluid", "pint", "quart", "gallon",
                    "tablespoons", "cups", "ounces", "pounds", "packages",
                    "teaspoons", "pints", "quarts", "gallons"]

used_urls = set()

def create_recipe_features(search_term, recursive=False):
    urls = set()
    r = requests.get("https://www.allrecipes.com/search/results/?wt={}".format(search_term))
    page = r.text
    while page.find(recipe_key) >= 0:
        match_index = page.find(recipe_key)
        match_end_index = page.find(recipe_end_key, match_index)
        url = page[match_index:match_end_index]
        if url not in used_urls:
            urls.add(url)
            used_urls.add(url)
        page = page[match_end_index:]

    features = []

    for url in urls:
        print(url)
        r = requests.get(url)
        page = r.text
        ingredients = []
        directions = []

        # Get the ingredients and instructions.
        while page.find(ingredient_key) >= 0:
            ingredient_match_index = page.find(ingredient_key)
            ingredient_start = ingredient_match_index + len(ingredient_key)
            ingredient_end = page.find(ingredient_end_key, ingredient_start)
            ingredient = page[ingredient_start:ingredient_end]

            # Discriminate between adjectives and nouns, removing everything else (units of measure)
            ingredient_name = ""
            ingredient_adjectives = []
            for item in ingredient.split(" ")[1:]:
                item = item.lower()
                item = item.replace(",", "")
                item = item.replace("(", "")
                item = item.replace(")", "")
                try:
                    units_of_measure.index(item)
                except ValueError:
                    if len(item) > 2:
                        item = item.replace(" ", "")
                        nouns = wn.synsets(item, wn.NOUN)
                        adjectives = wn.synsets(item, wn.ADJ)
                        # The word is a noun.
                        if len(nouns) > len(adjectives):
                            ingredient_name = ingredient_name + " " + item
                        # The word is an adjective.
                        else:
                            try:
                                float(item)
                            except ValueError:
                                ingredient_adjectives.append(item)
            
            if len(ingredient_name) > 0:
                # Create embedding with ingredient noun and its adjectives.
                for adjective in ingredient_adjectives:
                    features.append((ingredient_name, adjective))

                ingredients.append(ingredient_name)
                
            page = page[ingredient_end:]
        while page.find(direction_key) >= 0:
            direction_match_index = page.find(direction_key)
            direction_start = direction_match_index + len(direction_key)
            direction_end = page.find(direction_end_key, direction_start)
            direction = page[direction_start:direction_end]
            for instruction in direction.split("."):
                directions.append(instruction)
            page = page[direction_end:]

        # Match the ingredients with actions through instructions.
        for direction in directions:
            verb = None
            for word in direction.split(" "):
                if len(wn.synsets(word, wn.VERB)) > 0:
                    verb = word
                    for ingredient in ingredients:
                        if direction.find(ingredient) >= 0:
                            features.append((ingredient, verb))
        
        # Get features from ingredients without making recursive call.
        if recursive == True:
            print(ingredients)
            for ingredient in ingredients:
                ingredient_features = create_recipe_features(ingredient, False)
                for embedding in ingredient_features:
                    features.append(embedding)
        
    return features

def create_embeddings(features):
    embeddings = pd.DataFrame(columns=['label'])
    for feature in features:
        print("{},{}".format(feature[0], feature[1]))
        if len(embeddings[embeddings['label'] == feature[0]]) == 0:
            embeddings.loc[embeddings.shape[0]] = ["" for x in range(embeddings.shape[1])]
            (embeddings.loc[embeddings.shape[0]-1])['label'] = feature[0]
        ind = int(embeddings[embeddings['label'] == feature[0]].index[0])
        try:
            embeddings[feature[1]].loc[ind] = 1
        except KeyError:
            embeddings[feature[1]] = np.zeros(embeddings.shape[0])
            embeddings[feature[1]].loc[ind] = 1
        try:
            embeddings[feature[0]].loc[ind] = 1
        except KeyError:
            embeddings[feature[0]] = np.zeros(embeddings.shape[0])
            embeddings[feature[0]].loc[ind] = 1
        embeddings['label_id'] = ind
    embeddings.drop(columns='label')
    return embeddings

print("Create features from recipes.")
features_tuples = create_recipe_features(search_term, False)
print("Size of features: {}".format(len(features_tuples)))
reccomendation_matrix = create_embeddings(features_tuples)
print("Created reccommendation matrix.")

def partition_data(data, train_percentage=0.8):
    data_matrix = data.as_matrix()
    train_size = int(data_matrix.shape[0]*0.8)
    print(train_size)
    print(data_matrix.shape)
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    return [train_X, train_Y, test_X, test_y]

model = RandomForestClassifier()
[train_X, train_y, test_X, test_y] = partition_data(reccomendation_matrix)