import numpy as np 
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
# Load the Dataset for the Model
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
# List of The Features
## Genres, ID, Keywords, Title, Overview, Cast, Crew
movies = movies.merge(credits, on = "title")

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace = True) #Drops all the missing values in the dataset

def convert(obj):   #Evaluates the string and returns a corresponding list,dictionary. 
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(obj):  # Same as the other but only takes the first 3 names out.
    counter = 0
    L = []
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):    # Fetches the director name in the crew feature
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            L.append(i['name'])
            break
    return L

movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split()) # Splits the string into a list for each entity
movies['generes'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x]) # Removes the space between the words so that they work as a single entity 
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x]) 
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] # Concatination of all the columns into one 

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x)) #All the 
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features = 5000, stop_words= 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1]) [1: 6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')