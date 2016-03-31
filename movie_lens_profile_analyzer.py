# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:08:13 2016

@author: canalli
"""

import pandas as pd
from numpy import concatenate

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin1')


genre = pd.read_csv('ml-100k/u.genre', sep='|', names=['name', 'code'], encoding='latin1')
genre_names = genre['name'].values


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin1' )

good_ratings = ratings[ratings['rating'] > 4]
bad_ratings = ratings[ratings['rating'] < 2]

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=concatenate([m_cols, genre_names]), encoding='latin1')

# create one merged DataFrame
movie_ratings = pd.merge(movies, ratings)
lens = pd.merge(movie_ratings, users)

user_likes = pd.merge(pd.merge(movies, good_ratings), users)
user_likes = user_likes[concatenate([['user_id'], genre_names])]
user_likes = user_likes.groupby('user_id').sum()

user_dislikes = pd.merge(pd.merge(movies, bad_ratings), users)
user_dislikes = user_dislikes[concatenate([['user_id'], genre_names])]
user_dislikes = user_dislikes.groupby('user_id').sum()

user_preferences = (user_likes - user_dislikes).dropna()


popular_movies = pd.merge(movies, good_ratings)
popular_comedies = popular_movies.loc[popular_movies['Comedy'] == 1]
popular_comedies = (popular_comedies.groupby('title')).size().sort_values()

