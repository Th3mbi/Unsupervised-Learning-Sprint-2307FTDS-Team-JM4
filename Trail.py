"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer 
usable_data = pd.read_csv("resources/data/usableData.csv")

# Importing data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),
                     min_df=0.02, stop_words='english')
tf_sim_matrix = tf.fit_transform(usable_data['combine'])

cosine_sim_matrix = cosine_similarity(tf_sim_matrix, 
                                        tf_sim_matrix)

movies = usable_data['movieId']
indices = pd.Series(usable_data.index, index=usable_data['movieId'])

trained = pd.read_csv('resources/data/trained.csv')

import operator # <-- Convienient item retrieval during iteration 
import heapq

def content_generate_top_N_recommendations(movie_Id, N=10):
    # Convert the string book title to a numeric index for our 
    # similarity matrix
    b_idx = indices[movie_Id]
    # Extract all similarity values computed with the reference book title
    sim_scores = list(enumerate(cosine_sim_matrix[b_idx]))
    # Sort the values, keeping a copy of the original index of each value
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Select the top-N values for recommendation
    sim_scores = sim_scores[1:N]
    # Collect indexes 
    movie_indices = [i[0] for i in sim_scores]
    # Convert the indexes back into titles 
    return movies.iloc[movie_indices]
