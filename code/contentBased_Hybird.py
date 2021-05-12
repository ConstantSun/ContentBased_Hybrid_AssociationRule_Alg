import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import logging

import warnings; warnings.simplefilter('ignore')

logging.basicConfig(filename="../log/result.txt",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)



def load_data(dir_links_small='../input/links_small.csv', dir_metadata='../input/movies_metadata.csv'):
    """
    This function for: Load the Dataset and preprocessing data
    Args:
        dir_links_small: đường dẫn đến file links small  
        dir_metadata   : đường dẫn đến file meta data 
    Return:    
        links_small , md : pandas frame  
    """
    links_small = pd.read_csv(dir_links_small)
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

    md = pd. read_csv(dir_metadata)
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    return links_small, md


def get_vote_counts(md):
    """
    This function for: 
        get vote_counts
    Args:
        md : meta data, a pandas frame
    Return:    
        m : m is the minimum votes required to be listed in the chart
    """       
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    m = vote_counts.quantile(0.95)
    return m 


def get_mean_vote(md):
    """
    This function for: 
        get mean vote
    Args:
        md : meta data, a pandas frame
    Return:    
        C : C is the mean vote across the whole report
    """      
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    return C

def get_small_movies_metatdata(md): 
    """
    This function for: 
        get small movies meta data.
    Args:
        md : meta data, a pandas frame
    Return:    
        smd: small meta data.
    """       
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    md = md.drop([19730, 29503, 35587]) # these numbers presents row indices which rows contain bad format data; 
                                        #       just try
                                        #       md['id'] = md['id'].astype(int)
                                        #       u will get an error indicating it cannot convert '1975-xx-xx'.
    md['id'] = md['id'].astype('int')
    smd = md[md['id'].isin(links_small)]
    return smd





# Movie Description Based Recommender
def get_quantitative_matrix(smd):
    """
    This function for: 
        get quantitative_matrix
    Args:
        smd : small meta data, a pandas frame
    Return:    
        smd: small meta data.
        tfidf_matrix: quantitative matrix.
    """       
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')

    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])

    logging.info(f"tfidf_matrix shape: {tfidf_matrix.shape}")
    return smd, tfidf_matrix



def get_similarity_between2movies(tfidf_matrix):
    """
    This function for: 
        get similarity between 2 movies
    Args:
        tfidf_matrix: quantitative matrix.
    Return:    
        cosine_sim : similarity between 2 movies
    """ 
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim[0]
    return cosine_sim


def get_recommendations(title, cosine_sim, titles):
    """
    This function for: 
        get_recommendations
    Args:
        title: title of the movie
        cosine_sim: a matrix that indicates similarity betwwen 2 movies
        titles: all titles
    Return:    
        The 30 most similar movies based on the cosine similarity score.
    """     
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]










# Hybrid Recommender
# Input: User ID and the Title of a Movie
# Output: Similar movies sorted on the basis of expected ratings by that particular user.

def convert_int(x):
    """
    THis function for: convert x to int
    """
    try:
        return int(x)
    except:
        return np.nan

def get_movieID_indMovie(convert_int, smd, filename='../input/links_small.csv'):
    """
    This function for: 
        get movies id, index of movies
    Args:
        convert_int: function
        smd: small metadata
        filename: path to csv file 
    Return:    
        id_map: movies id            
        indices_map:  index of movies
    """         
    id_map = pd.read_csv(filename)[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')

    indices_map = id_map.set_index('id')
    return id_map, indices_map



def read_and_train_svd(filename='../input/ratings_small.csv'):
    """
    This function for: 
        read data and train SVD alg to fit data.
    Args:
        filename: path to data.
    Return:    
        svd : SVD alg after training with the dataset.
    """ 
    reader = Reader()
    ratings = pd.read_csv(filename)
    ratings.head()

    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd = SVD()
    # Run 5-fold cross-validation and then print results
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    trainset = data.build_full_trainset()
    svd.fit(trainset)
    return svd


def hybrid(userId, title, indices, id_map, cosine_sim, smd, indices_map):
    """
    This function for: 
        build a simple hybrid recommender that brings together techniques
        we have implemented in the content based and collaborative filter based engines.    
    Args:
        userId: User ID 
        title: the Title of a Movie
        indices : a list of movie indices 
        id_map: movies id      
        cosine_sim : similarity between 2 movies
        smd : small meta data
        indices_map: a list of movie indices    
    Return:    
        10 similar movies sorted on the basis of expected ratings by that particular user.
    """     
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


if __name__ == "__main__":

# Content Based Recommender: 
    links_small, md = load_data()
    m = get_vote_counts(md)
    C = get_mean_vote(md)
    smd = get_small_movies_metatdata(md)
    smd, tfidf_matrix = get_quantitative_matrix(smd)
    cosine_sim = get_similarity_between2movies(tfidf_matrix)

    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    logging.info(f"Top 10 recommendations for the movie: The Godfather:\n{get_recommendations('The Godfather', cosine_sim, titles).head(10)}")
    logging.info(f"Top 10 recommendations for the movie: The Dark Knight:\n{get_recommendations('The Dark Knight',  cosine_sim, titles).head(10)}")


# Hybrid Recommender

    id_map, indices_map = get_movieID_indMovie(convert_int, smd)
    svd = read_and_train_svd()
    logging.info(f"Top 10 movies for person with id 1, movie: Avatar:\n{hybrid(1, 'Avatar', indices, id_map, cosine_sim, smd, indices_map)}")
    logging.info(f"Top 10 movies for person with id 500, movie: Avatar:\n{hybrid(500, 'Avatar', indices, id_map, cosine_sim, smd, indices_map)}")


            