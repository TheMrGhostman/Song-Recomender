import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import sparse
import sys
from time import time
from IPython.core.debugger import set_trace

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


class Recommender(object):
    def __init__(self, model_type, metric='cosine', n_neighbors=20, verbose=False):
        """
        :param model_type:      Type of model we want to use - "user_based_NN" or "content_based_NN".
                                Core of model is same for both of them but for content_based model
                                matrix has not to be so sparse and there are more optimal algorithms.
                                User_based (Collaborative filtering) data matrix is very sparse.
                                So "brute" force algorithm is necessary.
        :param metric:          Metric for Nearest Neighbors model. For very sparse matrix is
                                recommended cosine matric.
        :param n_neighbors:     Number of neighbors (recommendations) for each input.
        :param verbose:         bool - Prints log of recommendations with distances.
        """
        self.metric = metric
        self.n_neighbors = n_neighbors
        if model_type in ['user_based_NN', "content_based_NN"]:
            self.model_type = model_type
        else:
            raise ValueError("Unknown model type.")
        self.song_label_encoder = LabelEncoder()
        self.user_label_encoder = LabelEncoder()
        self.verbose = verbose

    def create_model(self):
        """
        Helper function for model creation.
        """
        if self.model_type == 'user_based_NN':
            self.model = NearestNeighbors(metric=self.metric, algorithm='brute', n_neighbors=self.n_neighbors, n_jobs=-1)
            # brute force search is needed due to very sparse data matrix
        elif self.model_type == 'content_based_NN':
            self.model = NearestNeighbors(metric=self.metric, algorithm='auto', n_neighbors=self.n_neighbors, n_jobs=-1)
            # data matrix is not sparse so more efficient algorithms are possible to use
        else:
            raise ValueError("Something unexpected happened!")

    def fit(self, X):
        """
        :param X:      Sparse data matrix where columns are users and rows are items.
                       Format must be np.ndarray or scipy.sparse.csr_matrix.
        """
        if not isinstance(X, np.ndarray) and not isinstance(X, sparse.csr_matrix):
            raise ValueError("Dataset is not in correct format. Use np.ndarray or scipy.sparse.csr_matrix!")
        if "self.model" not in locals():
            self.create_model()
        self.model.fit(X)
        return self.model

    def save_model(self, model_name):
        """
        :param model_name:      Name or path where to save Nearest Neighbors model.
        """
        if ".joblib" not in model_name:
            model_name = model_name + ".joblib"
        joblib.dump(self.model, model_name)

    def load_model(self, model_name):
        """
        :param model_path:      Name or path from where you want to load model.
        """
        if ".joblib" not in model_name:
            model_name = model_name + ".joblib"
        self.model = joblib.load(model_name)
        self.n_neighbors = self.model.n_neighbors
        self.metric = self.model.metric

    def load_encoders(self, path_songs, path_users=None):
        """
        :param path_songs:      Path to saved encoding for songs_id.
        :param path_users:      Path to saved encoding for users_id.
        """
        self.song_label_encoder.classes_ = np.load(path_songs, allow_pickle=True)
        if path_users != None:
            self.user_label_encoder.classes_ = np.load(path_users, allow_pickle=True)

    def recommendNtoN(self, all_recommendations, distances):
        """
        :param all_recommendations:     matrix of recommendations
        :param distances:               matrix of distances from input song
        """
        best_ten = []
        best_ten_dist = []
        for i, rec in enumerate(all_recommendations):
            for j, song in enumerate(rec):
                if song not in best_ten:
                    best_ten.append(song)
                    best_ten_dist.append(distances[i,j])
                    break
        return np.array(best_ten), np.array(best_ten_dist)

    def recommend(self, song_list, full_name=False, return_song_id=True):
        """
        :param song_list:           List of songs to which we want to recommend different songs.
        :param full_name:           bool - song_list consists of song_ids insted of encodings
        :param return_id:           bool - True = return song_id / False = return songs encodings

        Returns matrix of recommendations where first column is encodings (song_id) of recommended songs
            and second column is distance from input song.
        """
        if full_name:
            # if song_list contains of full song_ids. It's needed to encode them.
            song_list = self.song_label_encoder.transform(np.array(song_list)).squeeze() #.reshape(len(song_list), 1)
        else:
            song_list = np.array(song_list) # make sure that data are in correct format

        list_len = len(song_list)
        distances, recommended_songs = self.model.kneighbors(self.model._fit_X[song_list], n_neighbors=self.n_neighbors+1)

        sorted_distances = np.argsort(-distances, axis=1)
        recommended_songs = np.vstack([sor[dist] for sor, dist in zip(recommended_songs,sorted_distances)])[:, 1:] # the closest one is same as input song
        distances = np.vstack([sor[dist] for sor, dist in zip(distances, sorted_distances)])[:, 1:]

        if len(np.unique(recommended_songs[:,0])) == list_len: # whole class is made only for x to x recommendations
            recommended_songs = recommended_songs[:,0]
            distances = distances[:,0]
        else:
            recommended_songs, distances = self.recommendNtoN(recommended_songs, distances)

        if return_song_id:  #recommendations[:,0]
            recommended_songs = self.song_label_encoder.inverse_transform(np.array(recommended_songs, dtype=int)) # label encoder does not accept float
        if self.verbose:
            if return_song_id:
                print(pd.DataFrame({"recommended_songs_id": recommended_songs, "cosine distance":distances}))
            else:
                print(pd.DataFrame({"recommended_songs_code": recommended_songs, "cosine distance":distances}))
        return recommended_songs, distances


def setup_based_model():
    model = Recommender("user_based_NN", verbose=True)
    model.load_encoders(path_songs="data/label_encoder_songs_classes_.npy",
                      path_users="data/label_encoder_users_classes_.npy")
    X = sparse.load_npz("data/song-user_matrix_with_rating.npz")
    model.fit(X=X)
    model.save_model(model_name="Item-User_KNN_model")
    print("Model prepared and saved!")


def user_based_model(verbose=True):
    """simplified function for model loading and setting"""
    model = Recommender("user_based_NN", verbose=verbose)
    model.load_encoders(path_songs="data/label_encoder_songs_classes_.npy",
                        path_users="data/label_encoder_users_classes_.npy")
    model.load_model(model_name="Item-User_KNN_model")
    return model


def content_based_model(verbose=True):
    """simplified function for model loading and setting"""
    model = Recommender("content_based_NN", verbose=verbose)
    model.load_encoders(path_songs="data/content_based_label_encoder_songs_classes_.npy",
                        path_users=None)
    model.load_model(model_name="Content_based_KNN_model")
    return model
