import numpy as np
import pandas as pd
import os
import sys
import sklearn.preprocessing
import scipy.sparse
from tqdm import tqdm
USERS_THRESHOLD = 50
SONGS_THRESHOLD = 20


data_dir = os.getcwd()
if "train_triplets.txt" not in os.listdir(data_dir + "/data/"):
    raise ValueError("Data not found!")

user_song_path = os.path.join(data_dir,'data/train_triplets.txt')
user_song_data_original = pd.read_csv(user_song_path, sep="\t", header=None)
user_song_data_original.columns = ["user", "song id", "count"]

print("Selecting songs.")
# taking songs which were listened more then 20 times
number_of_listenings = user_song_data_original[['user','song id']].groupby('song id').count()
songs_more_than_m = number_of_listenings[number_of_listenings['user']>=SONGS_THRESHOLD]['user'].keys()
users_50_plus = user_song_data_original[user_song_data_original['song id'].isin(songs_more_than_m)].sort_values(['user'])

# taking users whick listened more then 50 songs from previous selection
number_of_songs_listened = users_50_plus[['user','song id']].groupby('user').count()
users_more_than_n = number_of_songs_listened[number_of_songs_listened['song id']>=USERS_THRESHOLD]['song id'].keys()
users_50_plus = users_50_plus[users_50_plus['user'].isin(users_more_than_n)].sort_values(['user'])
print("Songs are selected!")

# preprocessing
print("Encoding started.")
users_le = sklearn.preprocessing.LabelEncoder()
songs_le = sklearn.preprocessing.LabelEncoder()

tmp_u = users_le.fit_transform(users_50_plus["user"])
tmp_s = songs_le.fit_transform(users_50_plus["song id"])

# save encoders classes_
np.save("data/label_encoder_users_classes_.npy",users_le.classes_)
np.save("data/label_encoder_songs_classes_.npy",songs_le.classes_)
print("Encodning is done and classes are saved!")

users_50_plus["user"] = tmp_u
users_50_plus["song id"] = tmp_s

# transforming to sparse matrix datatype
matrix = scipy.sparse.csr_matrix((users_50_plus["count"], (users_50_plus["song id"], users_50_plus["user"])))
scipy.sparse.save_npz("data/users_50_plus_song-user.npz", matrix)
print("Sparse matrix saved! - it's shape is ", matrix.shape)

 # creating test dataset
SIZE = 10        # sample size
REPLACE = False  # with replacement
print("Creating train and test datset.")
fn = lambda obj: obj.loc[np.random.choice(obj.index, SIZE, REPLACE),:]
users_50_plus_train = users_50_plus.groupby('user', as_index=False).apply(fn)
users_50_plus_test = pd.concat([users_50_plus, users_50_plus_train]).drop_duplicates(keep=False)

# saving dataframe
file_name_test = 'data/user_50_plus_test.csv'
users_50_plus_test.to_csv(file_name_test, sep=';', index=False)
print("Test dataset is saved! Starting to transfrom counts to ratings.")

# transformation from listenings to rating
grouped = users_50_plus_train.groupby('user')
stars = []
for name, group in tqdm(grouped):
    stars_current=pd.cut(list((group['count'])), bins=5, labels=[1,2,3,4,5])
    stars.append(stars_current)

users_50_plus_train['stars'] = np.concatenate(stars)

# creating sparse matrix with ratings
matrix = scipy.sparse.csr_matrix((users_50_plus_train["stars"], (users_50_plus_train["song id"], users_50_plus_train["user"])))
scipy.sparse.save_npz("data/song-user_matrix_with_rating_train.npz", matrix)

# transformation from listenings to rating
grouped = users_50_plus.groupby('user')
stars = []
for name, group in tqdm(grouped):
    stars_current=pd.cut(list((group['count'])), bins=5, labels=[1,2,3,4,5])
    stars.append(stars_current)

users_50_plus['stars'] = np.concatenate(stars)

# creating sparse matrix with ratings
matrix = scipy.sparse.csr_matrix((users_50_plus["stars"], (users_50_plus["song id"], users_50_plus["user"])))
scipy.sparse.save_npz("data/song-user_matrix_with_rating.npz", matrix)

print("preprocessing is done!")
