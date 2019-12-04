import tables
import numpy as np
from tqdm import tqdm
import pandas as pd

data_yes = []
data_maybe = []
columns_yes = ["metadata.songs.cols.artist_familiarity",
               "metadata.songs.cols.artist_hotttnesss",
               "metadata.songs.cols.artist_id",
               "musicbrainz.songs.cols.idx_artist_mbtags",
               "metadata.songs.cols.artist_name",
               "analysis.songs.cols.loudness",
               "metadata.songs.cols.song_id",
               "analysis.songs.cols.tempo",
               "metadata.songs.cols.song_hotttnesss",
               "analysis.songs.cols.time_signature",
               "metadata.songs.cols.title",
               "analysis.songs.cols.key"]

columns_yes_names = [i.split(".")[-1] for i in columns_yes]

columns_maybe = ["metadata.songs.cols.idx_artist_terms",
                 "analysis.songs.cols.duration",
                 "analysis.songs.cols.mode",
                 "analysis.songs.cols.mode_confidence",
                 "analysis.songs.cols.time_signature_confidence",
                 "analysis.songs.cols.key_confidence"]

columns_all_names = columns_yes + [i.split(".")[-1] for i in columns_maybe]

f = tables.open_file("subset_msd_summary_file.h5", mode="r", driver="H5FD_CORE")

for i in tqdm(columns_yes):
    g = f.root
    for j in i.split("."):
        g = getattr(g,j)
    data_yes.append(np.array(g[:]).reshape(10000,1))

for i in tqdm(columns_maybe):
    g = f.root
    for j in i.split("."):
        g = getattr(g,j)
    data_maybe.append(np.array(g[:]).reshape(10000,1))

f.close()

data_all = data_yes+data_maybe
data_yes = np.concatenate(data_yes, axis=1)
data_all = np.concatenate(data_all, axis=1)

print(data_yes.shape)
print(data_all.shape)

df_yes = pd.DataFrame(data_yes, columns=columns_yes_names)
df_all = pd.DataFrame(data_all, columns=columns_all_names)

print("saving")

df_yes.to_csv("song_info_table.csv", sep=';', index = False)
df_all.to_csv("song_info_table_with_maybe_columns.csv", sep=';', index = False)

print("done")
