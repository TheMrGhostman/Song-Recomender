import tables
import numpy as np
import tqdm
import pandas as pd

data_yes = []
data_maybe
columns_yes = ["artist_familiarity",
               "artist_hotttnesss",
               "artist_id",
               "artist_mbtags",
               "artist_name",
               "loudness",
               "song_id",
               "tempo",
               "song_hotttnesss",
               "time_signature",
               "title",
               "key"]

columns_maybe = ["artist_terms",
                 "duration",
                 "mode",
                 "model_confidence",
                 "time_signature_confidence",
                 "key_confidence"]

f = tables.open_file("subset_msd_summary_file.h5", mode="r", driver="H5FD_CORE")

for i in tqdm(columns_yes):
    data_yes.append(np.array(f.root.metadata.song.cols[i][:]))

for i in tqdm(columns_maybe):
    data_maybe.append(np.array(f.root.metadata.song.cols[i][:]))

f.close()

data_yes = np.concatenate(data_yes, axis=0).T
data_all = np.concatenate(data_yes+data_maybe, axis=0).T

df_yes = pd.DataFrame(data_yes, columns=columns_yes)
df_all = pd.DataFrame(data_all, columns=columns_yes+columns_maybe)

df_yes.to_csv("song_info_table.csv", sep=';', index = False)
df_all.to_csv("song_info_table_with_maybe_columns.csv", sep=';', index = False)
