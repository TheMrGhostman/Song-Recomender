from flask import Flask, render_template, request,session, redirect
import pandas as pd
import numpy as np
from flask_table import Table, Col
from scipy import sparse
# from IPython.core.debugger import set_trace
from NNS import Recommender, user_based_model

#building flask table for showing recommendation results
class Results(Table):
    id = Col('Id', show=False)
    title = Col('Recommendation List')

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('welcome.html')

#Welcome Page
@app.route("/")
def welcome():
    return render_template('welcome.html')

#Rating Page
@app.route("/rating", methods=["GET", "POST"])
def rating():
    sle = np.load("data/label_encoder_songs_classes_.npy", allow_pickle=True)
    #movs = pd.read_csv('artist_song_song_id.csv',sep = ';')
    #movs = movs[movs.song_id.isin(sle)]
    #movs = movs[:100]
    #movs = [tuple(r) for r in movs.to_numpy()]
    movs = pd.read_csv("data/song_info_table_fix.csv", sep = ';')
    movs = movs.sort_values(by = ['artist_familiarity'], ascending = False)
    cols = [0,1,2,3,5,7,8,9,11]
    movs = movs.drop(movs.columns[cols],axis =1)
    art = movs['artist_name'].str[2:-1]
    song = movs['song_id'].str[2:-1]
    tit = movs['title'].str[2:-1]
    movs = pd.concat([art,song,tit],axis =1)
    movs = movs[movs.song_id.isin(sle)]
    movs = movs[:500]
    movs = [tuple(r) for r in movs.to_numpy()]


    if request.method=="POST":
        #selected = request.form.getlist("movie")
        return render_template('recommendation.html',test = movs) 
    return render_template('rating.html', test = movs) 

#Results Page
@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    if request.method == 'POST':
        
        
        selected = request.form.getlist("movie")

        clf = Recommender("user_based_NN", verbose = True)
        clf.load_encoders(path_songs="data/label_encoder_songs_classes_.npy", path_users="data/label_encoder_users_classes_.npy")
        X = sparse.load_npz("song-user_matrix_with_rating.npz")
        clf.fit(X=X)
        clf.save_model(model_name="Item-User_KNN_model")
        recs = clf.recommend(song_list= selected, full_name=True, return_song_id=True)
        
        movs = pd.read_csv("data/song_info_table_fix.csv", sep = ';')
        cols = [0,1,2,3,5,7,8,9,11]
        movs = movs.drop(movs.columns[cols],axis =1)
        art = movs['artist_name'].str[2:-1]
        song = movs['song_id'].str[2:-1]
        tit = movs['title'].str[2:-1]
        movs = pd.concat([art,song,tit],axis =1)
        
        tmp = movs[movs.song_id.isin(recs[0])]
        tmp = tmp['artist_name'] + " - " + tmp['title']

        output= list(tmp)
        table = Results(output)
        table.border = True
        return render_template('recommendation.html', table=table)

if __name__ == '__main__':
   app.run(debug = True)