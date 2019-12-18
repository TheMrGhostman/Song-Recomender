import sys
import os.path
import pandas as pd
import numpy as np

#to make sure that we get the whole dataframe printed (without dots)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from NNS import Recommender, user_based_model,content_based_model

def baseline():
    top = 'Code/tabulky/top_10_listened_songs.csv'
    dftop = pd.read_csv(top, ';')
    return dftop

def songs_are_valid(list_of_songs, all_songs_ids):
    if len(list_of_songs)>10:
        print("Your list has more than 10 songs, try again.")
        return False
    if len(list_of_songs)<10:
        print("Your list has less than 10 songs, try again.")
        return False
    if not len(list_of_songs) == len(set(list_of_songs)):  #Checking for duplicates
        print("Your list has duplicated entries")
        return False
    for i in range(10):
        if not list_of_songs[i] in all_songs_ids:
            print("Your song on ",i,"-th place is not in our database. For list of song ids see songs_table.csv.")
            return False
    return True


def main():
    
    #novy soubor
    table_songs_UB = 'Code/tabulky/songs_table_UB.csv'
    table_songs_CB = 'Code/tabulky/songs_table_CB.csv'
    dfsongsUB = pd.read_csv(table_songs_UB,';')
    dfsongsCB = pd.read_csv(table_songs_CB,';')
    
    
    #loading models
    model_UB = user_based_model(verbose=False)
    model_CB = content_based_model(verbose=False)
    UB_songs_ids = model_UB.song_label_encoder.classes_
    CB_songs_ids = model_CB.song_label_encoder.classes_
    
    model_chosen = False
    while not model_chosen:
        model_number = input("""
        Select model:
        Baseline model                    ... 1
        User-based model                  ... 2
        Content-based model               ... 3
        Get recommendation for all models ... 4
        """)
        try:
            print(model_number)
            assert model_number in ['1','2','3','4']
            model_chosen = True
        except:
            print('Type either one of the following number: 1, 2, 3, 4')
    
    #if user selects baseline or content-based model, we can search through broader database as len(CB_songs_ids) = 381 526
    #if user selects user-based, we need to narrow the database as len(UB_songs_ids) = 161 173
    if model_number in ['1','3']:
        all_songs_ids = CB_songs_ids
    else:
        all_songs_ids = UB_songs_ids
        
    song_num = 10
    song_ids = [0]*song_num
    i = 0
    
    if model_number in ['1','3']:
        print("For list of song IDs see file 'songs_table_CB.csv'")
    else:
        print("For list of song IDs see file 'songs_table_UB.csv'")
    
    songs_are_inserted = False
    while not songs_are_inserted:
        list_is_prepared = input(
            "Have you prepared your list of songs in a format \"song1_id;song2_id;...;song10_id\" [y/n]")
        if list_is_prepared == "n":
            while i < song_num:
                song_ids[i] = input("Insert song id: ")
                try:
                    assert song_ids[i] in all_songs_ids
                    if song_ids[i] not in song_ids[:i]:
                        i += 1
                        print(song_num - i, ' song ids remaining to enter')
                    else:
                        print('You have already entered this song.')
                except:
                    if model_number in ['1','3']:
                        print("For list of song IDs see file 'songs_table_CB.csv'")
                    else:
                        print("For list of song IDs see file 'songs_table_UB.csv'")
            songs_are_inserted = True

        elif list_is_prepared == "y":
            raw_song_id_list = input("Please insert your list: ")
            list_of_song_ids = raw_song_id_list.split('; ')

            if songs_are_valid(list_of_song_ids, all_songs_ids):
                song_ids = list_of_song_ids
                songs_are_inserted = True
        else:
            print("Wrong input")

    #baseline
    if model_number in ['1','4']:
        dftop = baseline()
        print('\n', 'Baseline recommended songs are: ','\n', dftop)
    
    #user_based
    if model_number in ['2','4']:
        rec_UB_IDs = model_UB.recommend(song_list=song_ids, full_name=True, return_song_id=True)[0]
        rec_UB_artists = list(map(lambda x: dfsongsUB[dfsongsUB['Song ID']==x]['Artist'].values[0],rec_UB_IDs))
        rec_UB_song_names = list(map(lambda x: dfsongsUB[dfsongsUB['Song ID']==x]['Title'].values[0],rec_UB_IDs))
        rec_UB_data = {'Artist':rec_UB_artists, 'Title':rec_UB_song_names,'Song ID':rec_UB_IDs} 
        dfUB = pd.DataFrame(rec_UB_data)
        print('\n', 'User-based model recommended songs are:','\n', dfUB)
        
    #content_based
    if model_number in ['3','4']:
        rec_CB_IDs = model_CB.recommend(song_list=song_ids, full_name=True, return_song_id=True)[0]
        rec_CB_artists = list(map(lambda x: dfsongsCB[dfsongsCB['Song ID']==x]['Artist'].values[0],rec_CB_IDs))
        rec_CB_song_names = list(map(lambda x: dfsongsCB[dfsongsCB['Song ID']==x]['Title'].values[0],rec_CB_IDs))
        rec_CB_data = {'Artist':rec_CB_artists, 'Title':rec_CB_song_names,'Song ID':rec_CB_IDs} 
        dfCB = pd.DataFrame(rec_CB_data)
        print('\n', 'Content-based model recommended songs are:','\n', dfCB)


if __name__ == '__main__':
    main()




