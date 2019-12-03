import sys
import os.path
import pandas as pd
import numpy as np


# import models #our models

def metric(reco_songs, list_of_users, dfdiff):
    # reco_songs ... list of 10 recommended song ids

    print('Calculating MAP:')
    best = 3 + 3 / 4 + 3 / 5 + 3 / 6 + 3 / 7 + 3 / 8 + 3 / 9 + 3 / 10
    num_songs = len(reco_songs)

    ###AP = [0]*len(list_of_users)
    AP = {i: 0 for i in list_of_users}
    dfdiff = dfdiff[dfdiff['user'].isin(list_of_users)]
    for user in list_of_users:
        num_of_correct_predictions = 0
        for i in np.arange(num_songs):
            if reco_songs[i] in dfdiff[dfdiff['user'] == user]['song id'].values:
                AP[user] += num_of_correct_predictions / (i + 1)
                num_of_correct_predictions += 1
        AP[user] /= best
    return np.mean([AP[i] for i in AP])


def filter_users(input_songs, dfpart, dfdiff):
    # input_songs ... 10 input songs ids

    list_of_users = []
    for song in input_songs:
        list_of_users.extend(dfpart[dfpart['song id'] == song].user.tolist())
    return list(dict.fromkeys(list_of_users))
    ###return dfdiff[dfdiff['user'].isin(list_of_users)]


def baseline():
    top = 'tabulky/top_10_listened_songs.csv'
    dftop = pd.read_csv(top, ';')
    return dftop


def songs_are_valid(list_of_songs, dffull):
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
        if not list_of_songs[i] in dffull['song id'].values:
            print("Your song on ",i,"-th place is not in our database. For list of song ids see artist_song_song_id.csv.")
            return False
    return True


def main():
    # basic interface
    full = 'tabulky/users_6_plus.csv'
    part = 'tabulky/users_6_plus_partial.csv'
    diff = 'tabulky/user_6_plus_diff.csv'

    dffull = pd.read_csv(full, ';')
    dfpart = pd.read_csv(part, ';')
    dfdiff = pd.read_csv(diff, ';')

    song_num = 10
    song_ids = [0] * song_num
    i = 0
    print("For list of song ids see file 'artist_song_song_id.csv'")
    songs_are_inserted = False
    while not songs_are_inserted:
        list_is_prepared = input(
            "Have you prepared your list of songs in a format \"song1_id;song2_id;...;song10_id\" [y/n]")
        if list_is_prepared == "n":
            while i < song_num:
                song_ids[i] = input("Insert song id: ")
                try:
                    assert song_ids[i] in dffull['song id'].values
                    if song_ids[i] not in song_ids[:i]:
                        i += 1
                        print(song_num - i, ' song ids remaining to enter')
                    else:
                        print('You have already entered this song.')
                except:
                    print("Wrong input. Enter again. For list of song ids see artist_song_song_id.csv")
            songs_are_inserted = True

        elif list_is_prepared == "y":
            raw_song_id_list = input("Please insert your list: ")
            list_of_song_ids = raw_song_id_list.split('; ')

            if songs_are_valid(list_of_song_ids, dffull):
                song_ids = list_of_song_ids
                songs_are_inserted = True
        else:
            print("Wrong input")

    # baseline
    dftop = baseline()
    print('\n', 'Recommended songs are: ', dftop)

    # calculating MAP
    list_of_users = filter_users(song_ids, dfpart, dfdiff)
    MAP = metric(dftop['song id'].values.tolist(), list_of_users, dfdiff)
    print('MAP: ', MAP)


if __name__ == '__main__':
    main()




