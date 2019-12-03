from interface import songs_are_valid
import pandas as pd


def test_songs_are_valid():
    raw_song_id_list_ok = "SORCKJU12A67021859; SOUBCNK12AB017CA1C; SOIHJSD12A6701EB04; SOOWKSP12AB018043B; SOWDXXY12A8C133818; SOGQWTX12AB017FD78; SOCWAFA12A8AE48A78; SOXFSTR12A8AE463B0; SORRGXW12A8C139847; SOFGXIF12A670215E4"
    raw_song_id_list_nok_count = "SOUBCNK12AB017CA1C; SOIHJSD12A6701EB04; SOOWKSP12AB018043B; SOWDXXY12A8C133818; SOGQWTX12AB017FD78; SOCWAFA12A8AE48A78; SOXFSTR12A8AE463B0; SORRGXW12A8C139847; SOFGXIF12A670215E4"
    raw_song_id_list_nok_formatting = "SOUBCNK12AB017CA1C,   SOIHJSD12A6701EB04; SOOWKSP12AB018043B; SOWDXXY12A8C133818; SOGQWTX12AB017FD78; SOCWAFA12A8AE48A78; SOXFSTR12A8AE463B0; SORRGXW12A8C139847; SOFGXIF12A670215E4"
    raw_song_id_list_nok_song_not_exist = "SORCKJU12A6702; SOUBCNK12AB017CA1C; SOIHJSD12A6701EB04; SOOWKSP12AB018043B; SOWDXXY12A8C133818; SOGQWTX12AB017FD78; SOCWAFA12A8AE48A78; SOXFSTR12A8AE463B0; SORRGXW12A8C139847; SOFGXIF12A670215E4"

    list_of_song_ids_ok = raw_song_id_list_ok.split('; ')
    list_of_song_ids_nok_count = raw_song_id_list_nok_count.split('; ')
    list_of_song_ids_nok_formatting = raw_song_id_list_nok_formatting.split('; ')
    list_of_song_ids_nok_song_not_exist = raw_song_id_list_nok_song_not_exist.split('; ')

    full = 'tabulky/users_6_plus.csv'
    dffull = pd.read_csv(full, ';')

    assert songs_are_valid(list_of_song_ids_ok, dffull) == True
    assert songs_are_valid(list_of_song_ids_nok_count, dffull) == False
    assert songs_are_valid(list_of_song_ids_nok_formatting, dffull) == False
    assert songs_are_valid(list_of_song_ids_nok_song_not_exist, dffull) == False

