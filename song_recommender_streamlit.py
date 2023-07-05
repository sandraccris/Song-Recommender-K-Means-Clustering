import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import pairwise_distances_argmin_min
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import seaborn as sns
import random as r
import numpy as np
from streamlit_lottie import st_lottie
import json
import requests


# Lottie animation
url = requests.get("https://assets6.lottiefiles.com/packages/lf20_6jfc4gby.json")
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
      print("Error in URL")


# Loading Datasets
top_songs = pd.read_csv("top_songs.csv")
df5 = pd.read_csv("both_playlists_audio_features.csv")

st.title("Music Recommendation System")

# Adding Animation
st_lottie(url_json)

#st.image("music.jpg", width=700)

st.subheader("Are You Tired of Listening To The Same Music?")

# Choosing columns for X
X = df5.drop(columns=['Unnamed: 0','type', 'id', 'uri', 'track_href', 'analysis_url','song_title', 'songs_id', 'artist'])

# Standardize the numerical data because K means clustering is based on distances between features.
scaler = StandardScaler()
scale_x = scaler.fit_transform(X)   #scale_x is an array

# Training model and predicting clusters

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit_predict(scale_x) #predicting/assigning the clusters to our data points.
clusters = kmeans.predict(scale_x)


# adding column song_title, artist, scale_x array and clusters array to the df.

scaled_df = pd.DataFrame(scale_x, columns=X.columns)
scaled_df['song_title'] = df5['song_title']
scaled_df['artist'] = df5['artist']
scaled_df['cluster'] = clusters
scaled_df.head()

def song_recommender(song_title):
    #user_input = input("Please insert a Song Name: ").lower()
    for i in top_songs['song']:
        if user_input in i:
            rows = top_songs.shape[0]
            random_row = r.randrange(rows)
            print('\n [Your Recommended Artist and Song:')
            return ' - '.join(top_songs.iloc[random_row, :])
    if user_input not in i:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="af3a4e21d9974f798b0ddef081728f2b",
                                                                   client_secret="99a65d20eff04d64bcf24b11824dffc4"))
        results = sp.search(q=f'track:{user_input}', limit=1)
        track_id = results['tracks']['items'][0]['id']  # obtain user input song id
        audio_features = sp.audio_features(track_id)  # get song features with the obtained id

        df_ = pd.DataFrame(audio_features)  # create dataframe
        new_features = df_[X.columns]

        scaled_x = scaler.transform(new_features)  # scale features for the kmeans model
        cluster_pred = kmeans.predict(scaled_x)  # predict cluster of user input song

        filtered_df = scaled_df[scaled_df['cluster'] == cluster_pred[0]][
            X.columns]  # filter dataset to predicted cluster(cluster is an array with only 1 element therefore index 0)

        closest, _ = pairwise_distances_argmin_min(scaled_x,
                                                   filtered_df)  # get closest song from filtered dataset with same cluster as user input song
        recommended_songs = scaled_df.loc[closest]['song_title'].values[0], scaled_df.loc[closest]['artist'].values[0]
        return recommended_songs


st.write('Discover a New Song based on Your Taste!')

user_input = st.text_input("**Please Insert a Song You Like:**")

#Defining user_input
if user_input:
    selected_song = user_input

# Button to run the recommendation function
if st.button('Recommend Me a Song'):
    recommended_songs = song_recommender(selected_song)
    st.write("Your Recommended Song is: ")
    st.success(recommended_songs)
