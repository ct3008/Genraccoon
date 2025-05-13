# Genraccoon
Genraccoon - an analysis of NN based song genre prediction and recommenders

HEAD
Files:

Data Processing:
- helper_functions/main.py: data processing. Assumes MillionSongs dataset is in the folder outside of the current repo. Recursively traverses to read all h5 files and put data into csv format
	- output: msd_dataset_full.csv w/ audio features, genre, lyrics (bag of words)

- helper_functions/helper.py: appends information from track_metadata.db, artist_term.db, artist_similarity.db
	- output: msd_dataset_enriched_with_similar_songs.csv

- helper_functions/fill_genre.py: enhances msd_dataset_enriched_with_similar_songs.csv by using all 3 tagtraum files to find proper genre for each song. If no match is found, uses artist tags from artist_term.db to perform soft matches and find appropriate genre tags.
	- output: msd_dataset_enriched_with_similar_songs.csv

 - lyrics.py: calls upon genius API to search on artist_name and song_title from our dataset. If found, append properly formatted lyrics to dataset
	- output: msd_dataset_with_genius_lyrics.csv

 - genius_lyrics_clean.ipynb: cleans raw Genius lyrics by removing metadata and punctuation and standardizes genre labels into 15 unified categories

-------------------------------------------------------------

GRU Code: 
- gru_hyperparam.ipynb: notebook used to train and test basic GRU model on our dataset. Does hyperparameter tuning and tests performance on test data both with and without metadata present to analyze its effect.

- gru_sliding_window.ipynb: notebook used to train improved GRU model using sliding window method and perform same analyses as gru_hyperparam.ipynb.

- gru+attention.ipynb: notebook used to train improved GRU model using sliding window method WITH ATTENTION LAYERS and perform same analyses as gru_hyperparam.ipynb.

- recommender.ipynb: notebook used to train recommendation system using lyrics + metadata and genre.
	- output: *_recommendations.csv, *_formatted_recommendations.txt
-------------------------------------------------------------

LSTM Code:
- lstm.ipynb: notebook used to train LSTM model

- lstm_attention.ipynb: notebook used to train LSTM + Attention model


-------------------------------------------------------------

Miscellaneous:
- old code/: folder of previous versions of our code for bookkeeping
- output_files/: folder holding our relocated example generated csv's for cleanliness
	- output_files/test_set.csv -> test set created in recommmendations.ipynb to train model
	- output_files/train_set.csv -> train set created in recommmendations.ipynb to train model
- requirements.txt: requirements to run code

------------------------------------------------------------------------------------------------------------------------

Useful Datasets to download:
Genre tagging: https://www.tagtraum.com/msd_genre_datasets.html 
- download:
	- msd_tagtraum_cd2.cls.zip
   	- msd_tagtraum_cd1.cls.zip
   	- msd_tagtraum_cd2c.cls.zip

Lyrics: http://millionsongdataset.com/musixmatch/#getting
- download the train and test dataset (hyperlinks)

Million Song Subset: http://millionsongdataset.com/pages/getting-dataset/#subset
- download "MILLION SONG SUBSET" (hyperlinks)

useful DB files
- Get from: http://millionsongdataset.com/pages/getting-dataset/#subset
  	- the three SQLite database files (8,9,10)
  	- http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db
  	- http://www.ee.columbia.edu/~thierry/artist_term.db
  	- http://www.ee.columbia.edu/~thierry/artist_similarity.db
 
Genius API:
- https://docs.genius.com/


 

