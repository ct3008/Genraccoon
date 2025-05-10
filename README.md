# Genraccoon
Genraccoon - an analysis of NN based song genre prediction and recommenders

<<<<<<< HEAD
Files:
- main.py: data processing. Assumes MillionSongs dataset is in the folder outside of the current repo. Recursively traverses to read all h5 files and put data into csv format
	- output: msd_dataset_full.csv w/ audio features, genre, lyrics (bag of words)

- helper.py: appends information from track_metadata.db, artist_term.db, artist_similarity.db
	- output: msd_dataset_enriched_with_similar_songs.csv

- gru.py: basic gru code to take in msd_dataset_enriched_with_similar_songs.csv and run training and validation on. Not enough data currently to do testing. Uses sliding window method for best performance.

- hier_gru.py: hierarchical gru method. Work in progress
=======
Useful Datasets to download:
Genre tagging: https://www.tagtraum.com/msd_genre_datasets.html 
- download msd_tagtraum_cd2.cls.zip

Lyrics: http://millionsongdataset.com/musixmatch/#getting
- download the train and test dataset (hyperlinks)

Million Song Subset: http://millionsongdataset.com/pages/getting-dataset/#subset
- download "MILLION SONG SUBSET" (hyperlinks)
>>>>>>> 9758a0a29c16c9ce560ff09ed00b7f6ce8af5ed5
