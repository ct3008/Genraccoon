# Genraccoon
Genraccoon - an analysis of NN based song genre prediction and recommenders

Files:
- main.py: data processing. Assumes MillionSongs dataset is in the folder outside of the current repo. Recursively traverses to read all h5 files and put data into csv format
	- output: msd_dataset_full.csv w/ audio features, genre, lyrics (bag of words)

- helper.py: appends information from track_metadata.db, artist_term.db, artist_similarity.db
	- output: msd_dataset_enriched_with_similar_songs.csv

- gru.py: basic gru code to take in msd_dataset_enriched_with_similar_songs.csv and run training and validation on. Not enough data currently to do testing. Uses sliding window method for best performance.

- hier_gru.py: hierarchical gru method. Work in progress