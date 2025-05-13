# 🎵 Genraccoon  
*A Neural Network-Based Approach to Genre Prediction and Song Recommendation*

---

## 🧠 Project Overview  
**Genraccoon** explores deep learning techniques to classify music by genre and recommend songs based on lyrical and audio metadata. We experiment with GRU and LSTM-based architectures, incorporating attention mechanisms and convolutional filters to enhance performance.

---

## 📁 File Structure

### 🔧 Data Processing

| File | Description |
|------|-------------|
| `helper_functions/main.py` | Parses Million Song Dataset (MSD) HDF5 files to create a CSV of audio features, genre, and lyrics. **Output:** `msd_dataset_full.csv` |
| `helper_functions/helper.py` | Enhances CSV with metadata from `track_metadata.db`, `artist_term.db`, `artist_similarity.db`. **Output:** `msd_dataset_enriched_with_similar_songs.csv` |
| `helper_functions/fill_genre.py` | Fills in genre labels using Tagtraum files. Falls back on artist tags if needed. |
| `lyrics.py` | Uses the Genius API to retrieve lyrics for songs by artist and title. **Output:** `msd_dataset_with_genius_lyrics.csv` |
| `genius_lyrics_clean.ipynb` | Cleans lyrics and standardizes genre labels into 15 unified classes. **Output:** `msd_dataset_with_only_clean_lyrics.csv` |

---

### 🧠 GRU-Based Models

| File | Description |
|------|-------------|
| `gru_hyperparam.ipynb` | Baseline GRU model with hyperparameter tuning. Compares metadata vs. lyrics-only versions. |
| `gru_sliding_window.ipynb` | GRU model with sliding window over lyrics for better sequential modeling. |
| `gru+attention.ipynb` | GRU + Attention model combining sliding windows and context-aware focus. |
| `recommender.ipynb` | Song recommendation pipeline using genre predictions and metadata.  |
**Output:** `_recommendations.csv`, `_formatted_recommendations.txt` |

---

### 🧮 LSTM-Based Models

| File | Description |
|------|-------------|
| `lstm.ipynb` | Basic LSTM model with frozen pre-trained embeddings. |
| `new_lstm_attention_baseline.ipynb` | LSTM+Attention baseline model before adding convolution layers. |
| `lstm_attention.ipynb` | Final LSTM+Attention model with Conv1D, trainable embeddings, and optimized architecture. |

---

### 📂 Miscellaneous

| File/Folder        | Description                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `old_code/`        | Archived versions of earlier model/code attempts.                      |
| `output_files/`    | Clean CSVs for testing and training (`train_set.csv`, `test_set.csv`). |
| `requirements.txt` | Python package dependencies.                                           |

---

## 📦 Required Datasets & Resources

### 🔗 Datasets to Download

- **Tagtraum Genre Annotations**  
  [https://www.tagtraum.com/msd_genre_datasets.html](https://www.tagtraum.com/msd_genre_datasets.html)  
  - Download:
    - `msd_tagtraum_cd1.cls.zip`
    - `msd_tagtraum_cd2.cls.zip`
    - `msd_tagtraum_cd2c.cls.zip`

- **Lyrics Dataset**  
  [http://millionsongdataset.com/musixmatch/#getting](http://millionsongdataset.com/musixmatch/#getting)

- **Million Song Subset**  
  [http://millionsongdataset.com/pages/getting-dataset/#subset](http://millionsongdataset.com/pages/getting-dataset/#subset)

### 🗃 SQLite Metadata (Required)

- `track_metadata.db`
- `artist_term.db`
- `artist_similarity.db`

Download from the MSD subset page:  
[http://millionsongdataset.com/pages/getting-dataset/#subset](http://millionsongdataset.com/pages/getting-dataset/#subset)

### 🎤 Genius API  
- Documentation: [https://docs.genius.com/](https://docs.genius.com/)

---

## 📌 Notes  
- Properly formatted lyrics are fetched via the Genius API.
- Audio and metadata features are derived from MSD and aligned using track IDs.
- Models were evaluated using accuracy, F1, and genre-level confusion matrices.
