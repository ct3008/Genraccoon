import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import sqlite3

conn = sqlite3.connect('track_metadata.db')


# --------------------------------------
# 1. AUDIO FEATURE EXTRACTION
# --------------------------------------
def extract_audio_features(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            md = f['metadata']['songs'][0]
            an = f['analysis']['songs'][0]

            track_id = an['track_id'].decode('utf-8')
            artist_name = md['artist_name'].decode('utf-8')
            title = md['title'].decode('utf-8')
            duration = an['duration']
            tempo = an['tempo']
            key = an['key']
            loudness = an['loudness']

            segments_pitches = f['analysis']['segments_pitches'][:]
            segments_timbre = f['analysis']['segments_timbre'][:]

            pitches_mean = np.mean(segments_pitches, axis=0)
            pitches_std = np.std(segments_pitches, axis=0)
            timbre_mean = np.mean(segments_timbre, axis=0)
            timbre_std = np.std(segments_timbre, axis=0)

            features = {
                'track_id': track_id,
                'artist_name': artist_name,
                'title': title,
                'duration': duration,
                'tempo': tempo,
                'key': key,
                'loudness': loudness
            }

            for i in range(12):
                features[f'pitch_mean_{i}'] = pitches_mean[i]
                features[f'pitch_std_{i}'] = pitches_std[i]
                features[f'timbre_mean_{i}'] = timbre_mean[i]
                features[f'timbre_std_{i}'] = timbre_std[i]

            return features
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        return None


def process_dataset(h5_directory):
    data = []
    for root, _, files in os.walk(h5_directory):
        for file in files:
            if file.endswith('.h5'):
                h5_path = os.path.join(root, file)
                features = extract_audio_features(h5_path)
                if features:
                    data.append(features)
    print(f"Processed {len(data)} .h5 audio files.")
    return pd.DataFrame(data)


# --------------------------------------
# 2. GENRE LOADER
# --------------------------------------
def load_tagtraum_genres(cls_file_path):
    genre_data = []
    with open(cls_file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                track_id = parts[0]
                genres = ' '.join(parts[1:]).strip()
                genre_data.append((track_id, genres))
    genre_df = pd.DataFrame(genre_data, columns=['track_id', 'genre'])
    print(f"Loaded {len(genre_df)} genre entries from {cls_file_path}")
    return genre_df


# --------------------------------------
# 3. LYRICS LOADER (FILTERED)
# --------------------------------------
def load_mxm_lyrics_incrementally(path, track_ids_to_extract):
    lyrics = {}
    idx_to_word = {}
    found_vocab = False

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if line.startswith('%') and not found_vocab:
                vocab = line[2:].split(',')
                idx_to_word = {i+1: word for i, word in enumerate(vocab)}
                found_vocab = True
                print(f"Loaded {len(idx_to_word)} vocab words from line {line_num}")
                continue

            parts = line.split(',')
            if len(parts) < 3:
                continue

            track_id = parts[0]
            if track_id not in track_ids_to_extract:
                continue

            bow = parts[2:]
            words = []
            for token in bow:
                if ':' not in token:
                    continue
                try:
                    idx, count = map(int, token.split(':'))
                    word = idx_to_word.get(idx, '')
                    words.extend([word] * count)
                except:
                    continue

            if words:
                lyrics[track_id] = ' '.join(words)

    print(f"Extracted lyrics for {len(lyrics)} tracks from {path}")
    return lyrics


# --------------------------------------
# 4. MAIN EXECUTION
# --------------------------------------
if __name__ == "__main__":
    h5_directory = '../MillionSongSubset/'
    genre_file = './msd_tagtraum_cd2.cls'
    mxm_train_path = './mxm_dataset_train.txt'
    mxm_test_path = './mxm_dataset_test.txt'

    # Step 1: Audio
    df_audio = process_dataset(h5_directory)
    track_ids = set(df_audio['track_id'])
    # df_audio.to_csv("audio_basic.csv")

    # Step 2: Genre Appending
    genre_df = load_tagtraum_genres(genre_file)
    df_audio = df_audio.merge(genre_df, on='track_id', how='left')

    # Step 3: Lyrics Appending
    lyrics_train = load_mxm_lyrics_incrementally(mxm_train_path, track_ids)
    lyrics_test = load_mxm_lyrics_incrementally(mxm_test_path, track_ids)
    all_lyrics = {**lyrics_test, **lyrics_train}

    df_audio['lyrics'] = df_audio['track_id'].map(all_lyrics)
    df_audio = df_audio.dropna(subset=['lyrics'])

    print(f"Final dataset has {len(df_audio)} tracks with audio, genre, and lyrics.")
    df_audio.to_csv('msd_dataset_full.csv', index=False)
    print("Saved: msd_dataset_full.csv")
