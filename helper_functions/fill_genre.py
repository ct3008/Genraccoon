import pandas as pd
import sqlite3
import re
from collections import Counter

# --- Load genre annotations from multiple sources ---
def load_tagtraum_genres(file_path):
    genre_data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                track_id = parts[0]
                genre = ' '.join(parts[1:]).strip()
                genre_data.append((track_id, genre))
    return pd.DataFrame(genre_data, columns=['track_id', 'genre'])

# --- Normalize strings for comparison ---
def normalize(text):
    return re.sub(r'[^a-z]', '', text.lower()) if isinstance(text, str) else ''

# --- Load and prioritize genres from multiple sources ---
def merge_genre_sources(df, cd2, cd2c, cd1):
    df = df.copy()
    df = df.merge(cd2.rename(columns={'genre': 'genre_cd2'}), on='track_id', how='left')
    df = df.merge(cd2c.rename(columns={'genre': 'genre_cd2c'}), on='track_id', how='left')
    df = df.merge(cd1.rename(columns={'genre': 'genre_cd1'}), on='track_id', how='left')

    # Collect and count all genre values
    all_genres = pd.concat([cd2['genre'], cd2c['genre'], cd1['genre']]).dropna().tolist()
    genre_counts = Counter(all_genres)
    top_genres = [genre for genre, _ in genre_counts.most_common(30)]

    # Build normalized genre map for top genres only
    norm_genre_map = {normalize(g): g for g in top_genres if normalize(g)}
    top_normalized = set(norm_genre_map.keys())

    def assign_genre(row):
        for col in ['genre_cd2', 'genre_cd2c', 'genre_cd1']:
            if pd.notna(row.get(col)) and row[col] in top_genres:
                return row[col]

        tags = row.get('artist_tags', '')
        if isinstance(tags, str):
            for tag in tags.split(','):
                norm_tag = normalize(tag)
                # Direct match
                if norm_tag in norm_genre_map:
                    return norm_genre_map[norm_tag]
                # Soft match with components
                for component in norm_tag.split():
                    if component in top_normalized:
                        return norm_genre_map[component]
                # Soft match with partials
                for norm_g, original_g in norm_genre_map.items():
                    if norm_tag in norm_g or norm_g in norm_tag:
                        return original_g
                        
        return None

    df['genre'] = df.apply(assign_genre, axis=1)
    df.drop(columns=['genre_cd2', 'genre_cd2c', 'genre_cd1'], inplace=True, errors='ignore')
    return df

# --- Main workflow ---
if __name__ == '__main__':
    df = pd.read_csv("msd_dataset_enriched_with_similar_songs.csv")
    print(f"Loaded {len(df)} tracks")

    # Load genre sources
    cd2 = load_tagtraum_genres("msd_tagtraum_cd2.cls")
    cd2c = load_tagtraum_genres("msd_tagtraum_cd2c.cls")
    cd1 = load_tagtraum_genres("msd_tagtraum_cd1.cls")

    # Merge genre information
    df = merge_genre_sources(df, cd2, cd2c, cd1)

    df.to_csv("msd_dataset_enriched_with_similar_songs.csv", index=False)
    print("Saved: msd_dataset_enriched_with_similar_songs.csv")
    print("Total with genre:", df['genre'].notna().sum())
