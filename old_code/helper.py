import sqlite3
import pandas as pd

# Load base CSV
df = pd.read_csv("msd_dataset_full.csv")
print(f"Loaded {len(df)} rows from msd_dataset_full.csv")

# Step 1: Load track metadata
conn_track = sqlite3.connect("track_metadata.db")
track_meta = pd.read_sql_query(
    "SELECT track_id, artist_id, song_id, release, year, artist_familiarity, artist_hotttnesss "
    "FROM songs", conn_track)
conn_track.close()
df = df.merge(track_meta, on="track_id", how="left")
print(f"Merged with track_metadata.db")

# Step 2: Load artist terms
conn_terms = sqlite3.connect("artist_term.db")
artist_terms = pd.read_sql_query("SELECT artist_id, term FROM artist_term", conn_terms)
conn_terms.close()
artist_tag_summary = artist_terms.groupby("artist_id")["term"].apply(lambda x: ', '.join(x)).reset_index()
artist_tag_summary.columns = ["artist_id", "artist_tags"]
df = df.merge(artist_tag_summary, on="artist_id", how="left")
print(f"Merged with artist_term.db")

# Step 3: Load artist similarity
conn_sim = sqlite3.connect("artist_similarity.db")
artist_sim = pd.read_sql_query("SELECT target AS artist_id, similar AS similar_artist_id FROM similarity", conn_sim)
conn_sim.close()

# Mapping: artist -> similar artists
artist_sim_dict = artist_sim.groupby("artist_id")["similar_artist_id"].apply(list).to_dict()

# Build mapping of artist_id -> track_ids and track_id -> title
artist_to_tracks = df.groupby("artist_id")["track_id"].apply(list).to_dict()
track_to_title = df.set_index("track_id")["title"].to_dict()

# Helper: Get top 5 similar song titles
def find_similar_songs(track_id, artist_id):
    similar_artists = artist_sim_dict.get(artist_id, [])
    collected = []

    for sim_artist in similar_artists:
        for candidate_id in artist_to_tracks.get(sim_artist, []):
            if candidate_id != track_id:
                title = track_to_title.get(candidate_id)
                if candidate_id in track_to_title:
                    collected.append(candidate_id)
            if len(collected) >= 5:
                break
        if len(collected) >= 5:
            break
    return "; ".join(collected)

df["top_5_similar_songs"] = df.apply(
    lambda row: find_similar_songs(row["track_id"], row["artist_id"]), axis=1)

# Save final enriched dataset
df.to_csv("msd_dataset_enriched_with_similar_songs.csv", index=False)
print("Saved final dataset: msd_dataset_enriched_with_similar_songs.csv")
