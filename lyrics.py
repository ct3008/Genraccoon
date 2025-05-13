import csv
import lyricsgenius
import time
import os
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()  
token = os.getenv("GENIUS_TOKEN")
genius = lyricsgenius.Genius(token, timeout=15,
retries=3, remove_section_headers=True)

infile  = "msd_dataset_enriched_with_similar_songs.csv"
outfile = "msd_dataset_with_genius_lyrics.csv"

with open(infile, newline="", encoding="utf8") as fin, \
     open(outfile, "w", newline="", encoding="utf8") as fout:

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # copy header from o/g msd file + add new column 
    header = next(reader)
    writer.writerow(header + ["genius_lyrics"])

    for i, row in enumerate(reader, start=1):
        artist = row[ header.index("artist_name") ]
        title  = row[ header.index("title") ]
        song = genius.search_song(title, artist)
        lyrics = song.lyrics if song else ""
        writer.writerow(row + [lyrics])

        if i % 100 == 0:
            print(f"Processed {i} rows…")
        time.sleep(genius.sleep_time)

print(f"Finished :)) — see {outfile}")
