# hierarchical_gru.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')

SEED = 42
np.random.seed(SEED)
df = pd.read_csv("msd_dataset_enriched_with_similar_songs.csv")
df = df[~df['genre'].isna() & ~df['lyrics'].isna()]
stopwords_set = set(stopwords.words('english'))

def clean_lyrics(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    return ' '.join([t for t in tokens if t not in stopwords_set])

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
df = df[df['genre'].map(df['genre'].value_counts()) > 1]

# Tokenize lines
def split_lines(text):
    return [line.split() for line in text.split('\n') if line.strip()]

df['lines'] = df['cleaned_lyrics'].apply(lambda txt: split_lines(txt))

# Word2Vec
all_tokens = [token for lines in df['lines'] for token in lines]
w2v_model = Word2Vec(sentences=all_tokens, vector_size=50, window=5, min_count=3, workers=4)
vocab = set(w2v_model.wv.index_to_key)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([' '.join(line) for lines in df['lines'] for line in lines])
tokenizer.word_index = {word: i+1 for i, word in enumerate(vocab)}

MAX_LINES = 10
MAX_WORDS = 20
X = []
y = []

for i, row in df.iterrows():
    lines = row['lines'][:MAX_LINES]
    line_seqs = tokenizer.texts_to_sequences([' '.join(line)[:MAX_WORDS] for line in lines])
    padded_lines = pad_sequences(line_seqs, maxlen=MAX_WORDS)
    while len(padded_lines) < MAX_LINES:
        padded_lines = np.vstack([padded_lines, np.zeros((1, MAX_WORDS))])
    X.append(padded_lines)
    y.append(row['genre'])

X = np.array(X)
y = to_categorical(LabelEncoder().fit_transform(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 50))
for word, i in tokenizer.word_index.items():
    embedding_matrix[i] = w2v_model.wv[word]

# Hierarchical GRU model
word_input = Input(shape=(MAX_WORDS,))
word_emb = Embedding(input_dim=embedding_matrix.shape[0], output_dim=50, weights=[embedding_matrix], trainable=False)(word_input)
word_gru = Bidirectional(GRU(32))(word_emb)
word_encoder = Model(word_input, word_gru)

sent_input = Input(shape=(MAX_LINES, MAX_WORDS))
sent_encoder = TimeDistributed(word_encoder)(sent_input)
sent_gru = Bidirectional(GRU(64))(sent_encoder)
drop = Dropout(0.3)(sent_gru)
out = Dense(y.shape[1], activation='softmax')(drop)

model = Model(sent_input, out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, callbacks=[early], verbose=2)

print("\nHierarchical GRU training complete.")