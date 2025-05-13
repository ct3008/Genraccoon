
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from gensim.models import Word2Vec
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping

# # Step 1: Load and preprocess
# nltk.download('stopwords')
# df = pd.read_csv("msd_dataset_enriched_with_similar_songs.csv")
# df = df[~df['genre'].isna() & ~df['lyrics'].isna()]
# stopwords_set = set(stopwords.words('english'))

# def clean_lyrics(text):
#     text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
#     tokens = text.split()
#     return ' '.join([t for t in tokens if t not in stopwords_set])

# df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
# df = df[df['genre'].map(df['genre'].value_counts()) > 1]

# # Step 2: Tokenize and embed
# tokenized = df['cleaned_lyrics'].apply(str.split)
# w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=3, workers=4)
# valid_words = set(w2v_model.wv.index_to_key)

# # Build tokenizer
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df['cleaned_lyrics'])
# filtered_index = {word: i+1 for i, word in enumerate(valid_words)}
# tokenizer.word_index = filtered_index
# tokenizer.index_word = {i: word for word, i in filtered_index.items()}

# # Convert lyrics
# sequences = tokenizer.texts_to_sequences(df['cleaned_lyrics'])
# X_lyrics = pad_sequences(sequences, maxlen=300)

# embedding_matrix = np.zeros((len(filtered_index) + 1, 50))
# for word, i in filtered_index.items():
#     embedding_matrix[i] = w2v_model.wv[word]

# # Labels
# le = LabelEncoder()
# y = to_categorical(le.fit_transform(df['genre']))

# X_train, X_test, y_train, y_test = train_test_split(X_lyrics, y, test_size=0.2, stratify=y, random_state=42)
# print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# # Step 3: Run GRU experiments
# best_val_acc = 0
# best_config = None
# results = []

# hidden_units = [16, 32, 64, 96, 128]
# dropouts = [0.2, 0.3, 0.4, 0.5]
# layer_depths = [1, 2]  # single vs stacked GRUs

# for units in hidden_units:
#     for dropout_rate in dropouts:
#         for depth in layer_depths:
#             print(f"\nTraining GRU model: units={units}, dropout={dropout_rate}, layers={depth}")

#             input_ = Input(shape=(300,))
#             x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=50,
#                           weights=[embedding_matrix], trainable=False)(input_)
#             for _ in range(depth - 1):
#                 x = Bidirectional(GRU(units, return_sequences=True))(x)
#                 x = Dropout(dropout_rate)(x)
#             x = Bidirectional(GRU(units))(x)
#             x = Dropout(dropout_rate)(x)
#             output = Dense(y.shape[1], activation='softmax')(x)

#             model = Model(inputs=input_, outputs=output)
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#             early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)

#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_test, y_test),
#                 batch_size=64,
#                 epochs=30,
#                 callbacks=[early_stop],
#                 verbose=2
#             )

#             final_acc = max(history.history['val_accuracy'])
#             print(f"Final val accuracy: {final_acc:.4f}")
#             results.append((units, dropout_rate, depth, final_acc))
#             if final_acc > best_val_acc:
#                 best_val_acc = final_acc
#                 best_config = (units, dropout_rate, depth)
#                 best_model = model

# print("\nSummary of Experiments:")
# for units, dropout, depth, acc in results:
#     print(f"units={units}, dropout={dropout}, layers={depth} → val_acc={acc:.4f}")
# print(f"\nBest config: GRU({best_config[0]}) dropout={best_config[1]}, layers={best_config[2]} → val_acc={best_val_acc:.4f}")


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and preprocess
nltk.download('stopwords')
df = pd.read_csv("msd_dataset_enriched_with_similar_songs.csv")
df = df[~df['genre'].isna() & ~df['lyrics'].isna()]
stopwords_set = set(stopwords.words('english'))

def clean_lyrics(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = text.split()
    return ' '.join([t for t in tokens if t not in stopwords_set])

df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)
df = df[df['genre'].map(df['genre'].value_counts()) > 1]

# Step 2: Tokenize and embed
tokenized = df['cleaned_lyrics'].apply(str.split)
w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=3, workers=4)
valid_words = set(w2v_model.wv.index_to_key)

# Build tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_lyrics'])
filtered_index = {word: i+1 for i, word in enumerate(valid_words)}
tokenizer.word_index = filtered_index
tokenizer.index_word = {i: word for word, i in filtered_index.items()}

# Convert lyrics
sequences = tokenizer.texts_to_sequences(df['cleaned_lyrics'])
X_lyrics = pad_sequences(sequences, maxlen=300)

embedding_matrix = np.zeros((len(filtered_index) + 1, 50))
for word, i in filtered_index.items():
    embedding_matrix[i] = w2v_model.wv[word]

# Labels
le = LabelEncoder()
y_labels = le.fit_transform(df['genre'])
y = to_categorical(y_labels)

# Hold out a true test set
X_temp, X_final_test, y_temp, y_final_test, y_temp_labels, y_final_labels = train_test_split(
    X_lyrics, y, y_labels, test_size=0.05, stratify=y_labels, random_state=42
)

# Remaining split
X_train, X_val, y_train, y_val, y_train_labels, y_val_labels = train_test_split(
    X_temp, y_temp, y_temp_labels, test_size=0.2, stratify=y_temp_labels, random_state=42
)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# Search best config
hidden_units = [16, 32, 64, 96, 128]
dropouts = [0.2, 0.3, 0.4, 0.5]
layer_depths = [1, 2]

best_val_acc = 0
best_config = None
best_model = None

for units in hidden_units:
    for dropout_rate in dropouts:
        for depth in layer_depths:
            print(f"\nTraining GRU model: units={units}, dropout={dropout_rate}, layers={depth}")
            input_ = Input(shape=(300,))
            x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=50, weights=[embedding_matrix], trainable=False)(input_)
            for _ in range(depth - 1):
                x = Bidirectional(GRU(units, return_sequences=True))(x)
                x = Dropout(dropout_rate)(x)
            x = Bidirectional(GRU(units))(x)
            x = Dropout(dropout_rate)(x)
            output = Dense(y.shape[1], activation='softmax')(x)

            model = Model(inputs=input_, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=64, epochs=20, callbacks=[early_stop], verbose=1
            )
            val_acc = max(history.history['val_accuracy'])
            print(f"Val Accuracy: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (units, dropout_rate, depth)
                best_model = model

# Summary
print(f"\nBest config: GRU({best_config[0]}), dropout={best_config[1]}, layers={best_config[2]} → val_acc={best_val_acc:.4f}")

# Evaluate best model with and without class weights
units, dropout_rate, depth = best_config

print("\nRetraining best model WITHOUT class weights...")
input_ = Input(shape=(300,))
x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=50, weights=[embedding_matrix], trainable=False)(input_)
for _ in range(depth - 1):
    x = Bidirectional(GRU(units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
x = Bidirectional(GRU(units))(x)
x = Dropout(dropout_rate)(x)
output = Dense(y.shape[1], activation='softmax')(x)

model_no_weights = Model(inputs=input_, outputs=output)
model_no_weights.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_no_weights.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=20, callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)], verbose=0)
y_pred_no_weights = np.argmax(model_no_weights.predict(X_final_test), axis=1)
acc_no_weights = accuracy_score(y_final_labels, y_pred_no_weights)

print("\nRetraining best model WITH class weights...")
model_with_weights = Model(inputs=input_, outputs=output)
model_with_weights.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_with_weights.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=20, callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)], class_weight=class_weight_dict, verbose=0)
y_pred_with_weights = np.argmax(model_with_weights.predict(X_final_test), axis=1)
acc_with_weights = accuracy_score(y_final_labels, y_pred_with_weights)

print("\nTest Accuracy:")
print(f"Without class weights → {acc_no_weights:.4f}")
print(f"With class weights    → {acc_with_weights:.4f}")
