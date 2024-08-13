import os
import re
import pickle
import chardet
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

nltk.download('punkt', quiet=True)

# List of Nepali stopwords
nepali_stopwords = [
    'यो', 'छ', 'र', 'थियो', 'हुने', 'मा', 'लाई', 'पनि', 'भएको', 'तथा', 'यस', 'तर', 'कि', 'उनको', 'भन्ने', 
    'को', 'वा', 'रूपमा', 'को', 'नै', 'हो', 'गरेको', 'गर्ने', 'उनले', 'हुन्छ', 'गरे', 'गर्न', 'के', 'संग', 'गरेका',
    'अझै', 'अथवा', 'अर्थात', 'अर्को', 'अगाडी', 'अझ', 'आज', 'अनुसार', 'अन्य', 'अभि', 'अवस्था', 
    'आधा', 'आदि', 'आजकल', 'आपको', 'आफ्नै', 'आफ्नो', 'आफू', 'आफैं', 'आवश्यक', 'इ', 'इन', 
    'उनी', 'उनीहरू', 'उनको', 'उनकै', 'उहाँ', 'ए', 'एक', 'एउटा', 'कति', 'कसैले', 'कहाँ', 'का', 
    'कि', 'कुनै', 'कुनैपनि', 'के', 'कसैलाई', 'कसले', 'को', 'कुन', 'किन', 'कि', 'कृपया', 'खास', 
    'खासगरी', 'गरे', 'गरेको', 'गर्न', 'गरेका', 'गर्ने', 'गरेपछि', 'गर्नु', 'गर्छ', 'गर्छन्', 'गर्नेछन्', 
    'गर्छु', 'गर्दा', 'गर्दछ', 'गर्‍यो', 'गइरहेको', 'गैर', 'घटना', 'चलिरहेको', 'चलाउँछ', 'छ', 
    'छैन', 'चाहन्छ', 'जस्तो', 'जब', 'जसले', 'जसको', 'जहाँ', 'जस्तै', 'जसरी', 'जस', 'जसलाई', 
    'जस्तोसुकै', 'जति', 'जसमा', 'जसले', 'जससँग', 'जुन', 'जुनसुकै', 'जुनसुकै', 'तत्काल', 'तपाईं', 
    'तपाईँ', 'तपाई', 'तपाईंको', 'तपाईँको', 'तपाईको', 'तिमी', 'तिम्रो', 'तिमीले', 'तपाईँले', 'तर', 
    'तथा', 'त्यस', 'त्यसको', 'त्यसैले', 'त्यसो', 'त्यस्तै', 'त्यसपछि', 'तिनका', 'तिनी', 'तिनीहरु', 
    'तिनीहरूको', 'तिनको', 'तिमीहरु', 'त्यो', 'तिनीलाई', 'तपाईंहरू', 'तपाईंले', 'तिमील', 'तिम्रा', 
    'तिनै', 'त्यति', 'थियो', 'थिए', 'थिईन', 'थिएन', 'थियो', 'दुई', 'देखि', 'देखि', 'दिए', 'दिने', 
    'दिएको', 'दिएको', 'देखा', 'दुबै', 'दोश्रो', 'न', 'नभएको', 'नभएपछि', 'नभन्ने', 'नजर', 
    'नजिकै', 'नत्र', 'नयाँ', 'न', 'नहुने', 'नहि', 'निश्चित', 'नया', 'निको', 'निको', 'नियम', 
    'पनि', 'पहिलो', 'परेको', 'पर्याप्त', 'पर्दछ', 'पर्छ', 'परेका', 'पहिले', 'प्राय', 'परेर', 
    'पटक', 'पनि', 'फेरि', 'बनी', 'बनाइ', 'बनाइएको', 'बनाई', 'बनाएको', 'बने', 'बन्न', 'बलियो', 
    'बनाउने', 'बन्दै', 'बस', 'बीच', 'बारे', 'भरि', 'भर', 'भए', 'भएर', 'भएको', 'भन्छ', 'भन्ने', 
    'भएकोले', 'भने', 'भनिन्छ', 'भनेपछि', 'भयो', 'म', 'माथि', 'मात्रै', 'मात्र', 'मेरो', 'मैले', 
    'माझ', 'मात्र', 'मध्ये', 'माथिको', 'मात्र', 'माफ', 'मेरा', 'मै', 'मसँग', 'यति', 'यदि', 'यद्यपि', 
    'यहाँ', 'यही', 'यहीँ', 'यस', 'यसबारे', 'यसको', 'यसले', 'यसमा', 'यस्तो', 'यसै', 'या', 'यो', 
    'र', 'रख', 'रहेका', 'रहेछ', 'रह्यो', 'रहने', 'राखेको', 'रहेको', 'रखिएको', 'राख्ने', 'रहने', 
    'रखिएको', 'राख्न', 'लगायत', 'लिएर', 'लिए', 'लिएपछि', 'लिएर', 'लिन्छ', 'ले', 'लेख', 
    'लेख्ने', 'लागि', 'लगायतका', 'लिएर', 'लाई', 'लिएर', 'लिएर', 'वा', 'शायद', 'सक्छ', 'सक्ने', 
    'सबै', 'सबैका', 'सुरु', 'समेत', 'सधै', 'सँग', 'साथै', 'सक्छन्', 'समय', 'सकिन्छ', 'सक्ने', 
    'सबैभन्दा', 'सम्बन्धित', 'सम्भव', 'सो', 'सोही', 'सक्नु', 'सम्म', 'सय', 'सयौं', 'सवै', 
    'सीधा', 'सम्बन्धित', 'सक्ने', 'सम्पूर्ण', 'सरोकार', 'हाल', 'हालै', 'हालसम्म', 'हिजो', 
    'हुने', 'हुनु', 'हुनसक्छ', 'हुन', 'हुँदै', 'हुनुहुन्छ', 'हुन्छ', 'हुनसक्ने', 'हुने', 
    'हुनेछ', 'हुनुहुन्छ', 'हुन्', 'हेर्न', 'होस्', 'हो', 'होला', 'होइन', 'हुँदा', 'हुँदैन', 
    'हुन्छ', 'हुनु', 'हुनेछन्', 'हुन्छ', 'हुँदैछ', 'हुनुहुन्छ', 'हुँदा', 'हुने', 'हुँदैन'
]

def preprocess_text(text):
    # Remove special characters but keep some punctuation
    text = re.sub(r'[^\w\s।?!]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove Nepali stopwords
    filtered_tokens = [word for word in tokens if word not in nepali_stopwords]
    
    return filtered_tokens

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def load_data_from_folders(base_folder):
    categories = []
    texts = []
    
    for category in os.listdir(base_folder):
        category_folder = os.path.join(base_folder, category)
        if os.path.isdir(category_folder):
            for file_name in os.listdir(category_folder):
                file_path = os.path.join(category_folder, file_name)

                encoding = detect_encoding(file_path)
                if not encoding:
                    print(f"Encoding detection failed for {file_name}, using utf-8 as fallback.")
                    encoding = 'utf-8'
                
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        text = file.read()
                except UnicodeDecodeError:
                    print(f"Failed to decode {file_name} with {encoding}, using 'replace' error handling.")
                    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                        text = file.read()
                
                # Clean and tokenize the text
                tokens = preprocess_text(text)
                texts.append(tokens)
                categories.append(category)
    
    return pd.DataFrame({'text': texts, 'category': categories})

# Load the dataset
base_folder = '/Users/pthapa/Documents/src/github/personal/nepal-news-category/nepali_news_dataset_20_categories_large/nepali_news_dataset_20_categories_large/'
df = load_data_from_folders(base_folder)

# Encode labels
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.2, random_state=42)

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=X_train, vector_size=300, window=10, min_count=2, workers=4)
word2vec_model.save("word2vec.model")

# Create an embedding matrix for the Embedding layer in the LSTM model
vocab_size = len(word2vec_model.wv.key_to_index) + 1  # Adding 1 to account for the padding word
embedding_dim = 300  # Updated to match the Word2Vec vector size
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word2vec_model.wv.key_to_index.items():
    embedding_vector = word2vec_model.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Prepare the data for LSTM input
max_len = 200
X_train_pad = pad_sequences([[word2vec_model.wv.key_to_index.get(word, 0) for word in text] for text in X_train], maxlen=max_len)
X_test_pad = pad_sequences([[word2vec_model.wv.key_to_index.get(word, 0) for word in text] for text in X_test], maxlen=max_len)

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(256, return_sequences=True),  # Increased units
    LSTM(128),  # Increased units
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train_pad, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test, verbose=0)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model and related components
model.save('news_classifier_lstm_with_word2vec.h5')

with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(le, le_file)

print("Model and related components saved successfully!")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()