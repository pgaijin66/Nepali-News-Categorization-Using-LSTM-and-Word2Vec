## Nepali News Classification Project Using Word2Vec and LSTM

This project is a **Nepali news classification system** developed using machine learning techniques. It check the news snippet passed and predicts which category that news falls under. The project involves several key steps, including data preprocessing, model training, and text classification using deep learning. Below is a detailed breakdown of the steps involved in the project:

<!-- TOC --><a name="table-of-contents"></a>
### Table Of Contents

- [Nepali News Classification Project Using Word2Vec and LSTM](#nepali-news-classification-project-using-word2vec-and-lstm)
   * [Usage ](#usage)
   * [1. Data Preprocessing](#1-data-preprocessing)
      + [Loading and Cleaning Data](#loading-and-cleaning-data)
      + [Tokenization and Stopword Removal](#tokenization-and-stopword-removal)
   * [2. Word Embedding with Word2Vec](#2-word-embedding-with-word2vec)
      + [Training Word2Vec Model](#training-word2vec-model)
      + [Embedding Matrix Construction](#embedding-matrix-construction)
   * [3. Model Building and Training](#3-model-building-and-training)
      + [LSTM Model Architecture](#lstm-model-architecture)
      + [Model Training](#model-training)
   * [4. Model Evaluation and Saving](#4-model-evaluation-and-saving)
      + [Evaluation](#evaluation)
      + [Saving the Model](#saving-the-model)
   * [5. Text Classification](#5-text-classification)
      + [Loading Models for Prediction](#loading-models-for-prediction)
      + [Preprocessing and Prediction](#preprocessing-and-prediction)
   * [6. Prediction Example](#6-prediction-example)
- [Summary](#summary)

<!-- TOC --><a name="usage"></a>
### Usage 

1. Install the Pre-trained GloVe Model
To ensure that the model functions correctly, you need to install the GloVe (Global Vectors for Word Representation) model with 300-dimensional vectors. The specific file required is glove.6B.300d.txt.

    Download the GloVe Model:
    You can download the glove.6B.300d.txt file from the official GloVe website. Choose the "glove.6B.zip" file, which contains several GloVe models. Once downloaded, extract the file and ensure glove.6B.300d.txt is accessible in your working directory.

2. Download and Extract the Nepali News Dataset
The dataset used for this project is named nepali_news_dataset_20_categories_large. This dataset consists of 20 different categories, with each category folder containing multiple text files.

    Download the Dataset:
    Obtain the nepali_news_dataset_20_categories_large.zip file from the provided source. If no specific source is provided, you may need to contact the project maintainer or check documentation for the dataset link.

    Extract the Dataset:
    Once downloaded, extract the contents of the nepali_news_dataset_20_categories_large.zip file into a directory of your choice. 

3. Install Dependencies
This project uses Python and requires specific dependencies listed in the requirements.txt file. To manage the environment and install dependencies efficiently, pipenv is recommended.

    Install Pipenv (if not already installed):

    ```
    pip install pipenv
    ```

    Install the Required Dependencies:
    Navigate to the project directory where requirements.txt is located and run the following command:

    ```
    pipenv install -r requirements.txt
    ```

    This command will create a virtual environment and install all necessary packages specified in the requirements.txt file.

4. Running the Project
With all dependencies installed and the dataset prepared, you can now run the project.

    Activate the Pipenv Shell:

    ```
    pipenv shell
    ```

    This command activates the virtual environment.

    Run the training Script:
    Once inside the virtual environment, execute the train_model script:

    ```
    python train_model.py
    ```

    Ensure that the script paths and configurations are correctly set up in the code to reference the GloVe model and the dataset files.

    Run the predicting Script:
    Once inside the virtual environment, execute the pridicting script (replace train_model.py with the predict_news.py script name ):

    ```
    python train_model.py
    ```

    Ensure that the script paths and configurations are correctly set up in the code to reference the GloVe model and the dataset files.

5. Additional Configuration
If there are any additional configurations (e.g., environment variables, config files), make sure to set them up before running the project. Refer to any additional documentation provided in the repository for specific details.


<!-- TOC --><a name="1-data-preprocessing"></a>
### 1. Data Preprocessing

<!-- TOC --><a name="loading-and-cleaning-data"></a>
#### Loading and Cleaning Data
- The project loads a dataset of Nepali news articles categorized into various categories. Each category is represented by a folder, and within each folder, there are text files containing the news articles.
- The `detect_encoding()` function is used to identify the encoding of each file to ensure it can be read correctly.
- The `preprocess_text()` function is applied to clean the text by removing special characters (except certain punctuation), converting it to lowercase, tokenizing it, and removing Nepali stopwords from the text.

<!-- TOC --><a name="tokenization-and-stopword-removal"></a>
#### Tokenization and Stopword Removal
- The text is tokenized into individual words using NLTK’s `word_tokenize`.
- A comprehensive list of Nepali stopwords is provided, and these stopwords are removed from the text to reduce noise and focus on meaningful words.

<!-- TOC --><a name="2-word-embedding-with-word2vec"></a>
### 2. Word Embedding with Word2Vec

<!-- TOC --><a name="training-word2vec-model"></a>
#### Training Word2Vec Model
- A `Word2Vec` model is trained on the tokenized news articles (`X_train`) to generate word embeddings. These embeddings capture the semantic meaning of words and are used to represent words as vectors in a high-dimensional space.
- The embeddings are saved in a model file (`word2vec.model`).

<!-- TOC --><a name="embedding-matrix-construction"></a>
#### Embedding Matrix Construction
- An embedding matrix is created using the trained `Word2Vec` model. This matrix maps each word to its corresponding embedding vector. The matrix is used as weights in the embedding layer of the LSTM model.

<!-- TOC --><a name="3-model-building-and-training"></a>
### 3. Model Building and Training

<!-- TOC --><a name="lstm-model-architecture"></a>
#### LSTM Model Architecture
- An LSTM (Long Short-Term Memory) model is built using TensorFlow/Keras. The model includes:
  - An Embedding layer initialized with the Word2Vec embedding matrix.
  - Two LSTM layers with 256 and 128 units, respectively.
  - A Dense layer with 128 units followed by a Dropout layer to prevent overfitting.
  - A final Dense layer with a softmax activation function to output the probabilities for each category.

<!-- TOC --><a name="model-training"></a>
#### Model Training
- The model is trained using the training data (`X_train_pad` and `y_train`) for 20 epochs with early stopping to prevent overfitting. The training process is monitored using validation data (a subset of the training data).
- The model’s performance is evaluated on both the training and test datasets, and the accuracy and loss are plotted to visualize the model's learning process.

<!-- TOC --><a name="4-model-evaluation-and-saving"></a>
### 4. Model Evaluation and Saving

<!-- TOC --><a name="evaluation"></a>
#### Evaluation
- After training, the model's accuracy and loss on both the training and test datasets are evaluated and printed.
- The training process, including accuracy and loss, is plotted to help understand the model’s performance over time.

<!-- TOC --><a name="saving-the-model"></a>
#### Saving the Model
- The trained model is saved as `news_classifier_lstm_with_word2vec.h5`.
- The label encoder (used to transform category labels into numerical values) is saved as `label_encoder.pkl`.

<!-- TOC --><a name="5-text-classification"></a>
### 5. Text Classification

<!-- TOC --><a name="loading-models-for-prediction"></a>
#### Loading Models for Prediction
- The saved LSTM model, `Word2Vec` model, and label encoder are loaded for use in text classification.

<!-- TOC --><a name="preprocessing-and-prediction"></a>
#### Preprocessing and Prediction
- New Nepali text is processed similarly to the training data (cleaning, tokenization, stopword removal, and conversion to sequences).
- The processed text is passed through the model to predict its category.
- The predicted category is decoded using the label encoder to get the human-readable category label.

<!-- TOC --><a name="6-prediction-example"></a>
### 6. Prediction Example

- A new Nepali text is classified using the trained model. The text is processed and the category is predicted, showing the effectiveness of the model in categorizing unseen news articles.

<!-- TOC --><a name="conclusion"></a>
## Conclusion

This project demonstrates a comprehensive approach to building a text classification system for Nepali news articles using deep learning techniques. It involves data preprocessing, embedding generation, model training with LSTM, and the implementation of a system to classify new text based on the trained model. The entire workflow is automated and can be reused for similar text classification tasks in the Nepali language.

