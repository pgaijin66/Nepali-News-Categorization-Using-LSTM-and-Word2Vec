## Usage 

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