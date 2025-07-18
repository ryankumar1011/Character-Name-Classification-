# Character-Name-Classification

This project involved a NLP taks to classify character names based on dialogue lines from an online novel. A BERT model is fine-tuned for this purpose. 

## References
For fine tuning BERT: https://www.kaggle.com/code/neerajmohan/fine-tuning-bert-for-text-classification/comments

Using Hugging Face Transform Library: https://huggingface.co/learn/llm-course/chapter1/1

Parsing HTML with BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/


## Project Structure

### `data_extraction.py`
Extracts and preprocesses dialogue data from web-scraped novel PDFs.

### `data.csv`
Store extracted `Text` and `Label` columns 

### `classical_ml.py`
Implements baseline classifiers including:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)

Performance is evaluated using confusion matrices to provide baselines for comparison against the BERT model.

### `model.py`
Fine-tunes a BERT model using the HuggingFace Transformers library with the following default hyperparameters:

- `MAX_LENGTH = 128`
- `BATCH_SIZE = 16`
- `EPOCHS = 3`
- `LEARNING_RATE = 2e-5`

### `bert_dialogue_classifier/`
Contains assets used for training and inference:
- Tokenizer configuration
- Label mappings
- Trained model weights
- other

### `experiment.py`
Interactive script to test the fine-tuned model on custom dialogue inputs.

## Install Dependencies 
pip3 install -r requirements.txt


   







