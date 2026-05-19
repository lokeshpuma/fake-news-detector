# 📰 AI Fake News Detector

Welcome to the **AI Fake News Detector** project! This repository contains a Machine Learning-powered web application designed to help users identify unreliable or fake news articles based on their author and title.

## 🌟 Project Overview
With the rapid spread of misinformation, identifying fake news has become crucial. This project leverages Natural Language Processing (NLP) and Machine Learning techniques to classify news articles as Real or Fake. 

The core model is trained using Python, Scikit-Learn, and NLTK (for stemming and text processing). We extract features from the news text using TF-IDF Vectorization and then apply a classification algorithm to predict the news's authenticity.

## 📊 Dataset
Due to GitHub's file size limits, the `train.csv` dataset is hosted externally. You can download the dataset here:
[Download train.csv from Google Drive](https://drive.google.com/file/d/1edqVek---c_bcnxonMaxT5rI0a5KNepT/view?usp=drive_link)

## ✨ Features
- **Interactive UI**: Built with Streamlit for a fast and beautiful web interface.
- **NLP Text Processing**: Uses `PorterStemmer` and stop-words removal to process incoming text.
- **Real-time Prediction**: Instantly predicts whether a news piece is likely Real or Fake.

## 🛠️ Built With
- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) - For the web interface.
- [Scikit-Learn](https://scikit-learn.org/) - For machine learning models.
- [NLTK](https://www.nltk.org/) - For text processing and stemming.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - For data manipulation.

## 🚀 Getting Started

### Prerequisites
Make sure you have [Conda](https://docs.conda.io/en/latest/) or Python installed on your local machine.

### Installation & Setup

1. **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/your-username/fake-news-detector.git
    cd fake-news-detector
    ```

2. **Activate your environment**:
    If you're using conda, activate your environment (e.g., `tf`):
    ```bash
    conda activate tf
    ```

3. **Install Dependencies**:
    Install all required Python libraries via the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure Models are Present**:
    Make sure `model.pkl` (and ideally `vectorizer.pkl`) are in the same directory.
    *(Note: The TF-IDF vectorizer must be fitted on your training dataset and saved as `vectorizer.pkl` for the input text transformation to work correctly!)*

### Running the App
Start the Streamlit application by running the following command in your terminal:
```bash
streamlit run app.py
```
This will launch the app in your default web browser (usually at `http://localhost:8501`).

## 💡 How it works
1. **Input**: The user inputs the Author Name and the News Title.
2. **Preprocessing**: The inputs are combined, lowercased, stripped of non-alphabetic characters, and stemmed to their root words while removing stop words.
3. **Vectorization**: The processed string is converted into a numerical vector using TF-IDF.
4. **Prediction**: The Machine Learning model processes the vector and classifies the text as either `Real` (0) or `Fake` (1).

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
