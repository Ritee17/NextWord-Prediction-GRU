# 🎭 Next Word Prediction with GRU (Shakespeare's Hamlet)

A Deep Learning project that predicts the next word in a sequence of text, trained on the complete text of Shakespeare's *"Hamlet"*. This project explores the performance of **Gated Recurrent Units (GRU)** for sequence modeling and is deployed as an interactive web application using **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📌 Project Overview
This project focuses on Natural Language Processing (NLP) to teach a machine to "speak" like Shakespeare. By analyzing character-by-character or word-by-word sequences from *Hamlet*, the GRU model learns the probability distribution of the next word given a context window.

### Key Features
* **GRU Architecture:** Utilizes Gated Recurrent Units, which are often faster and more efficient than LSTMs for smaller datasets.
* **Dataset:** The NLTK Gutenberg corpus version of *Shakespeare's Hamlet*.
* **Text Generation:** Capable of generating Shakespearean-style text based on user input.
* **Interactive UI:** A user-friendly web interface built with Streamlit.
* **Performance Tracking:** Integrated with TensorBoard to visualize training metrics.

## 🧠 Model Architecture
The model is built using **Keras/TensorFlow** with the following layers:

1.  **Embedding Layer:** Transforms words into dense vectors (100 dimensions).
2.  **GRU Layer 1:** 150 units, returns sequences to pass context to the next layer.
3.  **Dropout Layer:** 20% dropout rate to mitigate overfitting.
4.  **GRU Layer 2:** 100 units, captures higher-level abstractions.
5.  **Dense Output:** Uses Softmax activation to predict the probability of the next word from the vocabulary size (4,818 words).

## 📂 Repository Structure

| File Name | Description |
| :--- | :--- |
| `experiments.ipynb` | Jupyter Notebook containing data extraction, preprocessing, and model training. |
| `hamlet.txt` | The raw text data used for training. |
| `next_word_GRU.h5` | The trained GRU model saved in HDF5 format. |
| `tokenizer.pickle` | The tokenizer object (saves word-to-index mappings). |
| `requirements.txt` | List of dependencies required to run the app. |

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/next-word-prediction-gru.git](https://github.com/your-username/next-word-prediction-gru.git)
cd next-word-prediction-gru

```

### 2. Set Up Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
streamlit run app.py

```

## 📊 Experiment Results

I trained the model for 150 epochs. The results highlight the aggressive learning capability of GRUs on small datasets:

* **Training Accuracy:** ~81% (Model memorized the text effectively)
* **Training Loss:** 0.73
* **Observations:** The model achieved high training accuracy quickly but showed signs of overfitting on the validation set, a common challenge with small, complex corpora like Shakespeare.

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** NLTK, NumPy, Pandas, Scikit-learn
* **Visualization:** TensorBoard, Matplotlib


```
