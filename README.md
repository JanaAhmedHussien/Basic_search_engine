# Basic Search Engine

## Description
This project is a **Basic Search Engine** designed to retrieve relevant documents from a collection of text files. It implements core principles of information retrieval, including preprocessing, indexing, query processing, and ranking. The project demonstrates how search engines work and provides insights into effective data retrieval techniques.

---

## Features
- **Data Collection**: Gather and preprocess a corpus of text documents.
- **Preprocessing**: 
  - Tokenization
  - Lowercasing
  - Stopword removal
  - Stemming/Lemmatization
- **Indexing**: Create an inverted index to map terms to document IDs and term frequencies.
- **Query Processing**: 
  - Parse user queries.
  - Retrieve relevant documents using the inverted index.
  - Rank results using TF-IDF.
- **Query Expansion**: Incorporate relevance feedback and synonyms/related terms for enhanced search results.
- **User Interface**: 
  - A simple interface for inputting queries.
  - Display ranked search results.
- **Evaluation**: Test and evaluate accuracy, speed, and retrieval quality.

---

## Requirements
- **Programming Language**: Python
- **Libraries Used**:
  - NLTK: For text preprocessing (tokenization, stopword removal, etc.)
  - NumPy/Pandas: For data handling and calculations
  - Flask/Streamlit (optional): For building the user interface
  - Scikit-learn: For TF-IDF implementation

---

## Project Steps

### 1. **Data Collection**
- Collected a set of text documents (e.g., articles, web pages, or sample corpora).
- Ensured documents are in a consistent and parsable format.

### 2. **Preprocessing**
- **Tokenization**: Split text into individual words (tokens).
- **Lowercasing**: Converted text to lowercase for case insensitivity.
- **Stopword Removal**: Removed common words that do not contribute significantly to meaning (e.g., "and", "the").
- **Stemming/Lemmatization**: Reduced words to their base form for normalization (e.g., "running" → "run").

### 3. **Indexing**
- Built an **Inverted Index** to map terms to:
  - Document IDs where they appear.
  - Frequency of occurrence in each document.

### 4. **Query Processing**
- Implemented query parsing and preprocessing (same steps as above).
- Retrieved relevant documents containing all query terms using the inverted index.
- Ranked documents based on **TF-IDF** scores.

### 5. **Query Expansion**
- Analyzed top-ranked documents for relevance feedback.
- Integrated synonyms or related terms using embeddings like **ELMo** and **BERT** for semantic understanding.

### 6. **User Interface**
- Developed a basic interface (e.g., command-line, Flask, or Streamlit) to:
  - Accept user queries.
  - Display ranked search results with document titles/snippets.

### 7. **Evaluation**
- Tested the search engine with diverse queries.
- Evaluated retrieval accuracy, speed, and ranking effectiveness.

---

## Installation and Usage

### Prerequisites
1. Install Python (version 3.8+).
2. Install required libraries:
   ```bash
   pip install nltk numpy pandas flask sklearn
