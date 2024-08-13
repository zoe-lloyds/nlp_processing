import pandas as pd
import string
import spacy
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Custom stopword list (add your own custom words here)
custom_stopwords = set([
    "team", "different", "change", "level", "place", 
    "terms", "months", "rmi", "rmp", "line", "part", "bank", 
    "better", "clear", "journey", "around", "together", 
    "advice", "ready", "open"
])

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize and create a spaCy doc object
    doc = nlp(text)
    
    # Lemmatize, remove stopwords, and remove custom words
    processed_tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in stop_words 
        and token.lemma_ not in custom_stopwords 
        and not token.is_punct 
        and not token.is_space
    ]
    
    # Join tokens back into a single string (or return as a list of tokens)
    return " ".join(processed_tokens)

def preprocess_dataframe(df, column_name):
    # Apply the preprocess_text function to each row in the specified column
    df[column_name] = df[column_name].apply(preprocess_text)
    return df

# Example Usage:
# Assuming you have a DataFrame `df` with a text column 'text_column_name'
# df = preprocess_dataframe(df, 'text_column_name')




Apart from using Gensim's LDA for topic modeling, there are several other libraries and models that you can use. Here’s an overview of some popular alternatives:

### 1. **scikit-learn**
   - **Techniques Available**: 
     - **Latent Dirichlet Allocation (LDA)**: Implemented using variational Bayes.
     - **Non-Negative Matrix Factorization (NMF)**: A linear algebra-based approach to topic modeling.
     - **Latent Semantic Analysis (LSA)**: Based on Singular Value Decomposition (SVD).
   - **Usage**: Scikit-learn is a versatile machine learning library that includes implementations of several topic modeling algorithms. It’s particularly user-friendly and integrates well with other scikit-learn functionalities.
   - **Example**:
     ```python
     from sklearn.decomposition import LatentDirichletAllocation, NMF
     from sklearn.feature_extraction.text import CountVectorizer

     # Example with LDA
     vectorizer = CountVectorizer(stop_words='english')
     X = vectorizer.fit_transform(documents)

     lda = LatentDirichletAllocation(n_components=10, random_state=0)
     lda.fit(X)

     # Example with NMF
     nmf = NMF(n_components=10, random_state=0)
     nmf.fit(X)
     ```

### 2. **Mallet**
   - **Techniques Available**: 
     - **LDA**: Mallet is a popular Java-based tool for performing LDA with Gibbs sampling.
   - **Usage**: Mallet is known for its efficiency and can often produce better results than other LDA implementations, especially for large datasets.
   - **Integration**: You can use Mallet with Python via Gensim, which has a wrapper for Mallet’s LDA implementation.
   - **Example**:
     ```python
     from gensim.models.wrappers import LdaMallet

     mallet_path = 'path/to/mallet'  # Update this with the path to your Mallet binary
     lda_mallet = LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)
     ```

### 3. **BERTopic**
   - **Techniques Available**:
     - **BERT-based Topic Modeling**: Uses embeddings from BERT (or other transformers) combined with clustering algorithms like HDBSCAN to create topics.
   - **Usage**: BERTopic is a modern approach to topic modeling that leverages transformer-based embeddings, making it effective for handling nuanced language and capturing more context.
   - **Example**:
     ```python
     from bertopic import BERTopic

     topic_model = BERTopic()
     topics, probs = topic_model.fit_transform(documents)
     ```

### 4. **spaCy + scikit-learn**
   - **Techniques Available**: 
     - **Custom LDA/NMF/LSA**: By using spaCy for advanced preprocessing (tokenization, lemmatization, etc.) combined with scikit-learn for topic modeling.
   - **Usage**: spaCy’s advanced NLP capabilities can be combined with scikit-learn’s topic modeling methods to create custom pipelines tailored to specific needs.
   - **Example**:
     ```python
     import spacy
     from sklearn.decomposition import NMF
     from sklearn.feature_extraction.text import CountVectorizer

     nlp = spacy.load("en_core_web_sm")
     processed_docs = [" ".join([token.lemma_ for token in doc if not token.is_stop]) for doc in nlp.pipe(documents)]

     vectorizer = CountVectorizer()
     X = vectorizer.fit_transform(processed_docs)

     nmf = NMF(n_components=10)
     nmf.fit(X)
     ```

### 5. **Top2Vec**
   - **Techniques Available**:
     - **Embedding-based Topic Modeling**: Top2Vec finds topics in documents by embedding documents and words in the same vector space and then clustering them.
   - **Usage**: This approach is very effective for finding coherent topics because it directly uses word embeddings and doesn't require predefining the number of topics.
   - **Example**:
     ```python
     from top2vec import Top2Vec

     model = Top2Vec(documents)
     topics = model.get_topics()
     ```

### 6. **LDA2Vec**
   - **Techniques Available**:
     - **Word2Vec + LDA Hybrid**: LDA2Vec combines LDA with Word2Vec embeddings to create topics that are informed by word embeddings.
   - **Usage**: This method captures the contextual relationships between words more effectively than traditional LDA.
   - **Example**: LDA2Vec is not as straightforward to implement as other methods and typically requires more customization. There are some implementations available on GitHub.

### 7. **TensorFlow Probability**
   - **Techniques Available**:
     - **Variational Autoencoders (VAE) for Topic Modeling**: Using deep learning methods for topic modeling.
   - **Usage**: TensorFlow Probability allows the use of probabilistic programming to create models like variational autoencoders for topic modeling.
   - **Example**: Implementation is more complex and typically requires a good understanding of TensorFlow and VAEs.

### 8. **Hugging Face Transformers**
   - **Techniques Available**:
     - **Custom Topic Models**: You can fine-tune transformer models for tasks related to topic extraction or leverage embeddings for clustering.
   - **Usage**: Hugging Face provides access to state-of-the-art NLP models that can be used to derive topics in combination with clustering techniques.
   - **Example**: You would typically generate embeddings and then use a clustering method to derive topics.

### 9. **TextBlob and NLTK**
   - **Techniques Available**:
     - **Basic Topic Modeling and Sentiment Analysis**: While not as advanced as LDA or BERT-based methods, TextBlob and NLTK can perform basic topic analysis through keyword extraction and n-grams.
   - **Usage**: These libraries are useful for quick, lightweight analysis and can be combined with more advanced methods for richer results.
   - **Example**:
     ```python
     from textblob import TextBlob

     blob = TextBlob("Your document text here")
     print(blob.noun_phrases)
     ```

### Conclusion
The choice of model depends on the size and nature of your data, the complexity of the language, and your specific goals. Gensim is a powerful and popular choice, but exploring these other models can help you achieve better or more tailored results.
