import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import NMF
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt

# Sample documents (replace with your own corpus)
documents = [
    "apple banana orange fruit",
    "dog cat pet animal",
    "apple orange tropical fruit",
    "dog wolf canine animal",
    "banana orange yellow fruit",
    "cat feline pet animal",
    "apple banana smoothie fruit",
    "wolf pack wild canine"
]

# Tokenize the documents
tokenized_docs = [doc.split() for doc in documents]

# Step 1: Train Word2Vec on the tokenized documents
word2vec_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1, workers=4)
word_vectors = word2vec_model.wv  # Word2Vec vocabulary

# Step 2: Create document vectors by averaging the Word2Vec embeddings of each word in the document
def document_vector(doc):
    # Filter words that are in the Word2Vec model's vocabulary
    doc = [word for word in doc if word in word_vectors]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean([word_vectors[word] for word in doc], axis=0)

# Create document embedding matrix
doc_vectors = np.array([document_vector(doc) for doc in tokenized_docs])

# Step 3: Function to calculate coherence score for NMF applied to Word2Vec embeddings
def calculate_coherence_score_word2vec(num_topics, doc_vectors, tokenized_docs, dictionary, word2vec_model):
    # Apply NMF on document embeddings
    nmf_model = NMF(n_components=num_topics, random_state=42)
    W = nmf_model.fit_transform(doc_vectors)  # Document-topic matrix
    H = nmf_model.components_  # Topic-term matrix
    
    # Get top words for each topic from NMF components
    top_words_per_topic = []
    num_top_words = 5  # Number of top words per topic
    for topic_idx, topic in enumerate(H):
        top_word_indices = topic.argsort()[:-num_top_words-1:-1]
        top_words = [word2vec_model.wv.index_to_key[i] for i in top_word_indices]
        top_words_per_topic.append(top_words)

    # Calculate coherence score using Gensim's CoherenceModel
    coherence_model = CoherenceModel(
        topics=top_words_per_topic,
        texts=tokenized_docs,  # Use original tokenized documents for coherence calculation
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    return coherence_score

# Step 4: Create a Gensim dictionary for coherence calculation
dictionary = Dictionary(tokenized_docs)

# Step 5: Loop through different numbers of topics and calculate coherence scores
topic_range = range(2, 11)  # Testing between 2 and 10 topics
coherence_scores = []

for num_topics in topic_range:
    score = calculate_coherence_score_word2vec(num_topics, doc_vectors, tokenized_docs, dictionary, word2vec_model)
    coherence_scores.append(score)
    print(f"Number of Topics: {num_topics}, Coherence Score: {score}")

# Find the optimal number of topics based on the coherence score
optimal_num_topics = topic_range[np.argmax(coherence_scores)]
print(f"\nOptimal number of topics: {optimal_num_topics}")

# Plot coherence scores to visualize the optimal number of topics
plt.plot(topic_range, coherence_scores, marker='o')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.title('Coherence Score vs Number of Topics (Word2Vec Embeddings)')
plt.show()
