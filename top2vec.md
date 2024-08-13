Top2Vec is a powerful algorithm for topic modeling that automatically finds topics and also provides a way to visualize them. It uses word embeddings to find topics, which makes it different from traditional methods like LDA. Below is a step-by-step guide on how to use Top2Vec on your DataFrame and tokens.

### 1. **Install the Top2Vec Library**
   - First, you need to install the `top2vec` library if you haven't already.

   ```bash
   pip install top2vec
   ```

### 2. **Prepare Your Data**
   - If you already have a DataFrame `df` with a column `tokens` containing preprocessed tokens, you can proceed directly. If not, you should preprocess your text similarly to how it was done before.

### 3. **Using Top2Vec**
   - You can now create a Top2Vec model using your text data.

```python
from top2vec import Top2Vec

# Assume df['tokens'] contains your preprocessed text tokens
# If not already a list of strings, join tokens into strings
df['text_joined'] = df['tokens'].apply(lambda x: ' '.join(x))

# Fit the Top2Vec model
model = Top2Vec(documents=df['text_joined'].tolist(), speed="learn", workers=4)

# Display the number of topics found
print(f"Number of topics: {model.get_num_topics()}")
```

### 4. **Exploring the Topics**
   - After fitting the model, you can explore the topics found by Top2Vec.

```python
# Get the top 5 words for each topic
topic_words, word_scores, topic_scores, topic_nums = model.get_topics()

for i in range(len(topic_words)):
    print(f"Topic {topic_nums[i]}: {', '.join(topic_words[i][:10])}")
```

### 5. **Assign Topics to Documents**
   - You can find the dominant topic for each document and assign it to your DataFrame.

```python
# Find the topic for each document
document_topics, document_scores, document_topic_nums = model.get_documents_topics(df['text_joined'].tolist())

# Add the topic numbers to the DataFrame
df['dominant_topic'] = document_topic_nums

# If you want to relabel topics based on interpretation:
topic_labels = {
    0: "Change/Improvement",
    1: "Travel/Guidance",
    2: "Clarity/Preparation"
}

df['topic_label'] = df['dominant_topic'].map(topic_labels)

# Display the DataFrame with topics and labels
print(df[['text', 'dominant_topic', 'topic_label']])
```

### 6. **Visualize the Topics**
   - Top2Vec also provides a way to visualize the topics.

```python
# Visualize topics
model.visualize_topics()
```

### 7. **Further Exploration**
   - You can find similar documents, explore the nearest words to a given word, and more with Top2Vec.

```python
# Find documents similar to a specific topic
similar_docs, similarity_scores = model.search_documents_by_topic(topic_num=0, num_docs=5)

for i, doc in enumerate(similar_docs):
    print(f"Document {i+1}: {doc} (Score: {similarity_scores[i]})")
```

### Summary

This guide provides an overview of how to apply Top2Vec to your text data:

1. **Installation:** Install Top2Vec.
2. **Model Creation:** Fit the Top2Vec model on your text data.
3. **Topic Exploration:** Extract and explore topics.
4. **Document Assignment:** Assign topics to documents.
5. **Visualization:** Optionally visualize the topics.

Top2Vec is very flexible and generally provides high-quality, interpretable topics thanks to its use of word embeddings.
