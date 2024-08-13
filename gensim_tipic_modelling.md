Certainly! Let's walk through the process of performing topic modeling using Gensim, and then we'll visualize the results, including displaying the top keywords for each topic.

### Step 1: Install Required Libraries

If you haven't already, install the necessary libraries:

```bash
pip install gensim pandas nltk matplotlib pyLDAvis
```

### Step 2: Import Libraries and Load Data

Let's import the libraries and load your data into a DataFrame.

```python
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Sample data
data = {'text_column_name': [
    "The team is ready to make a better change in the bank.",
    "We went around the place and got some advice.",
    "The project will change the level of service.",
    "Different parts of the bank are working together.",
    "This is a journey that will improve the bank."
]}
df = pd.DataFrame(data)
```

### Step 3: Preprocess the Data

We'll use the `preprocess_text` function we discussed earlier to clean the data, and then prepare it for topic modeling.

```python
import string
import spacy

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))
custom_stopwords = set([
    "team", "different", "change", "level", "place", 
    "terms", "months", "rmi", "rmp", "line", "part", "bank", 
    "better", "clear", "journey", "around", "together", 
    "advice", "ready", "open"
])

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    processed_tokens = [
        token.lemma_ for token in doc 
        if token.lemma_ not in stop_words 
        and token.lemma_ not in custom_stopwords 
        and not token.is_punct 
        and not token.is_space
    ]
    return processed_tokens

df['processed_text'] = df['text_column_name'].apply(preprocess_text)
```

### Step 4: Prepare Data for LDA

Gensim's LDA model requires the data to be in a specific format. We need to create a dictionary and a corpus from the preprocessed text.

```python
# Create Dictionary
id2word = corpora.Dictionary(df['processed_text'])

# Create Corpus
texts = df['processed_text']

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
```

### Step 5: Build the LDA Model

Now we'll build the LDA model using Gensim.

```python
# Build LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=id2word,
                     num_topics=3,  # Set the number of topics
                     random_state=100,
                     update_every=1,
                     chunksize=10,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)
```

### Step 6: Display the Top Keywords for Each Topic

We can now display the top keywords for each topic.

```python
# Print the Keywords in the 3 topics
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}\n')
```

### Step 7: Visualize the Topics

One popular way to visualize the topics is using `pyLDAvis`, which provides an interactive visualization.

```python
# Visualize the topics
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.display(vis)
```

### Step 8: Plot the Number of Documents per Topic

Let's also create a bar chart showing the number of documents associated with each topic.

```python
import numpy as np

# Get the dominant topic for each document
dominant_topics = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(corpus))]

# Plot the number of documents per topic
plt.hist(dominant_topics, bins=np.arange(0, lda_model.num_topics + 1) - 0.5, edgecolor='black')
plt.xticks(range(lda_model.num_topics))
plt.xlabel('Topic')
plt.ylabel('Number of Documents')
plt.title('Number of Documents per Topic')
plt.show()
```

### Summary

1. **Top Keywords per Topic**: We printed the top keywords associated with each topic using the `print_topics()` method.
2. **Interactive Topic Visualization**: `pyLDAvis` was used to create an interactive visualization that helps in understanding the relationships between topics.
3. **Documents per Topic**: A histogram was created to show how many documents were associated with each topic.

This process should give you a good starting point for analyzing topics in your text data using Gensim and visualizing the results.
