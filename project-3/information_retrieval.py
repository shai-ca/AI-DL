from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# Initialize tokenizer and model for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased")

def compute_embedding(text):
    encoded_input = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(**encoded_input)
    embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1)
    return embeddings.numpy()

# Load a subset of the wikipedia dataset (assuming structure and availability)
dataset = load_dataset("Cohere/wikipedia-22-12-en-embeddings",split="train", streaming=True)

#========Exercise 3.1 =========== 
# Fill in the following code
# ===============================
def find_most_relevant_article(query_embedding, dataset,max_num_of_articles=None):
    most_relevant_article = None
    max_similarity = None
#========End Exercise 3.1 ===========
    return most_relevant_article, max_similarity


# Example input string
input_text = "Deep Learning"

# Compute the embedding for the input text
input_embedding = compute_embedding(input_text)
print(input_embedding.shape)

# Find the most relevant article
#To reduce the runtime, look at only the first N articles
article, similarity = find_most_relevant_article(input_embedding, dataset,1000)
print("Most Relevant Article:", article)
print("Similarity Score:", similarity)