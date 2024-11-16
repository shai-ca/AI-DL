import matplotlib.pyplot as plt
from datasets import load_dataset

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('stopwords')

dataset = load_dataset("cnn_dailymail", '3.0.0')
df = dataset['train'].to_pandas()
df = df.head(1000)

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to lower case
    tokens = [token.lower() for token in tokens]
    # Remove stop words (optional)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

###====== Part 2.1 =====================
###Write a code that creates two new columns -  artice_len and highlights_len


###====== Part 2.2 =====================
### Fill in this code
def plot_histograms(df):

    return None

# plot_histograms(df)

###======Part 2.3 ================
### Fill in the code
def ngrams(text, n):
    # Preprocess the text first
    processed_text = preprocess_text(text)
    words = processed_text.split()
    return set(zip(*[words[i:] for i in range(n)]))


def rouge_n(reference, candidate, n):
    return 0.0
###=========== 2.3 ================    

# Example of calculating Rouge-1 and Rouge-2 for a dataframe
df['rouge_1'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 1), axis=1)
df['rouge_2'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 2), axis=1)

plt.figure(figsize=(12, 6))
plt.hist(df['rouge_2'], bins=30, color='blue', alpha=0.7)
plt.title('Rouge-2 score distribution on ground truth')

max_rouge_2_index = df['rouge_2'].argmax()
print("Index of article with highest Rouge-2 score:", max_rouge_2_index)
print("========================\n")
print("Article with highest Rouge-2 score:", df.iloc[max_rouge_2_index]['article'])
print("========================\n\n\n")
print("Highlights with highest Rouge-2 score:", df.iloc[max_rouge_2_index]['highlights'])



###=========== 2.4 ================    
# Initialize the summarization pipeline
summarizer = None

def summarize_text(text):
    # Summarizing the text using the pipeline
    summary = summarizer(text, max_length=20, min_length=5, do_sample=False)
    print("-")
    return summary[0]['summary_text']

#Calculate the rouge-2 score of the first 10 entries