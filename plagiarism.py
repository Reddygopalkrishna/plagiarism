import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Tokenize the text and remove punctuation
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    return words

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def plagiarism_checker(text1, text2):
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Create a set of unique words in both texts
    word_set = set(words1 + words2)

    # Create vectors with word frequency
    vec1 = np.array([words1.count(word) for word in word_set])
    vec2 = np.array([words2.count(word) for word in word_set])

    # Calculate cosine similarity
    similarity = calculate_cosine_similarity(vec1, vec2)

    return similarity

if __name__ == "__main__":
    text1 = "This is an example sentence."
    text2 = "This is another sentence that is similar to the first one."

    similarity_score = plagiarism_checker(text1, text2)
    print("Similarity Score:", similarity_score)
