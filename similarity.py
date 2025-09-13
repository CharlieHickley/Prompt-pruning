# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm
from numpy import mean
from numpy import zeros
from re import sub





model = KeyedVectors.load_word2vec_format("C:/Users/clwhi/gensim-data/word2vec-google-news-300/word2vec-google-news-300/GoogleNews-vectors-negative300.bin", binary=True)


def get_cosineSimilarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))

def get_avrage_vector(text, model):
    words = sub(r'[^\w\s]', '', text)
    words = words.lower().split()
    # print(words)
    words_vector = []
    for word in words:
        try:
            words_vector.append(model[word])
            # print(words_vector)
        except:
            print('no embedding for: ' + word)
            continue
    if words_vector:
        return mean(words_vector, axis=0)
    else:
        return zeros(300)    
            
def get_similarity(text1, text2, model = model):
    vector1 = get_avrage_vector(text1, model)
    vector2 = get_avrage_vector(text2, model)
    return get_cosineSimilarity(vector1, vector2)



# with open('text1.txt', 'r') as file:
#     test1 = file.read()
    
# with open('text2.txt', 'r') as file:
#     test2 = file.read()

# similarity = get_similarity(test1, test2)
# print(similarity)


