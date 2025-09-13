import gzip
import pickle
from sklearn.neighbors import KNeighborsClassifier


def ncd(x, y):
    x_compressed = len(gzip.compress(x.encode()))
    y_compressed = len(gzip.compress(y.encode()))
    xy_compressed = len(gzip.compress((" ".join([x, y])).encode()))
    return (xy_compressed - min(x_compressed, y_compressed)) / max(x_compressed, y_compressed)

value1 = "I love going to the park"
value2 = "I Like going to the park"
print(ncd(value1, value2))
