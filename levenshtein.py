# I have used distance here because the ratio function causes errors
# The ratio function decides that a replace is a cost of two when it should only be a cost of one like the distance function uses
# Therefore I have used the distance function and simply found which string is longest and divided the distance by that which is how the algorithm is intended to work

def levenshteinSimilarityCharacter(output1,output2):
    from Levenshtein import distance
    if len(output1) > len(output2):
        maxLength = len(output1)
    else:
        maxLength = len(output2)
    distance = distance(output1, output2)
    ratio = distance/maxLength
    similarity = 1 - ratio # High similarity means good
    return(similarity)

def levenshteinSimilarityWord(output1,output2):
    output1 = output1.split()
    output2 = output2.split()
    if len(output1) > len(output2):
        maxLength = len(output1)
    else:
        maxLength = len(output2)
    distance = levenshtein_distance_words(output1, output2)
    ratio = distance/maxLength
    similarity = 1 - ratio  # High similarity means good
    return(similarity)

def levenshtein_distance_words(words1, words2):
    d = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(len(words1) + 1):
        d[i][0] = i
    for j in range(len(words2) + 1):
        d[0][j] = j

    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + cost)

    return d[len(words1)][len(words2)]


