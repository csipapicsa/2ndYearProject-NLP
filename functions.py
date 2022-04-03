# Read-in the files numbers module
import sys
# read-ins
import gzip
import json
# nn
import torch
# dict
from collections import defaultdict 

def skip_gram(corpus):    
    #from collections import defaultdict 
    #import torch

    ## global settings
    PAD = "<pad>"
    window_size=2

    ### your code here
    word2idx = defaultdict(int) # a vocabulary mapper: word to index, with <pad>
    word2idx[PAD] = 0 # reserve 0 for padding
    idx2word = [PAD]

    # create idxs
    curIdx = 1
    for sent in corpus:
        for word in sent.split(' '):
            if word not in word2idx:
                word2idx[word] = curIdx
                curIdx += 1
                idx2word.append(word)
    #print(word2idx)

    # create data (this can definitely be done with shorter/more efficient code)
    fullData = []
    labels = []
    for sent in corpus:
        sent = sent.split(' ')
        for tgtIdx in range(len(sent)):
            labels.append(word2idx[sent[tgtIdx]])
            dataLine = []
            # backwards
            for dist in reversed(range(1,window_size+1)):
                srcIdx = tgtIdx - dist
                if srcIdx < 0:
                    dataLine.append(word2idx[PAD])
                else:
                    dataLine.append(word2idx[sent[srcIdx]])
            # forwards
            for dist in range(1,window_size+1):
                srcIdx = tgtIdx + dist
                if srcIdx >= len(sent):
                    dataLine.append(word2idx[PAD])
                else:
                    dataLine.append(word2idx[sent[srcIdx]])
            fullData.append(dataLine)

    labels = torch.tensor(labels)
    fullData = torch.tensor(fullData)
    #print(fullData)
    #print(labels)
    # expected output
    train_X = None # a matrix of instances x context windows (numeric)
    train_y = None # the target labels, encoded in Keras to_categorical
    return labels, fullData, train_X, train_y #train x and y are empty right now


def readJson(path):
    c = 0
    data = []
    keys = set()
    for line in gzip.open(path):
        review_data = json.loads(line)
        data.append(review_data)
        c +=1
    '''# returns back a list, where index is the review, and 
    the ["attribute"] is the values. for instance:
    d[0]["vote"] gives back 3, d[0]["reviewText"] is the review text'''
    print("Number of data: ",c)
    return data

    
def sent_result_converter(number):
    threshold = 0.5
    if number<threshold:
        return "negative"
    if number>threshold:
        return "positive"

        
if __name__ == "__main__":
    import sys
    import gzip
    import json