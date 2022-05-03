# Read-in the files numbers module
import sys
# read-ins
import gzip
import json
# nn
import torch
# dict
from collections import defaultdict 
#
import numpy as np

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

import matplotlib.pyplot as plt
import numpy as np

### simple statistics about the sets
def simple_stat(data, minbins=False):
    # input is an dataset of sentences
    lengths = []
    for i in data:
        lengths.append(len(i))
    # bins
    bins = []
    bin = [10, 20, 30,40,50,70,80,90,100,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,
           1700,1800,1900,2000,2200,2400,2600,2800,3000]
    #bin = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
    if minbins==True:
        print("##### SHOW NUMBER OF SHORTEST SENTENCES ONLY #####")
        bins = [0,1,2,3,4,5,6,7,8,9,10]
        counts, edges, bars = plt.hist(lengths, bins = bins)
        plt.bar_label(bars)
    else:
        plt.hist(lengths, bins = bin, alpha=0.7)
        plt.hist(lengths, bins = range(1,3400,10), color = "black", alpha=0.7)
    print("length of longest sentence: ",max(lengths))
    print("length of shortest sentence: ",min(lengths))    
    plt.ylabel("Count of instancies")
    plt.xlabel("Count of words")
    print("average number of words in the sentence")
    print(np.average(lengths))
    plt.show()
    print("###################\n")
    return None

import contractions
from nltk.corpus import wordnet

def concat_and_negation(text):
    exp_text = contractions.fix(text)
    found_not = False
    antonym = ""
    words = exp_text.split()
    new_sent = []
    for word in words:
        if word == "not":
            found_not = True
        elif found_not:
            found_not = False
            syns = wordnet.synsets(word)
            if len(syns) > 0:
                syns = wordnet.synsets(word)[0]
                if syns.pos() == "a" or syns.pos() == "r":
                    for syn in wordnet.synsets(word):
                        for l in syn.lemmas():
                            if l.antonyms():
                                antonym = (l.antonyms()[0].name())
                            else:
                                new_sent.append("not")
                                new_sent.append(word)
                    new_sent.append(antonym)
                else:
                    new_sent.append("not")
                    new_sent.append(word)
            else:
                new_sent.append("not")
                new_sent.append(word)
        else:
            new_sent.append(word)
    exp_text = ' '.join(new_sent)
    return exp_text

def print_short_sentence(data, length=1):
    # length defines the maximum length of a sentence
    rangee = np.arange(0, length+1, dtype=int)
    for i in data:
        if type(i) == str: # for the train sent
            if len(i.split()) in rangee:
                print(i)
        elif type(i) == list:
            if len(i) in rangee:
                print(i)

def json_divide(json_file):
    sent_dict = {"positive": 1, "negative": 0}
    data = json_file
    train_sent = []
    train_sentiment = []
    train_idx = []
    missing_indexies = []
    y_train = []
    length_of_sentencies_counter = []
    for i in range(len(data)):
        try:
            train_sent.append(data[i]["reviewText"])
            train_sentiment.append(data[i]["sentiment"])
            train_idx.append(i)
            y_train.append(sent_dict[data[i]["sentiment"]])
        #length_of_sentencies_counter.append(len(data[i]["reviewText"].split()))
        except KeyError:
            missing_indexies.append(i)
            continue
    return train_sent, train_sentiment, train_idx, missing_indexies
    

if __name__ == "__main__":
    import sys
    import gzip
    import json

######################### NOTES ##########################    
### Reloading a function if its modified in a file
# import functions as f
# from importlib import reload
# reload(f)