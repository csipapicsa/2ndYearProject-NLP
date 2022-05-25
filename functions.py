import imports as i

# Read-in the files numbers module
import sys
# read-ins
import gzip
import json
# nn
import torch
# dict
from collections import defaultdict 
# DF
import numpy as np
import pandas as pd

# time
from datetime import datetime
# plot 
import matplotlib.pyplot as plt

import preprocessing as pp

# lemma init
wnl = i.WordNetLemmatizer()

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

def simplify_contraction(text_list):
    new_data = []
    for text in text_list:
        #print(text)
        exp_text = contractions.fix(text)
        new_data.append(exp_text)
    return new_data

def simplify_negation(text_list):
    new_data = []
    for text in text_list:
        #print(text)
        found_not = False
        antonym = ""
        if type(text) == str:
            words = text.split()
        else:
            words = text
        new_sent = []
        #print(words)
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
        #print(new_sent)
        #text = ' '.join(new_sent) # removed by Gergo
        new_data.append(text)
    return new_data

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
    

######## old one, doesn't use pos tags
# def lemmatize_sentencelist(sentencelist): 
#     wnl = WordNetLemmatizer()
#     lemmatized_sentences = []
#     for sentence in sentencelist: 
#         #print(sentence)
#         temp_sentence = []
#         if type(sentence) == str:
#             #print("its a string")
#             temp_sentence = [wnl.lemmatize(word) for word in sentence.split(" ")]
#             #print(temp_sentence)
#             #lemmatized_sentences.append(" ".join(temp_sentence))
#             lemmatized_sentences.append(temp_sentence)
#         else:
#             temp_sentence = [wnl.lemmatize(word) for word in sentence]
#             lemmatized_sentences.append(temp_sentence)
#     return lemmatized_sentences

# helper function for lemmatize_sentencelist
def get_wordnet_pos(postag):
    if postag.startswith('J'):
        return wordnet.ADJ
    elif postag.startswith('V'):
        return wordnet.VERB
    elif postag.startswith('N'):
        return wordnet.NOUN
    elif postag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN
# NEW ONE WITH POS_TAGS   
def lemmatize_sentencelist(sentencelist):
    lemmatized_sentences = []
    for s in sentencelist: 
        s = s.lower()
        pos_s = nltk.pos_tag(s.split())
        lemmatized_sentences.append(" ".join([wnl.lemmatize(w[0], get_wordnet_pos(w[1])) for w in pos_s]))
    return lemmatized_sentences

#only works for english
def pos_tag_stringlist(strlist, shouldTokenize):
    pos_tagged_strlist = []
    if shouldTokenize: 
        for str in strlist: pos_tagged_strlist.append(pos_tag(word_tokenize(str)))
    else: 
        for str in strlist: pos_tagged_strlist.append(pos_tag(str))
    return pos_tagged_strlist
    
def init_log_for_training():
    dateTimeObj = datetime.now()
    save_time = str(dateTimeObj.year)+'-'+str(dateTimeObj.month)+'-'+str(dateTimeObj.day)+'-'+str(dateTimeObj.hour)+'-'+str(dateTimeObj.minute)+'-'+str(dateTimeObj.second)
    columns = ["Running ID", "Model Name", "Expand Contractions", "Basic Preprocessing", 
           "Grammar Correction", "Simplify Negotiations", "Lemmatize", "Remove Stop Words", "No. of Sentences", 
           "Train Accuracy STOP", "Test Accuracy STOP", "Train Loss STOP", "Test Loss STOP",
           "Train_sentence_fully_catched_ratio", "Test_sentence_fully_catched_ratio",
           "Length of Sentence", "Batch size of RNN"]
    # dataframe
    results_dataframe = pd.DataFrame(columns=columns)
    return results_dataframe, save_time


if __name__ == "__main__":
    import sys
    import gzip
    import json
    
    
def plot_model_history(history):
    x = range(0,len(history.history["accuracy"]))
    labels = range(1,len(history.history["accuracy"])+1)
    plt.xticks(x, labels)
    plt.plot(history.history["accuracy"], 'g--', label='train accuracy')
    plt.plot(history.history["loss"], 'r--', label='train loss')
    plt.plot(history.history["val_loss"], 'r:', label='validation loss')
    plt.plot(history.history["val_accuracy"], 'g:', label='validation accuracy')
    plt.legend()
    plt.xlabel('Epochs', fontsize=15, color='black')
    plt.ylabel('Accuracy\nLoss', fontsize=15, color='black')
    plt.show()


# read in the oldest file:
def oldest_combinations():
    import glob
    import os

    list_of_files = glob.glob('results/*.csv')
    #print(list_of_files)# * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getmtime)
    #print(latest_file)
    # read in
    p = pd.read_csv(latest_file, index_col=[0])
    past_combinations = p[["Expand Contractions","Basic Preprocessing",
   "Grammar Correction","Simplify Negotiations", 
   "Lemmatize", "Remove Stop Words", "No. of Sentences"]].values
    return past_combinations

def lengths_catch(train, test, length_of_sentence=40):
    train_lengths = [len(word) for word in train if len(word) <= length_of_sentence]
    test_lengths = [len(word) for word in test if len(word) <= length_of_sentence]
    train_percent = len(train_lengths) / len(train)
    test_percent = len(test_lengths) / len(test)
    return train_percent*100, test_percent*100


def grid_search(train_list, test_list, y_train, y_test):
    simp_contr = [0, 1]
    gram_cor = [0, 1]
    simp_neg = [0, 1]
    lemma = [0,1] # since the function is wrong - by Gergo
    rem_stop = [0, 1]
    basic_preprocessing = 1
    list_of_data = []
    for z in simp_contr:
        for x in gram_cor:
            for c in simp_neg:
                for v in lemma:
                    for b in rem_stop:
                        train = train_list
                        test = test_list
                        if z == 1: # contractions
                            train = simplify_contraction(train)
                            test = simplify_contraction(test)
                        if basic_preprocessing == 1: # basic preprocessing
                            train = pp.basic_preprocess(train)
                            test = pp.basic_preprocess(test)
                        if x == 1: # grammar correction 
                            train = pp.grammar_corrector(train)
                            test = pp.grammar_corrector(test)
                        if c == 1: # Simnplyfy Negotiation 
                            train = simplify_negation(train)
                            test = simplify_negation(test)
                        if v == 1: # Lemmatize 
                            train = pp.lemmatize_sentencelist(train)
                            test = pp.lemmatize_sentencelist(test)
                        if b == 1: # Remove stop words
                            train = pp.remove_stop_words(train)
                            test = pp.remove_stop_words(test)

                        list_of_data.append([[z, basic_preprocessing, x, c, v, b], train, test]) #
    return list_of_data, y_train, y_test
    
    
def grid_search_retrain(train_list, test_list, y_train, y_test, possibility_matrix):
    simp_contr = [possibility_matrix[0]]
    gram_cor = [possibility_matrix[2]]
    simp_neg = [possibility_matrix[3]]
    lemma = [possibility_matrix[4]] # since the function is wrong - by Gergo
    rem_stop = [possibility_matrix[5]]
    basic_preprocessing = 1
    list_of_data = []
    for z in simp_contr:
        for x in gram_cor:
            for c in simp_neg:
                for v in lemma:
                    for b in rem_stop:
                        train = train_list
                        test = test_list
                        if z == 1: # contractions
                            train = simplify_contraction(train)
                            test = simplify_contraction(test)
                        if basic_preprocessing == 1: # basic preprocessing
                            train = pp.basic_preprocess(train)
                            test = pp.basic_preprocess(test)
                        if x == 1: # grammar correction 
                            train = pp.grammar_corrector(train)
                            test = pp.grammar_corrector(test)
                        if c == 1: # Simnplyfy Negotiation 
                            train = simplify_negation(train)
                            test = simplify_negation(test)
                        if v == 1: # Lemmatize 
                            train = pp.lemmatize_sentencelist(train)
                            test = pp.lemmatize_sentencelist(test)
                        if b == 1: # Remove stop words
                            train = pp.remove_stop_words(train)
                            test = pp.remove_stop_words(test)

                        list_of_data.append([[z, basic_preprocessing, x, c, v, b], train, test]) #
    return list_of_data, y_train, y_test
    
def statistics_sets_sizes(data_sets, filename="corp_size", max_len=40):
    results, time = init_log_for_training()
    # Sentencies max length:
    for data_set in data_sets:
        # Tokenizer
        labels = data_set[0]
        # check whenever combination is already checked. Working only with RNN!:
        print("Combinations: ", labels)

        tokenizer = pp.tokenizer_init(data_set[1], data_set[2])
        corpus_size = len(tokenizer.word_counts)

        ratio_train, ratio_test = lengths_catch(data_set[1], data_set[2], length_of_sentence=max_len)

        new_row = {'Running ID':time, 
               "Model Name":"-", 
              "Expand Contractions":labels[0],
              "Basic Preprocessing":labels[1],
              "Grammar Correction":labels[2],
               "Simplify Negotiations": labels[3],
              "Lemmatize": labels[4],
              "Remove Stop Words": labels[5],
              "No. of Sentences": len(data_set[1]),
                  "Train_sentence_fully_catched_ratio": ratio_train,
                   "Test_sentence_fully_catched_ratio": ratio_test,
                   "corpus_size": corpus_size,
                   "label_array": str(labels)}

        results = results.append(new_row, ignore_index=True)
        # maybe we dont need it in every round but how knows
        try:
            results.to_csv("results/"+filename+"_corp_sizes_"+time+".csv")
        except: 
            continue  
        # CLEAN
        del labels
        del tokenizer

    # save results again
    print("Combinations were checked")
    try:
        results.to_csv("results/"+filename+"_corp_sizes_"+time+".csv")
    except:
        pass
    return None
    ######################### NOTES ##########################    
### Reloading a function if its modified in a file
# import functions as f
# from importlib import reload
# reload(f)