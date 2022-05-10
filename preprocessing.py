# imports
import string
import re

# we need to delete punctuations and some stop words
import string

def basic_preprocess(text):
    my_stop_words = ['$',"'","``","''","'s"]
    whitespace = [' ', '\t', '\n', '\r', '\x0b', '\x0c']
    punct = ['"','#','$','%', '&', "'",'(', ')', '*', '+', ',', '-', '/', ':', ';', '<', '=', '>', '@', '[', '\\',
             ']', '^', '_', '`', '{', '|', '}', '~'] # ? ! . are deleted
    
    stop_words_mess = ['\n', 'the', '\x1bthis']

    stop_words = set(list(my_stop_words) + list(string.digits) + punct + stop_words_mess)
    clean_text = []
    length_of_sentencies_counter = []
    
    for sent in text:
        # add whitespaces between punctuations, etc to be able to remove them
        sent = re.sub('(?<! )(?=[.,!?()~${}"|#&%*@\'^+-/_0123456789:>`<;=\[\]])|(?<=[.,!?()~${}"|#&%*@\'^+-/_0123456789:>`<;=\[\]])(?! )', r' ', sent)
        d_sent = []
        for c in sent.split():
            if c in stop_words:
                None
            else:
                d_sent.append(c)
        clean_text.append(list(d_sent))
        length_of_sentencies_counter.append(len(d_sent))
        #print(d_sent)
    return clean_text,length_of_sentencies_counter #length_of_sentencies_counter
    
    
def remove_stop_words(text): 
    stop_words = ['he','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

    clean_text = [] # for the whole set
    length_of_sentencies_counter = []
    d_sent = []
    for sent in text:
        d_sent = [] # temp for sentence 
        for c in sent:
            if c in stop_words:
                None
            else:
                d_sent.append(c)
        clean_text.append(list(d_sent))
        length_of_sentencies_counter.append(len(d_sent))
        #print(d_sent)
    return clean_text,length_of_sentencies_counter #length_of_sentencies_counter