# imports
import string
import re

# we need to delete punctuations and some stop words
import string

# Tokenizer
from keras.preprocessing.text import Tokenizer

# Grammar corrector
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)


dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

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
        clean_text.append(d_sent)
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
    for sent in text:
        d_sent = [] # temp for sentence 
        for w in sent:
            if w in stop_words:
                None
            else:
                d_sent.append(w)
        clean_text.append(d_sent)
        length_of_sentencies_counter.append(len(d_sent))
        #print(d_sent)
    return clean_text #length_of_sentencies_counter
    
    
    
def tokenizer_train(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    text_to_sequence = tokenizer.texts_to_sequences(text)
    return tokenizer, text_to_sequence
    
def tokenizer_test(text, tokenizer):
    text_to_sequence = tokenizer.texts_to_sequences(text)
    return text_to_sequence
    
    
def grammar_corrector(text):
    # spell checker turn everything into lovercase, since words with upper case letters give worst results
    cleaned_text = []
    for line in text:
        temp_line = []
        # FEED WITH LIST OF LIST OR SIMPLE SENTENCE
        if type(line) == str:
            for word in line.split():
                if word in ['!','?','.']: # keep the basic puntuations
                    temp_line.append(word)
                else:
                    suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST,max_edit_distance=2)
                    # if there is no suggestion append "UNK" token
                    if len(suggestions) == 0:
                        temp_line.append("UNK")
                    else:
                        temp_line.append(suggestions[0].term)
        else:
            for word in line:
                if word in ['!','?','.']: # keep the basic puntuations
                    temp_line.append(word)
                else:
                    suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST,max_edit_distance=2)
                    # if there is no suggestion append "UNK" token
                    if len(suggestions) == 0:
                        temp_line.append("UNK")
                    else:
                        temp_line.append(suggestions[0].term)
        cleaned_text.append(temp_line)
    return cleaned_text
    
    
def grammar_correction(text):
    cleaned_text = []
    for line in text:
        temp_line = []
        print(line)
        # FEED WITH LIST OF LIST OR SIMPLE SENTENCE
        if type(line) == str:
            for word in line.split():
            
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST,max_edit_distance=2)
                # if there is no suggestion append "UNK" token
                if len(suggestions) == 0:
                    temp_line.append("UNK")
                else:
                    temp_line.append(suggestions[0].term)
        else:
            for word in line:
                suggestions = sym_spell.lookup(word, Verbosity.CLOSEST,max_edit_distance=2)
                # if there is no suggestion append "UNK" token
                if len(suggestions) == 0:
                    temp_line.append("UNK")
                else:
                    temp_line.append(suggestions[0].term)
        print(temp_line)
        cleaned_text.append(temp_line)
    return cleaned_text