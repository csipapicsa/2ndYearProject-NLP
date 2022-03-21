import codecs

def read_arto(file_name):
    """
    read in conll file
    
    :param file_name: path to read from
    :returns: list with sequences of words and labels for each sentence
    """
    current_words = []
    current_tags = []
    data = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            if line[0] == '#':
                continue # skip comments
            tok = line.split('\t')
            word = tok[1]
            tag = tok[2]

            current_words.append(word)
            current_tags.append(tag)
        else:
            if current_words:  # skip empty lines
                data.append((current_words, current_tags))
            current_words = []
            current_tags = []

    # check for last one
    if current_tags != [] and not raw:
        data.append((current_words, current_tags))
    return data