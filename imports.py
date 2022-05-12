# imports

# readers
import gzip
import json

import codecs

# dataframe
import numpy as np
import pandas as pd


import random

# plot

import scipy.stats as stats
from sklearn.metrics import accuracy_score

# NN
## early stopping
import h5py
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# time
from datetime import datetime

# preprocessing
import re
from keras.preprocessing.text import Tokenizer
from collections import defaultdict

# spelling correction
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity

# reporting
from sklearn.metrics import accuracy_score

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# NN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import Pipeline

# develop
from importlib import reload