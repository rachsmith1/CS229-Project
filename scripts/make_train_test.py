import csv
import json
import numpy as np
import collections
import random

print('#### Make training and testing samples... ####')

# Paths to all the inputs
ftb1_path = 'data/ftb.concepts'
ftb2_path = 'data/ftb.concepts2'
ftb3_path = 'data/ftb.concepts3'
ftb_paths = [ftb1_path, ftb2_path, ftb3_path]

# Path to the csv file with the top 5000 English words
top5000_path = 'data/top5000.csv'

# Make a dictionary called 'top5000words' with the top 5000 English words
with open(top5000_path) as f:
    reader = csv.reader(f, delimiter="\t")
    top5000 = list(reader)
top5000_split = [entry[0].split(',') for entry in top5000]
top5000words = collections.OrderedDict()
for entry in top5000_split:
    top5000words[entry[0]] = {'freq':int(entry[2])}

# Make an array for the inputs
ftb_inputs = []

for path in ftb_paths:
    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")
        entries = list(reader)
    for entry in entries:
        ftb_inputs.append(entry)

# 70% of inputs for training, 30% for testing
testPercentage = 0.3

# Shuffle the inputs, then take the first 30% for testing, leave the rest for training
random.shuffle(ftb_inputs)
howManyToTest = int(round(testPercentage*len(ftb_inputs)))

test_inputs = ftb_inputs[:howManyToTest]
train_inputs = ftb_inputs[howManyToTest:]

# 'MakeParameters' makes the parameters for each input sample and returns a dictionary
# of processed input samples
def MakeParameters(inputs):
    inputs_split = [entry[0].split() for entry in inputs]

    dataset = collections.OrderedDict()
    for input in inputs_split:
        # (0) The word itself
        word = " ".join(input[1:])

        # (1) The frequency of the word (or phrase, tbh) in the textbook
        word_frequency = int(input[0])

        # (2) The length of the phrase
        word_phrase_length = len(input) - 1

        # (3) Number of characters in phrase
        word_num_characters = len(word)

        # (4) If in top 5000 English words, that word frequency
        word_english_freq = 1
        if word in top5000words.keys():
            word_english_freq = top5000words[word]['freq']

        # (5) Ratio of freqency_textbook/frequency_english_language
        word_tdf_sorta = word_frequency/word_english_freq

        dataset[word] = {
            'wf':word_frequency,
            'pl':word_phrase_length,
            'wl':word_num_characters,
            'ef':word_english_freq,
            'tdf':word_tdf_sorta
        }

    return dataset

# Save the newly made datasets
test_dataset = MakeParameters(test_inputs)
with open('data/test_dataset.json', 'w') as fp:
    json.dump(test_dataset, fp)

train_dataset = MakeParameters(train_inputs)
with open('data/train_dataset.json', 'w') as fp:
    json.dump(train_dataset, fp)
