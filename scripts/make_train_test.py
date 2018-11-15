import numpy as np
from collections import OrderedDict
import random
import json
import csv

# Machine learning libraries
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemma = nltk.wordnet.WordNetLemmatizer()

print('#### Making train/test datasets... ####')

# Dictionary where we will save the information for our uni/bi/tri-grams
train_test_dict = OrderedDict()

# Path to the csv file with the top 5000 English words
top5000_path = 'data/top5000.csv'

# Make a dictionary called 'top5000words' with the top 5000 English words
with open(top5000_path) as f:
    reader = csv.reader(f, delimiter="\t")
    top5000 = list(reader)
top5000_split = [entry[0].split(',') for entry in top5000]
top5000words = OrderedDict()
for entry in top5000_split:
    top5000words[entry[0]] = {'freq':int(entry[2])}

# Path to file with textbook sentences
f_path = 'data/textbook-sentences.txt'

# Open the file with textbook sentences and read them into an array
f = open(f_path, 'r')
f_data = f.read()
sentences = f_data.split('\n')

# Loop through the sentences
for sentence in sentences:
    # Split the 'sentence' into the sentence position and content
    # We only want the sentence if it HAS a position and content
    sentence_split = sentence.split('\t')
    if len(sentence_split)!=2: continue
    
    position, content = sentence_split[0], str(sentence_split[1])
    
    # Split the sentence's position into indices
    # We only want indices that have chapter, section, and paragraph info
    # We want indices that are integers
    indices = position.split('.')
    if len(indices)<4: continue

    indices_are_integers = True
    for i, ix_str in enumerate(indices):
        try:
            ix_int = int(ix_str)
            indices[i] = ix_int
        except ValueError:
            #Handle the exception
            indices_are_integers = False

    if (indices_are_integers==False): continue

    # Use scikitlearn to split the sentence into uni/bi/tri-grams
    # Save those strings in the train/test dictionary with their associated features

    unigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1))
    unigram_analyze = unigram_vectorizer.build_analyzer()
    unigrams = unigram_analyze(content)

    bigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2))
    bigram_analyze = bigram_vectorizer.build_analyzer() 
    bigrams = bigram_analyze(content)

    trigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(3,3))
    trigram_analyze = trigram_vectorizer.build_analyzer()
    trigrams = trigram_analyze(content)

    grams = [unigrams, bigrams, trigrams]

    # 'Lemmatization': reduce word to its lemma
    for i, gram in enumerate(grams):
        for j, string in enumerate(gram):
            words = string.split(' ')
            words = [lemma.lemmatize(word) for word in words]
            new_string = ' '.join(words)
            gram[j] = new_string
    

    # Check if the string is in our train/test dictionary
    for i, gram in enumerate(grams):
        for j, string in enumerate(gram):

            # We definitely don't want these words in our glossary
            has_bad_word = False
            bad_words = ['chapter', 'summary', 'figure', 'appendix', 'example', 'key concept', 'learning outcomes', 'term']
            for bad_word in bad_words:
                if bad_word in string:
                    has_bad_word = True
            if has_bad_word: continue
            # We don't want numbers in our glossary
            if any(k.isdigit() for k in string): continue

            if string not in train_test_dict.keys():
                # Term frequency if it is in the top 5000 English words...
                top_freq = 1000
                if string in top5000words.keys():
                    top_freq = top5000words[string]['freq']

                # Term mean/var frequency across all chapters
                TF_chap = list(np.zeros(52))
                TF_chap[indices[1]-1] = 1
                TF_mean = np.mean(TF_chap)
                TF_var = np.var(TF_chap)

                # 'Part of speech', no not the other thing POS stands for
                pos = 0
                if nltk.pos_tag([string])[0][1]=='NN':
                    pos = 100

                train_test_dict[string] = {
                    'string':string,
                    'TF':1,
                    'TF_chap':TF_chap,
                    'TF_mean':TF_mean,
                    'TF_var':TF_var,
                    'DF':1,
                    'in_chapters':[indices[1]],
                    'word_length':len(string)/(i+1),
                    'ngram_length':i+1,
                    'chapter':indices[1],
                    'section':indices[2],
                    'paragraph':indices[3],
                    'sentence_loc': j/len(gram),
                    'top_freq':top_freq,
                    'pos':pos
                }
            else:
                train_test_dict[string]['TF_chap'][indices[1]-1] += 1
                train_test_dict[string]['TF_mean'] = np.mean(train_test_dict[string]['TF_chap'])
                train_test_dict[string]['TF_var'] = np.var(train_test_dict[string]['TF_chap'])

                if indices[1]==train_test_dict[string]['chapter']:
                    train_test_dict[string]['TF'] += 1
                
                if indices[1] not in train_test_dict[string]['in_chapters']:
                    train_test_dict[string]['in_chapters'].append(indices[1])
                    train_test_dict[string]['DF'] = len(train_test_dict[string]['in_chapters'])

# Remove strings that don't appear very often
#keys = list(train_test_dict.keys())
#for key in keys:
#    if train_test_dict[key]['TF'] < 3:
#        del train_test_dict[key]

# Sort the dictionary according to string document frequency
#sorted_train_test_dict = OrderedDict(sorted(train_test_dict.items(), key=lambda item: item[1]['DF']))
#for key in sorted_train_test_dict.keys():
#    print(sorted_train_test_dict[key])

print('Some train/test dataset entries: ')
print(train_test_dict['skeleton'])
print(train_test_dict['life'])
print(train_test_dict['cell'])
print(train_test_dict['cell membrane'])

# 90% of inputs for training
trainPercentage = 0.9

# Shuffle the inputs, then take the first 30% for testing, leave the rest for training
inputs = list(train_test_dict.keys())
random.shuffle(inputs)
howManyToTrain = int(round(trainPercentage*len(inputs)))

train_inputs = inputs[:howManyToTrain]
test_inputs = inputs[howManyToTrain:]

# Fill train dataset
train_dict = OrderedDict()
for key in train_inputs:
    train_dict[key] = train_test_dict[key]

# Fill test dataset
test_dict = OrderedDict()
for key in test_inputs:
    test_dict[key] = train_test_dict[key]

print('Entries in train/test dataset: ')
print('-- Train Dataset: {} entries'.format(len(train_dict)))
print('-- Test Dataset:  {} entries'.format(len(test_dict)))

# Save the newly made datasets
with open('data/train_dataset.json', 'w') as fp:
    json.dump(train_dict, fp)

with open('data/test_dataset.json', 'w') as fp:
    json.dump(test_dict, fp)
