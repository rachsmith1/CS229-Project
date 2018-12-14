import numpy as np
from collections import OrderedDict
import json
import csv

# Machine learning libraries
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('wordnet')
lemma = nltk.wordnet.WordNetLemmatizer()

# ##################################################################
# First we will deal with the top 5000 words in the English language
# ##################################################################

# Path to top 5000 words
top5000_path = '../data/top5000.csv'

# Split csv file
with open(top5000_path) as f:
    reader = csv.reader(f, delimiter="\t")
    top5000 = list(reader)
top5000_split = [entry[0].split(',') for entry in top5000]

# Save the words and their frequency in the corpus into a dictionary
top5000words = {}
for entry in top5000_split:
    top5000words[entry[0]] = {'freq':int(entry[2])}

# ##################################################################
# Next deal with the chapter titles
# ##################################################################

# Path to file with textbook headers
headers_path = '../data/textbook-headers.txt'

# Open the file with textbook headers and read them into an array
headers_open = open(headers_path, 'r')
headers_data = headers_open.read()
headers = headers_data.split('\n')

# Have to manually input Chapter 51 for some reason...?
headers.append('7.51\tSalt and Water Balance and Nitrogen Excretion')

# We only want the chapters out of the headers
chapters = OrderedDict()
for header in headers:
    # Debug
    if len(header.split('\t')) != 2: continue
    
    # Split into header position and header content
    header_position, header_content = header.split('\t')[0], header.split('\t')[1]
    
    # We don't want 'Chapter #' in the chapter title
    header_content = header_content.replace('Chapter','')
    header_content = ''.join(i for i in header_content if not i.isdigit())

    # Save the chapter # and the chapter title into the chapter dictionary
    if len(header_position.split('.')) == 2:
        chapters[int(header_position.split('.')[1])] = {'chapter_title':header_content, 'chapter_length':0, 'chapter_terms':0}

# Vectorize, remove stop words, and lemmatize the chapter titles
unigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,1))
unigram_analyze = unigram_vectorizer.build_analyzer()

bigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(2,2))
bigram_analyze = bigram_vectorizer.build_analyzer()

trigram_vectorizer = CountVectorizer(stop_words='english', ngram_range=(3,3))
trigram_analyze = trigram_vectorizer.build_analyzer()

for k in chapters.keys():
    chapters[k]['unigrams'] = unigram_analyze(chapters[k]['chapter_title'])
    chapters[k]['unigrams'] = [lemma.lemmatize(unigram) for unigram in chapters[k]['unigrams']]
    chapters[k]['chapter_title'] = ' '.join(chapters[k]['unigrams']) 

    chapters[k]['bigrams'] = bigram_analyze(chapters[k]['chapter_title'])
    chapters[k]['trigrams'] = trigram_analyze(chapters[k]['chapter_title'])

# ##################################################################
# Finally deal with the textbook sentences
# We want our data sets to have the n-grams and their features
# ##################################################################

# Path to file with textbook sentences
sentences_path = '../data/textbook-sentences.txt'

# Open the file with textbook sentences and read them into an array
sentences_open = open(sentences_path, 'r')
sentences_data = sentences_open.read()
sentences = sentences_data.split('\n')

# Dictionary where we will save the information for our uni/bi/tri-grams
dataset = OrderedDict()

# Loop through the sentences
for sentence in sentences:
    # Debug
    if len(sentence.split('\t')) != 2: continue

    # Split into header position and header content
    sentence_position, sentence_content = sentence.split('\t')[0], sentence.split('\t')[1]

    # We want indices that are integers
    indices = sentence_position.split('.')
    indices_are_integers = True
    for i, ix_str in enumerate(indices):
        try:
            ix_int = int(ix_str)
            indices[i] = ix_int
        except ValueError:
            indices_are_integers = False
    if (indices_are_integers==False): continue

    # Find the length of the chapter
    chapters[indices[1]]['chapter_length'] += 1

    # Vectorize, remove stop words, and lemmatize the sentences
    unigrams = unigram_analyze(sentence_content)
    unigrams = [lemma.lemmatize(unigram) for unigram in unigrams]
    sentence_content = ' '.join(unigrams)

    bigrams = bigram_analyze(sentence_content)
    trigrams = trigram_analyze(sentence_content)

    # Find the number of terms in the chapter
    chapters[indices[1]]['chapter_terms'] += len(unigrams) + len(bigrams) + len(trigrams)

    for i, gram in enumerate([unigrams, bigrams, trigrams]):
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

            if string not in dataset.keys():
                dataset[string] = {}
                dataset[string]['string'] = string
                dataset[string]['index'] = len(dataset)-1

                # FEATURE #1: Term Frequency (TF)
                dataset[string]['TF'] = 1

                # FEATURE #2: Inverse Document Frequency (IDF)
                dataset[string]['in_chapters'] = [indices[1]]
                dataset[string]['IDF'] = np.log(len(chapters))

                # FEATURE #3: Position Ocurrence (PO)
                dataset[string]['PO'] = chapters[indices[1]]['chapter_length']

                # FEATURE #4: Position in Sentence (PS)
                dataset[string]['PS'] = j/len(gram)

                # FEATURE #5: Ocurrence in Title (OT)
                if string in chapters[indices[1]]['chapter_title']:
                    dataset[string]['OT'] = 1
                else:
                    dataset[string]['OT'] = 0

                # FEATURE #6: Ocurrence of Members in Title (OMT)
                dataset[string]['OMT'] = 0
                for substring in string.split(' '):
                    if substring in chapters[indices[1]]['chapter_title']:
                        dataset[string]['OMT'] += 1./len(string.split(' '))

                # FEATURE #7: Ocurrence in Top 5000 Words (OTFW)
                if string in top5000words.keys():
                    dataset[string]['OTFW'] = 1
                else:
                    dataset[string]['OTFW'] = 0

            else:
                # Update TF
                if indices[1] == dataset[string]['in_chapters'][0]:
                    dataset[string]['TF'] += 1

                # Update IDF
                if indices[1] not in dataset[string]['in_chapters']:
                    dataset[string]['in_chapters'].append(indices[1])
                    dataset[string]['IDF'] = np.log(len(chapters)/len(dataset[string]['in_chapters']))

for string in dataset.keys():
    # Update PO
    dataset[string]['PO'] = dataset[string]['PO'] / chapters[dataset[string]['in_chapters'][0]]['chapter_length']

    # Update TF
    dataset[string]['TF'] = dataset[string]['TF'] / chapters[dataset[string]['in_chapters'][0]]['chapter_terms']

# Save the newly made dataset
with open('../data/dataset.json', 'w') as fp:
    json.dump(dataset, fp)

#for k in chapters.keys():
#    print(k, chapters[k]['chapter_title'])


# EXAMPLES
print( dataset['cell'] )
print( dataset['protein'] )
print( dataset['carbohydrate'] )
print( dataset['lipid'] )











































