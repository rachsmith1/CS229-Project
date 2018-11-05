import csv
import json
import numpy as np
import collections

print('#### Making glossary... ####')

# Note that this path only makes sense if you call the script in the 'project' folder
# i.e. not the 'scripts' folder
glossary_path = 'data/Glossary-v4.csv'

# Open the csv glossary file and separate it by 
with open(glossary_path) as f:
    reader = csv.reader(f, delimiter="\t")
    entries = list(reader)

# Split the entries into columns
entries_split = [entry[0].split('%') for entry in entries]
# Make a separate column for glossary words
glossary_words = [entry[0] for entry in entries_split]
# Make a separate column for glossary definitions
glossary_defns = [entry[5] for entry in entries_split]

# Fill a python dictionary with all the glossary words and their definitions
glossary = collections.OrderedDict()
for word, defn in zip(glossary_words, glossary_defns):
    if (defn!=''):
        glossary[word] = {'defn':defn}

# Save the glossary as a json file. Note this should be done above the 'data' folder
with open('data/glossary.json', 'w') as fp:
        json.dump(glossary, fp)
