import json
import numpy as np
from sklearn.naive_bayes import GaussianNB

def main():                                              

    print('#### Gaussian Naive Bayes ####')

    # Open glossary                                      
    with open('data/glossary.json', 'r') as fp:          
        glossary = json.load(fp)                         
    # Open train dataset                                 
    with open('data/train_dataset.json', 'r') as fp:     
        train_dataset = json.load(fp)                    
    # Open test dataset                                  
    with open('data/test_dataset.json', 'r') as fp:      
        test_dataset = json.load(fp)                     
                                                         
    # Make classifier training sample inputs             
    x_train = []                                         
    y_train = []                                         
    for train_word in train_dataset.keys():
        parameters = [                                   
            train_dataset[train_word]['TF'],             
            train_dataset[train_word]['DF'],             
            train_dataset[train_word]['TF']*np.log(52/train_dataset[train_word]['DF']),             
            train_dataset[train_word]['word_length']/train_dataset[train_word]['ngram_length'],
            #train_dataset[train_word]['ngram_length'],
            #train_dataset[train_word]['chapter'],
            #train_dataset[train_word]['section'],
            #train_dataset[train_word]['paragraph'],
            (train_dataset[train_word]['sentence_loc']-0.5)**2,
            train_dataset[train_word]['top_freq'],
            train_dataset[train_word]['TF_mean'],
            train_dataset[train_word]['TF_var'],
            train_dataset[train_word]['pos']
        ]                                                
        x_train.append(parameters)                       
                                                         
        if train_word in glossary.keys():                
            y_train.append(1)                            
        else:                                            
            y_train.append(0) 

    # Make and fit a Gaussian naive bayes model
    clf = GaussianNB(priors=None, var_smoothing=1e-09)
    print ('Training...')
    clf.fit(x_train, y_train)

    # Make testing sample inputs
    x_test = []
    y_test = []
    print ('Predicting...')
    for test_word in test_dataset.keys():
        parameters = [
            test_dataset[test_word]['TF'],
            test_dataset[test_word]['DF'],
            test_dataset[test_word]['TF']*np.log(52/test_dataset[test_word]['DF']),
            test_dataset[test_word]['word_length']/test_dataset[test_word]['ngram_length'],
            #test_dataset[test_word]['ngram_length'],
            #test_dataset[test_word]['chapter'],
            #test_dataset[test_word]['section'],
            #test_dataset[test_word]['paragraph'],
            (test_dataset[test_word]['sentence_loc']-0.5)**2,
            test_dataset[test_word]['top_freq'],
            test_dataset[test_word]['TF_mean'],
            test_dataset[test_word]['TF_var'],
            test_dataset[test_word]['pos']
        ]
        x_test.append(parameters)

        if test_word in glossary.keys():
            y_test.append(1)
        else:
            y_test.append(0)

        prediction = clf.predict(np.array(parameters).reshape(1, -1))
        if prediction[0]==1 and test_word in glossary.keys():
            print('   Passed hypothesis: ', test_word, '-- IN glossary!!!')
        if prediction[0]==1 and test_word not in glossary.keys():
            print('   Passed hypothesis: ', test_word, '-- Nope')
        if prediction[0]==0 and test_word in glossary.keys():
            print('   Missed: ', test_word)

    # Mean accuracy on the given test sample and labels
    score = clf.score(x_test, y_test)
    print('Mean accuracy: ', score)

if __name__ == "__main__":
    main()                           
