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
            train_dataset[train_word]['wf'],             
            train_dataset[train_word]['pl'],             
            train_dataset[train_word]['wl'],             
            train_dataset[train_word]['ef'],             
            train_dataset[train_word]['tdf']             
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
            test_dataset[test_word]['wf'],
            test_dataset[test_word]['pl'],
            test_dataset[test_word]['wl'],
            test_dataset[test_word]['ef'],
            test_dataset[test_word]['tdf']
        ]
        x_test.append(parameters)

        if test_word in glossary.keys():
            y_test.append(1)
        else:
            y_test.append(0)

        prediction = clf.predict(np.array(parameters).reshape(1, -1))
        if prediction[0]==1 and test_word in glossary.keys():
            print('   Passed hypothesis: ', test_word, '-- CORRECT')
        if prediction[0]==1 and test_word not in glossary.keys():
            print('   Passed hypothesis: ', test_word, '-- Incorrect!!!')

    # Mean accuracy on the given test sample and labels
    score = clf.score(x_test, y_test)
    print('Mean accuracy: ', score)

if __name__ == "__main__":
    main()                           
