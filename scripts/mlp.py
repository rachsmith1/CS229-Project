import json
import numpy as np
from collections import OrderedDict
from sklearn.neural_network import MLPClassifier

def main():                                              

    print('#### Multilayer perceptron ####')

    # Open glossary                                      
    with open('data/glossary.json', 'r') as fp:          
        glossary = json.load(fp)                         
    # Open dataset                                 
    with open('data/dataset.json', 'r') as fp:     
        dataset = json.load(fp)                    

    train_dataset = OrderedDict()
    test_dataset = OrderedDict()

    inputs = list(dataset.keys())
    np.random.seed(100)
    np.random.shuffle(inputs)

    trainPercentage = 0.8
    howManyToTrain = int(round(trainPercentage*len(inputs)))

    train_inputs = inputs[:howManyToTrain]
    test_inputs = inputs[howManyToTrain:]

    print('Making training set')
    for key in train_inputs:
        train_dataset[key] = dataset[key]

    print('Making testing set')
    for key in test_inputs:
        test_dataset[key] = dataset[key]

    # Make classifier training sample inputs             
    x_train = []                                         
    y_train = []                                         
    for train_word in train_dataset.keys():
        parameters = [                                   
            train_dataset[train_word]['TF'],
            train_dataset[train_word]['TF']*train_dataset[train_word]['IDF'],
            train_dataset[train_word]['PO'],
            train_dataset[train_word]['PS'],
            train_dataset[train_word]['OT'],
            train_dataset[train_word]['OMT'],
            train_dataset[train_word]['OTFW']
        ]                                                
        x_train.append(parameters)                       
                                                         
        if train_word in glossary.keys():                
            for i in range(9):
                x_train.append(parameters)
            for i in range(10):
                y_train.append(1)                            
        else:                                            
            y_train.append(0) 

    # Make and fit a Gaussian naive bayes model
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1, verbose=True)
    print ('Training...')
    clf.fit(x_train, y_train)

    # Make testing sample inputs
    tp, fp, tn, fn = 0, 0, 0, 0

    x_test = []
    y_test = []
    print ('Predicting...')
    for test_word in test_dataset.keys():
        parameters = [
            test_dataset[test_word]['TF'],
            test_dataset[test_word]['TF']*train_dataset[train_word]['IDF'],
            test_dataset[test_word]['PO'],
            test_dataset[test_word]['PS'],
            test_dataset[test_word]['OT'],
            test_dataset[test_word]['OMT'],
            test_dataset[test_word]['OTFW']
        ]
        x_test.append(parameters)

        if test_word in glossary.keys():
            y_test.append(1)
        else:
            y_test.append(0)

        prediction = clf.predict(np.array(parameters).reshape(1, -1))
        if prediction[0]==1 and test_word in glossary.keys():
            tp += 1
            print('   Passed hypothesis: ', test_word, '-- IN glossary!!!')
        if prediction[0]==1 and test_word not in glossary.keys():
            fp += 1
            print('   Passed hypothesis: ', test_word, '-- Nope')
        if prediction[0]==0 and test_word in glossary.keys():
            fn += 1
            print('   Missed: ', test_word)
        if prediction[0]==0 and test_word not in glossary.keys():
            tn += 1

    # Mean accuracy on the given test sample and labels
    score = clf.score(x_test, y_test)
    print('Mean accuracy: ', score)
    print('True positives:  ', tp)
    print('False positives: ', fp)
    print('True negatives:  ', tn)
    print('False negatives: ', fn)
    precision = tp/(tp+fp)
    print('Precision: ', precision)
    recall = tp/(tp + fn)
    print('Recall: ', recall)
    f1 = (2*precision*recall)/(precision+recall)
    print('F1: ', f1)

if __name__ == "__main__":
    main()                           
