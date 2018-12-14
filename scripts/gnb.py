import json
import numpy as np
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB

def main():                                              
    print('#### Gaussian Naive Bayes ####')

    # Open glossary                                      
    with open('../data/glossary.json', 'r') as fp:          
        glossary = json.load(fp)                         
    # Open dataset                                 
    with open('../data/dataset.json', 'r') as fp:     
        dataset = json.load(fp)
    # Create output log file (dictionary)
    output = {'acc':[[0,0]], 'tp':[[0,0]], 'fp':[[0,0]], 'tn':[[0,0]], 'fn':[[0,0]], 'precision':[[0,0]], 'recall':[[0,0]], 'f1':[[0,0]], 'words':OrderedDict()}                    

    inputs = list(dataset.keys())
    np.random.seed(100)
    np.random.shuffle(inputs)

    # Make classifier inputs             
    x = []
    y = []                                                                                 
    for word in inputs:
        parameters = [                                   
            dataset[word]['TF'],
            dataset[word]['TF']*dataset[word]['IDF'],
            dataset[word]['PO'],
            dataset[word]['PS'],
            dataset[word]['OT'],
            dataset[word]['OMT'],
            dataset[word]['OTFW']
        ]                                                
        x.append(parameters)                    
        if word in glossary.keys():
            output['words'][word] = [[1, 1, -1]] 
            y.append(1)
        else:
            output['words'][word] = [[0, 0, -1]]
            y.append(0)
    
    #Split into train/dev and test sets
    trainPercentage = 0.9
    howManyToTrain = int(round(trainPercentage*len(inputs)))
    
    x_train = x[:howManyToTrain]
    y_train = y[:howManyToTrain]
    words_train = inputs[:howManyToTrain]
    x_test = x[howManyToTrain:]
    y_test = y[howManyToTrain:]
    words_test = inputs[howManyToTrain:]
    
    #k-fold cross-validation
    k = 10
    dev_size = int(len(x_train)/k)
    for i in range(k):
        print("Cross-Validation Round "+str(i+1)+" of "+str(k))
        inputs_dev = x_train[dev_size*i:dev_size*(i+1)]
        labels_dev = y_train[dev_size*i:dev_size*(i+1)]
        words_dev = words_train[dev_size*i:dev_size*(i+1)]
        inputs_train = x_train[0:dev_size*i]+x_train[dev_size*(i+1):len(x_train)]
        labels_train = y_train[0:dev_size*i]+y_train[dev_size*(i+1):len(x_train)]
        train_words = words_train[0:dev_size*i]+words_train[dev_size*(i+1):len(words_train)]
        #Fit model
        clf = GaussianNB(priors=None)
        print ('Training...')
        clf.fit(inputs_train, labels_train)
        print('Predicting...')
        tp, fp, tn, fn = 0, 0, 0, 0
        fp_list = []
        for j in range(dev_size):
            prediction = clf.predict(np.array(inputs_dev[j]).reshape(1, -1))
            label = labels_dev[j]
            if prediction[0]==1 and label==1:
                tp += 1
            if prediction[0]==1 and label!=1:
                fp += 1
                fp_list.append((j, words_dev[j]))
            if prediction[0]==0 and label==1:
                fn += 1
            if prediction[0]==0 and label!=1:
                tn += 1
            output['words'][words_dev[j]][i][2] = int(prediction[0])
        print("It's time to check the "+str(len(fp_list))+" false positives! "
          "Type 'y' if the word should be in the glossary, 'p' to go back one word, "
          "or press any key otherwise.")
        change_counter = 0
        fp_index = 0
        while fp_index<len(fp_list):
            b = input(str(fp_index)+". "+fp_list[fp_index][1]+": ")
            if b.lower()=='y':
                change_counter = change_counter + 1
                output['words'][fp_list[fp_index][1]][i][1] = 1
                y_train[i*dev_size+fp_list[fp_index][0]] = 1
                fp_index = fp_index + 1
            elif b.lower()=='p':
                fp_index = max(0, fp_index - 1)
                print("Going back one word.")
            else:
                fp_index = fp_index + 1
        output['acc'][i][0] = (tp+tn)/(tp+fp+tn+fn)
        output['tp'][i][0] = tp
        output['fp'][i][0] = fp
        output['tn'][i][0] = tn
        output['fn'][i][0] = fn
        if (tp+fp)==0:
            precision = float('nan')
        else:
            precision = tp/(tp+fp)
        output['precision'][i][0] = precision
        if (tp+fn)==0:
            recall = float('nan')
        else:
            recall = tp/(tp+fn)
        output['recall'][i][0] = recall
        if precision==float('nan') or recall==float('nan'):
            output['f1'][i][0] = float('nan')
        else:
            output['f1'][i][0] = (2*precision*recall)/(precision+recall)
        tp = tp + change_counter
        fp = fp - change_counter
        output['acc'][i][1] = (tp+tn)/(tp+fp+tn+fn)
        output['tp'][i][1] = tp
        output['fp'][i][1] = fp
        output['tn'][i][1] = tn
        output['fn'][i][1] = fn
        if (tp+fp)==0:
            precision = float('nan')
        else:
            precision = tp/(tp+fp)
        output['precision'][i][1] = precision
        if (tp+fn)==0:
            recall = float('nan')
        else:
            recall = tp/(tp+fn)
        output['recall'][i][1] = recall
        if precision==float('nan') or recall==float('nan'):
            output['f1'][i][1] = float('nan')
        else:
            output['f1'][i][1] = (2*precision*recall)/(precision+recall)
        for word in inputs:
            output['words'][word].append([output['words'][word][i][1], output['words'][word][i][1], -1])
        output['acc'].append([0,0])
        output['tp'].append([0,0])
        output['fp'].append([0,0])
        output['tn'].append([0,0])
        output['fn'].append([0,0])
        output['precision'].append([0,0])
        output['recall'].append([0,0])
        output['f1'].append([0,0])
        
    #Final test using the entire training set
    print("Final Test")
    clf = GaussianNB(priors=None)
    print ('Training...')
    clf.fit(x_train, y_train)
    print('Predicting...')
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    for j in range(len(y_test)):
        prediction = clf.predict(np.array(x_test[j]).reshape(1, -1))
        label = y_test[j]
        if prediction[0]==1 and label==1:
            tp += 1
        if prediction[0]==1 and label!=1:
            fp += 1
            fp_list.append((j, words_test[j]))
        if prediction[0]==0 and label==1:
            fn += 1
        if prediction[0]==0 and label!=1:
            tn += 1
        output['words'][words_test[j]][k][2] = int(prediction[0])
    print("It's time to check the "+str(len(fp_list))+" false positives! "
          "Type 'y' if the word should be in the glossary, 'p' to go back one word, "
          "or press any key otherwise.")
    change_counter = 0
    fp_index = 0
    while fp_index<len(fp_list):
        b = input(str(fp_index)+". "+fp_list[fp_index][1]+": ")
        if b.lower()=='y':
            change_counter = change_counter + 1
            output['words'][fp_list[fp_index][1]][k][1] = 1
            fp_index = fp_index + 1
        elif b.lower()=='p':
            fp_index = max(0, fp_index - 1)
            print("Going back one word.")
        else:
            fp_index = fp_index + 1
    output['acc'][k][0] = (tp+tn)/(tp+fp+tn+fn)
    output['tp'][k][0] = tp
    output['fp'][k][0] = fp
    output['tn'][k][0] = tn
    output['fn'][k][0] = fn
    if (tp+fp)==0:
        precision = float('nan')
    else:
        precision = tp/(tp+fp)
    output['precision'][k][0] = precision
    if (tp+fn)==0:
        recall = float('nan')
    else:
        recall = tp/(tp+fn)
    output['recall'][k][0] = recall
    if precision==float('nan') or recall==float('nan'):
        output['f1'][k][0] = float('nan')
    else:
        output['f1'][k][0] = (2*precision*recall)/(precision+recall)
    tp = tp + change_counter
    fp = fp - change_counter
    output['acc'][k][1] = (tp+tn)/(tp+fp+tn+fn)
    output['tp'][k][1] = tp
    output['fp'][k][1] = fp
    output['tn'][k][1] = tn
    output['fn'][k][1] = fn
    if (tp+fp)==0:
        precision = float('nan')
    else:
        precision = tp/(tp+fp)
    output['precision'][k][1] = precision
    if (tp+fn)==0:
        recall = float('nan')
    else:
        recall = tp/(tp+fn)
    output['recall'][k][1] = recall
    if precision==float('nan') or recall==float('nan'):
        output['f1'][k][1] = float('nan')
    else:
        output['f1'][k][1] = (2*precision*recall)/(precision+recall)
    with open('../data/gnb_output.json', 'w') as file_path:
        json.dump(output, file_path)

if __name__ == "__main__":
    main()                           
