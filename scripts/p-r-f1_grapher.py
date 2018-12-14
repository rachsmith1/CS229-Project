import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    print('#### Graph precision/recall/F1 as a function of cross-validation iterations ####')
    with open('../data/gnb_output_oversampled.json', 'r') as fp:          
        data = json.load(fp)
    precision_full = data['precision']
    recall_full = data['recall']
    f1_full = data['f1']
    precision = [precision_full[i][1] for i in range(len(precision_full)-1)]
    recall = [recall_full[i][1] for i in range(len(recall_full)-1)]
    f1 = [f1_full[i][1] for i in range(len(f1_full)-1)]
    k = np.arange(1, len(precision_full))
    plt.plot(k, precision, label='Precision', marker='o', c=(0.333, 0, 0))
    plt.plot(k, recall, label='Recall', marker='o', c=(0.667, 0, 0))
    plt.plot(k, f1, label='F1-Score', marker='o', c=(1, 0, 0))
    plt.legend(loc='upper left')
    plt.xlabel('Cross-Validation Iteration')
    plt.ylabel('Parameter Value')
    plt.title('Precision, Recall, and F1 Score for MLP Algorithm')
    plt.savefig('../data/gnb_p-r-f1_plot_oversampled.png')
if __name__ == "__main__":
    main()