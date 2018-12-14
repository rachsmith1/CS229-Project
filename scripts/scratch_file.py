# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:35:06 2018

@author: Anita Kulkarni
"""
import json

with open('../data/mlp_output_untagged.json', 'r') as fp:          
    data = json.load(fp)
precision_full = data['precision']
recall_full = data['recall']
f1_full = data['f1']
sum_precision = 0
sum_recall = 0
sum_f1 = 0
k = len(precision_full)-1
for i in range(k):
    sum_precision = sum_precision + precision_full[i][0]
    sum_recall = sum_recall + recall_full[i][0]
    sum_f1 = sum_f1 + f1_full[i][0]
print("Average Precision: "+str(sum_precision/k))
print("Average Recall: "+str(sum_recall/k))
print("Average F1: "+str(sum_f1/k))