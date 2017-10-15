# -*- coding: utf-8 -*-
import sys
import os
import csv
import re
import string
import operator
import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import pandas as pd
import math
import _pickle as pickle
import json
from collections import Counter
import codecs
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

class FeatureVector(object):
    def __init__(self):
        #self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
        # #self.Y =  np.zeros((numdata,), dtype=np.int)
        # #self.bow = {}
        self.y_index = 0
        self.t = 0
        self.alphabet_reference = {}


#making feature vectors
    def analyse_training_data(self):
        df = pd.read_csv('../../Data/train_set_x.csv',sep=',', header= 0 , dtype = {'id':int ,'Text':str} )
        train_output_data = pd.read_csv('../../Data/train_set_y.csv', sep=',', header=0)
        #df['Text'] = df['Text'].str.replace(u'![\u4e00-\u9fff，。／【】、v；‘:\"\",./[]-={}]+' , '')
        #df['Text'] = df['Text'].str.replace(r'r')
        #RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
        #df['Text']= df['Text'].str.replace(RE_PUNCTUATION, "")
        #df['Text']= df['Text'].str.replace('[+string.punctuation+]','')
        pattern = re.compile('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        #df['Text'] = df['Text'].str.replace(pattern, '')
        patternURL = re.compile('url[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        #patternJPG = re.compile('(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))?jpg')
        #df['Text'] = df['Text'].str.replace(patternJPG ,'')
        #df['Text'] = df['Text'].str.replace('[0-9]', '')
        #df['Text'] = df['Text'].str.replace('[0-9]', '')
        #df['Text'] = df['Text'].str.replace(r'(.)\1+', r'\1\1')
        alphabets_freq_main = {}
        alphabets_freq_0 = {}
        sl_count = 0
        fr_count = 0
        sp_count = 0
        gr_count = 0
        pl_count = 0
        garbage = 0
        alphabets_freq_1 = {}
        alphabets_freq_2 = {}
        alphabets_freq_3 = {}
        alphabets_freq_4 = {}
        for index in df.itertuples():
            text = str(index.__getitem__(2))
            #print(text)
            text_series = pd.Series(text)
            text_series = text_series.str.replace(pattern,'')
            text_series = text_series.str.replace(patternURL,'')
            #print('a')
            #text_series = text_series.str.replace(patternJPG,'')
            #print('b')
            text_series = text_series.str.replace('[0-9]', '')
            #print('c')
            text_series = text_series.str.replace(r'(.)\1+', r'\1\1')
            text = str(text_series.values[0])
            ##print(text)
            id  = int(index.__getitem__(1))
            #print(id)
            lang_id = int(train_output_data.iloc[id,1])
            word_list = text.split(' ')
            if lang_id==0:
                sl_count = sl_count+1
                for word in word_list:
                    for char in word:
                        if (ord(char)>=97 and ord(char)<=122) or (ord(char)>=192 and ord(char)<=286) or (ord(char)>=65 and ord(char)<=90):
                            if (char not in alphabets_freq_0):
                                alphabets_freq_0[char] = 1
                            else:
                                alphabets_freq_0[char] = alphabets_freq_0[char]+1
                            if (char not in alphabets_freq_main):
                                alphabets_freq_main[char] = 0
            elif lang_id==1:
                fr_count = fr_count+1
                for word in word_list:
                    for char in word:
                        if (ord(char) >= 97 and ord(char) <= 122) or (ord(char) >= 192 and ord(char) <= 286) or (ord(char) >= 65 and ord(char) <= 90):
                            if char not in alphabets_freq_1:
                                alphabets_freq_1[char] = 1
                            else:
                                alphabets_freq_1[char] = alphabets_freq_1[char]+1
                            if (char not in alphabets_freq_main):
                                alphabets_freq_main[char] = 0
            elif lang_id==2:
                sp_count = sp_count + 1
                for word in word_list:
                    for char in word:
                        if (ord(char) >= 97 and ord(char) <= 122) or (ord(char) >= 192 and ord(char) <= 286) or (ord(char) >= 65 and ord(char) <= 90):
                            if char not in alphabets_freq_2:
                                alphabets_freq_2[char] = 1
                            else:
                                alphabets_freq_2[char] = alphabets_freq_2[char]+1
                            if (char not in alphabets_freq_main):
                                alphabets_freq_main[char] = 0
            elif lang_id==3:
                gr_count = gr_count + 1
                for word in word_list:
                    for char in word:
                        if (ord(char) >= 97 and ord(char) <= 122) or (ord(char) >= 192 and ord(char) <= 286) or (ord(char) >= 65 and ord(char) <= 90):
                            if char not in alphabets_freq_3:
                                alphabets_freq_3[char] = 1
                            else:
                                alphabets_freq_3[char] = alphabets_freq_3[char]+1
                            if (char not in alphabets_freq_main):
                                alphabets_freq_main[char] = 0
            elif lang_id==4:
                pl_count = pl_count+1
                for word in word_list:
                    for char in word:
                        if (ord(char) >= 97 and ord(char) <= 122) or (ord(char) >= 192 and ord(char) <= 286) or (ord(char) >= 65 and ord(char) <= 90):
                            if char not in alphabets_freq_4:
                                alphabets_freq_4[char] = 1
                            else:
                                alphabets_freq_4[char] = alphabets_freq_4[char]+1
                            if (char not in alphabets_freq_main):
                                alphabets_freq_main[char] = 0
            else:
                garbage = garbage+1
        for k in alphabets_freq_main.keys():
            if k not in alphabets_freq_0.keys():
                #print(type(k))
                alphabets_freq_0[k] = 0
            if k not in alphabets_freq_1.keys():
                alphabets_freq_1[k] = 0
            if k not in alphabets_freq_2.keys():
                alphabets_freq_2[k] = 0
            if k not in alphabets_freq_3.keys():
                alphabets_freq_3[k] = 0
            if k not in alphabets_freq_4.keys():
                alphabets_freq_4[k] = 0
        arranged_0 = collections.OrderedDict(sorted(alphabets_freq_0.items()))
        arranged_1 = collections.OrderedDict(sorted(alphabets_freq_1.items()))
        arranged_2 = collections.OrderedDict(sorted(alphabets_freq_2.items()))
        arranged_3 = collections.OrderedDict(sorted(alphabets_freq_3.items()))
        arranged_4 = collections.OrderedDict(sorted(alphabets_freq_4.items()))
        print(len(arranged_0))
                    ########### PLOTTING THE HISTOGRAMS #####################
        ###0: Slovak, 1: French, 2: Spanish, 3: German, 4: Polish
        plt.bar(range(len(alphabets_freq_main)), arranged_0.values(), align='center' ,color = 'red' , label = 'Slovak')
        plt.xticks(range(len(alphabets_freq_0)), arranged_0.keys())
        plt.title(" for 0")
        plt.show()
        plt.title(" for 1")
        plt.bar(range(len(alphabets_freq_main)), arranged_1.values(), align='center' , color = 'blue' , label = 'french')
        plt.xticks(range(len(alphabets_freq_1)), arranged_1.keys())
        plt.show()
        plt.figure()
        plt.title("for 2")
        plt.bar(range(len(alphabets_freq_main)), arranged_2.values(), align='center' , color = 'black', label = 'spanish')
        plt.xticks(range(len(alphabets_freq_2)), arranged_2.keys())
        plt.show()
        plt.figure()
        plt.title("for 3")
        plt.bar(range(len(alphabets_freq_main)), arranged_3.values(), align='center' , color = 'green' , label = 'german')
        plt.xticks(range(len(alphabets_freq_3)),  arranged_3.keys())
        plt.show()
        plt.figure()
        plt.title("for 4")
        plt.bar(range(len(alphabets_freq_main)), arranged_4.values(), align='center' ,  color = 'pink' , label = 'Polish')
        plt.xticks(range(len(alphabets_freq_4)),  arranged_4.keys())
        plt.show()
        plt.legend()
        ############### PRINTING OTHER INFORMATION ###############################################
        f = open('out.csv','w')
        sys.stdout = f
        print("Stats - 0: Slovak, 1: French, 2: Spanish, 3: German, 4: Polish ")
        print("\n\n\n")
        print("*********************FOR SLOVAK*****************")
        print("Unique alphabets in Slovak is ", len(alphabets_freq_0))
        print("Most used alphabet in Slovak is ", max(alphabets_freq_0.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",sl_count)
        total_chars = sum(alphabets_freq_0.values())
        print(total_chars)
        sorted_dict = sorted(alphabets_freq_0.items(), key=operator.itemgetter(1) , reverse = True)
        print("Charecter occourances :: ")
        f.write(json.dumps(sorted_dict))
        print("Total chars : ",total_chars)
        for k in alphabets_freq_0.keys():
            alphabets_freq_0[k] = alphabets_freq_0[k]/total_chars
        print("Fraction of presence of alphabets ::")
        v = sorted(alphabets_freq_0.items(), key=operator.itemgetter(1), reverse = True )
        f.write(json.dumps(v))
        print("\n\n\n")


        print("******************FOR FRENCH*****************")
        print("Unique alphabets in French is ", len(alphabets_freq_1))
        print("Most used alphabet in French is ", max(alphabets_freq_1.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",fr_count)
        total_chars = sum(alphabets_freq_1.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_1.items(), key=operator.itemgetter(1) , reverse = True)
        print("Charecter occurrences :: ")
        f.write(json.dumps(sorted_dict))
        for k in alphabets_freq_1.keys():
            alphabets_freq_1[k] = alphabets_freq_1[k]/total_chars
        print("Fraction of presence of alphabets ::")
        v = sorted(alphabets_freq_1.items(), key=operator.itemgetter(1), reverse = True )
        f.write(json.dumps(v))
        print("\n\n\n")



        print("*****************FOR SPANISH**************")
        print("Unique alphabets in Spanish is ", len(alphabets_freq_2))
        print("Most used alphabet in Spanish is ", max(alphabets_freq_2.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",sp_count)
        total_chars = sum(alphabets_freq_2.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_2.items(), key=operator.itemgetter(1) , reverse = True)
        print("Charecter occurrences :: ")
        f.write(json.dumps(sorted_dict))
        for k in alphabets_freq_2.keys():
            alphabets_freq_2[k] = alphabets_freq_2[k]/total_chars
        print("Fraction of presence of alphabets ::")
        v = sorted(alphabets_freq_2.items(), key=operator.itemgetter(1), reverse = True )
        f.write(json.dumps(v))
        print("\n\n\n")



        print("***************FOR GERMAN***************/n")
        print("Unique alphabets in German is ", len(alphabets_freq_3))
        print("Most used alphabet in German is ", max(alphabets_freq_3.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",gr_count)
        total_chars = sum(alphabets_freq_3.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_3.items(), key=operator.itemgetter(1) , reverse = True)
        print("Charecter occurrences :: ")
        f.write(json.dumps(sorted_dict))
        for k in alphabets_freq_3.keys():
            alphabets_freq_3[k] = alphabets_freq_3[k]/total_chars
        print("Fraction of presence of alphabets ::")
        v = sorted(alphabets_freq_3.items(), key=operator.itemgetter(1), reverse = True )
        f.write(json.dumps(v))
        print("\n\n\n")



        print("***************FOR POLISH***************/n")
        print("Unique alphabets in Polish is ", len(alphabets_freq_4))
        print("Most used alphabet in Polish is ", max(alphabets_freq_4.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",pl_count)
        total_chars = sum(alphabets_freq_4.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_4.items(), key=operator.itemgetter(1) , reverse = True)
        print("Charecter occurrences :: ")
        f.write(json.dumps(sorted_dict))
        for k in alphabets_freq_4.keys():
            alphabets_freq_4[k] = alphabets_freq_4[k]/total_chars
        print("Fraction of presence of alphabets ::")
        v = sorted(alphabets_freq_4.items(), key=operator.itemgetter(1), reverse = True )
        f.write(json.dumps(v))
        print("GARBAGE : ", garbage)
        f.close()
        #print(type(train_input_data[1:10]))


    def preprocess_data_create_feature_train(self):
        df = pd.read_csv('../../Data/train_set_x.csv', sep=',', header= 0 , dtype = {'id':int ,'Text':str})

        df['Text'] = df['Text'].str.replace(u'![\u4e00-\u9fff，。／【】、v；‘:\"\",./[]-={}]+' , '')
        #df['Text'] = df['Text'].str.replace(r'r')
        RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
        df['Text']= df['Text'].str.replace(RE_PUNCTUATION, "")
        #df['Text']= df['Text'].str.replace('[+string.punctuation+]','')
        pattern = re.compile('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df['Text'] = df['Text'].str.replace(pattern, '')
        patternURL = re.compile('url[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df['Text'] = df['Text'].str.replace(patternURL, '')
        #patternJPG = re.compile('(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+jpg')
        #df['Text'] = df['Text'].str.replace(patternJPG ,'')
        df['Text'] = df['Text'].str.replace('[0-9]', '')
        df['Text'] = df['Text'].str.replace(r'(.)\1+', r'\1\1')
            #df['Text'] = df['Text'].str.encode("latin-1","ignore")

        #print(df['Text'])
        vectorizer = TfidfVectorizer(analyzer='char',lowercase=False)
        x = vectorizer.fit_transform(df['Text'].values.astype('U')).toarray()
        #print(x)
        for i,col in enumerate(vectorizer.get_feature_names()):
            if (ord(col)>=97 and ord(col)<=122) or (ord(col)>=192 and ord(col)<=286) or (ord(col)>=65 and ord(col)<=90):
                #print(col)
                df[col] = x[:, i]
                self.alphabet_reference[col] = 0
        features = df.drop('Id',axis=1)
        features = features.drop('Text',axis = 1)
        #print(self.alphabet_reference)
        #print(features)
        features.to_csv('train_featuresV2_mini.csv', sep=',', encoding='utf-8')


    def test_feature_vectorsV2(self):
        df = pd.read_csv('../../Data/test_set_x.csv', sep=',', header=0)
        df['Text'] = df['Text'].str.replace(u'![\u4e00-\u9fff，。／【】、v；‘:\"\",./[]-={}]+', '')
        # df['Text'] = df['Text'].str.replace(r'r')
        RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
        df['Text'] = df['Text'].str.replace(RE_PUNCTUATION, "")
        # df['Text']= df['Text'].str.replace('[+string.punctuation+]','')
        pattern = re.compile('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df['Text'] = df['Text'].str.replace(pattern, '')
        patternURL = re.compile('url[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        df['Text'] = df['Text'].str.replace(patternURL, '')
        #patternJPG = re.compile('(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+jpg')
        #df['Text'] = df['Text'].str.replace(patternJPG ,'')
        df['Text'] = df['Text'].str.replace('[0-9]', '')
        df['Text'] = df['Text'].str.replace(r'(.)\1+', r'\1\1')

        vectorizer = TfidfVectorizer(analyzer='char' , lowercase=False)
        x = vectorizer.fit_transform(df['Text'].values.astype('U')).toarray()
        test_feat_dict = copy.deepcopy(self.alphabet_reference)
        #print(test_feat_dict)
        #header = [key.encode('utf8').strip() for key in self.alphabet_reference.keys()]
        #test_feat_list.append(header)
        for key in test_feat_dict.keys():
            v = np.array([0] * x.shape[0])
            test_feat_dict[key] = v
        #print(test_feat_dict)
        for i, col in enumerate(vectorizer.get_feature_names()):
            if col in self.alphabet_reference:
                test_feat_dict[col] = x[:,i]
                #print(x.shape[0])
            #v = np.asarray([0]* x.shape[0])
            #test_feat_dict[col] = v
            #print(v)
        #print(test_feat_dict)
        testdf = pd.DataFrame(test_feat_dict)
        #testdf = testdf.drop(testdf.columns[0],axis = 1)
        testdf.to_csv('test_featuresV2_mini.csv', sep=',', encoding='utf-8')





    def create_feature_vectors(self, train_inputs):
        train_input_data = pd.read_csv('../../Data/train_set_x.csv', sep=',', header= 0 , dtype = {'id':int ,'Text':str})
        #train_output_data = pd.read_csv('../../Data/train_set_y.csv', sep=',', header=None)

        unique_alphabets = {}
        feat_dict_list = []
        i = 0
        for index in train_input_data.itertuples():
            text = str(index.__getitem__(2))
            id  = int(index.__getitem__(1))
            word_list = text.split(' ')
            for word in word_list:
                for char in word:
                    if char not in unique_alphabets:
                        unique_alphabets[char] = 0
        #print(unique_alphabets.keys())
        header = [ key.encode('utf8').strip() for key in unique_alphabets.keys()]
        feat_dict_list.append(header)
        i = 0
        with open( 'features.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for index in train_input_data.itertuples():
                i = i+1
                feat_dict = copy.deepcopy(unique_alphabets)
                text = str(index.__getitem__(2))
                id = int(index.__getitem__(1))
                #print(text)
                word_list = text.split(' ')
                for word in word_list:
                  for char in word:
                     feat_dict[char] = feat_dict[char]+1
                ttl_char_inrow = sum(feat_dict.values())
                row_list = []
                row_list = [ val/ttl_char_inrow for val in feat_dict.values()]
                #print(i)
                #print(row_list)
                feat_dict_list.append(row_list)
            for l in feat_dict_list:
                writer.writerow(l)

        ##print(feat_dict_list[6:8])

    def test_feature_vectors(self):
        train_input_data = pd.read_csv('../../Data/train_set_x.csv', sep=',', header=0, dtype={'id': int, 'Text': str})
        test_input_data = pd.read_csv('../../Data/test_set_x.csv', sep=',', header=0)
        unique_alphabets = {}
        feat_dict_list = []
        i = 0
        for index in train_input_data.itertuples():
            text = str(index.__getitem__(2))
            id = int(index.__getitem__(1))
            word_list = text.split(' ')
            for word in word_list:
                for char in word:
                    if char not in unique_alphabets:
                        unique_alphabets[char] = 0
        # print(unique_alphabets.keys())
        header = [key.encode('utf8').strip() for key in unique_alphabets.keys()]
        feat_dict_list.append(header)
        i = 0
        with open('test_features.csv', "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for index in test_input_data.itertuples():
                i = i + 1
                feat_dict = copy.deepcopy(unique_alphabets)
                text = str(index.__getitem__(2))
                #id = int(index.__getitem__(1))
                # print(text)
                word_list = text.split(' ')
                try:
                    for word in word_list:
                        for char in word:
                            feat_dict[char] = feat_dict[char] + 1
                    ttl_char_inrow = sum(feat_dict.values())
                    row_list = []
                    row_list = [val / ttl_char_inrow for val in feat_dict.values()]
                    # print(i)
                    # print(row_list)
                except Exception as ee:
                    print(str(ee))
                feat_dict_list.append(row_list)
            for l in feat_dict_list:
                writer.writerow(l)




test = FeatureVector()

tr = 'train_set_x.csv'
tes = 'test_set_x.csv'
#test.analyse_training_data()
test.preprocess_data_create_feature_train()
test.test_feature_vectorsV2()























