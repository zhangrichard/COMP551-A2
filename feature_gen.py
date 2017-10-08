import sys
import os
import operator
import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import pandas as pd
import math
from collections import Counter
import codecs


class FeatureVector(object):
    def __init__(self):
        #self.X =  np.zeros((numdata,self.vocabsize), dtype=np.int)
        # #self.Y =  np.zeros((numdata,), dtype=np.int)
        # #self.bow = {}
        self.y_index = 0
        self.t = 0


#making feature vectors
    def analyse_training_data(self, train_inputs, train_outputs):
        curpath = os.getcwd()
        print(curpath)
        par = Path(str(curpath)).parent
        print(par)
        new_path = os.path.join(par,'Data')

        train_input_data = pd.read_csv('Data/train_set_x.csv', sep=',', header= 1 , dtype = {'id':int ,'Text':str})
        train_output_data = pd.read_csv('Data/train_set_y.csv', sep=',', header=None)
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
        for index in train_input_data.itertuples():
            text = str(index.__getitem__(2))
            id  = int(index.__getitem__(1))
            lang_id = int(train_output_data.at[id,1])
            word_list = text.split(' ')
            if lang_id==0:
                sl_count = sl_count+1
                for word in word_list:
                    for char in word:
                        if char not in alphabets_freq_0:
                            alphabets_freq_0[char] = 1
                        else:
                            alphabets_freq_0[char] = alphabets_freq_0[char]+1
            elif lang_id==1:
                fr_count = fr_count+1
                for word in word_list:
                    for char in word:
                        if char not in alphabets_freq_1:
                            alphabets_freq_1[char] = 1
                        else:
                            alphabets_freq_1[char] = alphabets_freq_1[char]+1
            elif lang_id==2:
                sp_count = sp_count + 1
                for word in word_list:
                    for char in word:
                        if char not in alphabets_freq_2:
                            alphabets_freq_2[char] = 1
                        else:
                            alphabets_freq_2[char] = alphabets_freq_2[char]+1
            elif lang_id==3:
                gr_count = gr_count + 1
                for word in word_list:
                    for char in word:
                        if char not in alphabets_freq_3:
                            alphabets_freq_3[char] = 1
                        else:
                            alphabets_freq_3[char] = alphabets_freq_3[char]+1
            elif lang_id==4:
                pl_count = pl_count+1
                for word in word_list:
                    for char in word:
                        if char not in alphabets_freq_4:
                            alphabets_freq_4[char] = 1
                        else:
                            alphabets_freq_4[char] = alphabets_freq_4[char]+1
            else:
                garbage = garbage+1

        ########### PLOTTING THE HISTOGRAMS #####################
        ###0: Slovak, 1: French, 2: Spanish, 3: German, 4: Polish
        plt.bar(range(len(alphabets_freq_0)), alphabets_freq_0.values(), align='center' ,color = 'red' , label = 'Slovak')
        plt.xticks(range(len(alphabets_freq_0)), alphabets_freq_0.keys())
        plt.title(" for 0")
        #plt.show()
        #plt.title(" for 1")
        plt.bar(range(len(alphabets_freq_1)), alphabets_freq_1.values(), align='center' , color = 'blue' , label = 'french')
        plt.xticks(range(len(alphabets_freq_1)), alphabets_freq_1.keys())
        #plt.show()
        #plt.figure()
        plt.title("for 2")
        plt.bar(range(len(alphabets_freq_2)), alphabets_freq_2.values(), align='center' , color = 'black', label = 'spanish')
        plt.xticks(range(len(alphabets_freq_2)), alphabets_freq_2.keys())
        #plt.show()
        #plt.figure()
        plt.title("for 3")
        plt.bar(range(len(alphabets_freq_3)), alphabets_freq_3.values(), align='center' , color = 'green' , label = 'german')
        plt.xticks(range(len(alphabets_freq_3)), alphabets_freq_3.keys())
        #plt.show()
        #plt.figure()
        plt.title("for 4")
        plt.bar(range(len(alphabets_freq_4)), alphabets_freq_4.values(), align='center' ,  color = 'pink' , label = 'Polish')
        plt.xticks(range(len(alphabets_freq_4)), alphabets_freq_4.keys())
        plt.show()
        plt.legend()
        ############### PRINTING OTHER INFORMATION ###############################################
        f = open('out.txt','w')
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
        print("Total chars : ",total_chars)
        for k in alphabets_freq_0.keys():
            alphabets_freq_0[k] = alphabets_freq_0[k]/total_chars
        print("Fraction of presence of alphabets ::")
        print(sorted(alphabets_freq_0.items(), key=operator.itemgetter(1), reverse = True ))
        print("\n\n\n")


        print("******************FOR FRENCH*****************")
        print("Unique alphabets in French is ", len(alphabets_freq_1))
        print("Most used alphabet in French is ", max(alphabets_freq_1.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",fr_count)
        total_chars = sum(alphabets_freq_1.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_1.items(), key=operator.itemgetter(1) , reverse = True)
        print(sorted_dict)
        for k in alphabets_freq_1.keys():
            alphabets_freq_1[k] = alphabets_freq_1[k]/total_chars
        print("Fraction of presence of alphabets ::")
        print(sorted(alphabets_freq_1.items(), key=operator.itemgetter(1), reverse = True ))
        print("\n\n\n")



        print("*****************FOR SPANISH**************")
        print("Unique alphabets in Spanish is ", len(alphabets_freq_2))
        print("Most used alphabet in Spanish is ", max(alphabets_freq_2.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",sp_count)
        total_chars = sum(alphabets_freq_2.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_2.items(), key=operator.itemgetter(1) , reverse = True)
        print(sorted_dict)
        for k in alphabets_freq_2.keys():
            alphabets_freq_2[k] = alphabets_freq_2[k]/total_chars
        print("Fraction of presence of alphabets ::")
        print(sorted(alphabets_freq_2.items(), key=operator.itemgetter(1), reverse = True ))
        print("\n\n\n")



        print("***************FOR GERMAN***************/n")
        print("Unique alphabets in German is ", len(alphabets_freq_3))
        print("Most used alphabet in German is ", max(alphabets_freq_3.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",gr_count)
        total_chars = sum(alphabets_freq_3.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_3.items(), key=operator.itemgetter(1) , reverse = True)
        print(sorted_dict)
        for k in alphabets_freq_3.keys():
            alphabets_freq_3[k] = alphabets_freq_3[k]/total_chars
        print("Fraction of presence of alphabets ::")
        print(sorted(alphabets_freq_3.items(), key=operator.itemgetter(1), reverse = True ))
        print("\n\n\n")



        print("***************FOR POLISH***************/n")
        print("Unique alphabets in Polish is ", len(alphabets_freq_4))
        print("Most used alphabet in Polish is ", max(alphabets_freq_4.items(), key=operator.itemgetter(1))[0])
        print("No. of training samples : ",pl_count)
        total_chars = sum(alphabets_freq_4.values())
        print("Total chars : ",total_chars)
        sorted_dict = sorted(alphabets_freq_4.items(), key=operator.itemgetter(1) , reverse = True)
        print(sorted_dict)
        for k in alphabets_freq_4.keys():
            alphabets_freq_4[k] = alphabets_freq_4[k]/total_chars
        print("Fraction of presence of alphabets ::")
        print(sorted(alphabets_freq_4.items(), key=operator.itemgetter(1), reverse = True ))
        print("GARBAGE : ", garbage)

        #print(type(train_input_data[1:10]))


    def create_feature_vectors(self, train_inputs):




test = FeatureVector()

tr = 'train_set_x.csv'
tes = 'test_set_x.csv'
test.analyse_training_data(tr,tes)























