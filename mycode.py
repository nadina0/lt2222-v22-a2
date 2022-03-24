import gzip
import random
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import ngrams
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def sample_lines(file, lines):
    zipped_file = gzip.open(file, 'rb')
    all_lines = zipped_file.readlines()
    random_lines = random.choices(all_lines, k=lines)
    random_lines_list = []
    for line in random_lines:
        random_line = line.decode('utf8').strip()
        random_lines_list.append(random_line)
    return random_lines_list

def process_sentences(file):
    tokenizer = WordPunctTokenizer()
    tokenized_list = []
    for sent in file:
        tokenized_lines = tokenizer.tokenize(sent)
        tokenized_list.append(tokenized_lines)
    
    tagged_list = []
    for sent in tokenized_list:
        tagged_sent = nltk.pos_tag(sent)
        tagged_list.append(tagged_sent)

    stop_words = set(stopwords.words('english'))
    
    final_list = []

    for sent in tagged_list:
        temp_list = []
        for word, tag in sent:
            word = word.lower()
            if word in stop_words:
                continue
            elif word.isalpha() == False:
                continue
            elif word == ".":
                continue
            elif word == ",":
                continue
            elif len(word) < 2:
                continue
            else:
                pair = (word, tag)
                temp_list.append(pair)
        final_list.append(temp_list)
    
    return final_list

def create_samples(sentences, samples):
    fivegrams = []
    for sent in sentences:
        five_grams = list(ngrams(sent, 5))
        for item in five_grams:
            fivegrams.append(item)

    random_fivegrams = random.choices(fivegrams, k=samples)

    final_list = []
    for pair1, pair2, pair3, pair4, pair5 in random_fivegrams:
        sample1 = pair1[0][-2:] + '_1'
        sample2 = pair2[0][-2:] + '_2'
        sample4 = pair4[0][-2:] + '_4'
        sample5 = pair5[0][-2:] + '_5'
        verb = pair3[1]
        
        if "V" in verb:
            i = 1
        else:
            i = 0
        
        all_samples = (sample1, sample2, sample4, sample5), i
        
        final_list.append(all_samples)
    return final_list

def create_df(all_samples):

    all_columns = []
    for tuple, verb in all_samples:
        for ending in tuple:
            if ending not in all_columns:
                all_columns.append(ending)
    all_columns.append("V")


    counter = []
    for tuples, verbs in all_samples:
        count = []
        for column in all_columns:
            if column != "V":
                if column in tuples:
                    count.append(1)
                else:
                    count.append(0)
        count.append(verbs)
        counter.append(count)
    
    df = pd.DataFrame(counter, columns=all_columns)
    return df

def split_samples(fulldf, test_percent):
    
    X = fulldf.iloc[:, 0:-1]
    y = fulldf.iloc[:, -1]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_percent/100)
    return train_X, train_y, test_X, test_y

def train(train_X, train_y, kernel):
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_X, train_y)
    return clf

def eval_model(model, test_X, test_y):
    y_pred = model.predict(test_X)
    prec_score = precision_score(test_y, y_pred)
    rec_score = recall_score(test_y, y_pred)
    F1_score = f1_score(test_y, y_pred)

    print(f"The precision score is:{prec_score}")
    print(f"The recall score is: {rec_score}")
    print(f"The f1 score is: {F1_score}")


if __name__ == "__main__" :
    sampled_lines = sample_lines("UN-english.txt.gz", lines=100000)
    processed_sentences = process_sentences(sampled_lines)
    all_samples = create_samples(processed_sentences, samples=50000)

    fulldf = create_df(all_samples)


    train_X, train_y, test_X, test_y = split_samples(fulldf, test_percent=20)
    print(len(train_X), len(train_y), len(test_X), len(test_y))
    model_linear = train(train_X, train_y, kernel='linear')
    model_rbf = train(train_X, train_y, kernel="rbf")
    print(model_linear, model_rbf)
    eval_model(model_linear, test_X, test_y)
    eval_model(model_rbf, test_X, test_y)
