import pandas as pd
from keras.utils.np_utils import to_categorical
import numpy as np
import re, os, string,io
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
from emoji import UNICODE_EMOJI
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy import loadtxt
from underthesea import word_tokenize
from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils
from pyvi.ViTokenizer import tokenize
from gensim.models import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import KeyedVectors
from sklearn.ensemble import RandomForestClassifier
from keras.models import Model
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


# tính trung bình vector của câu

class MeanEmbeddingVectorizer(object):

    def __init__(self, word_model):
        self.word_model = word_model
        self.vector_size = word_model.wv.vector_size

    def predict(self, sent):
        return self.word_average(sent, 0)

    def fit(self):  # comply with scikit-learn transformer requirement
        return self

    def transform(self, docs):  # comply with scikit-learn transformer requirement
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector

    def word_average(self, sent,i):

        mean = []
        for word in sent:
            if word in self.word_model.wv.vocab:
                mean.append(self.word_model.wv.get_vector(word))

        if not mean:  # empty words
            # If a text is empty, return a vector of zeros.
            print("Từ xuất hiện 1 lần")
            print(sent,i)
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def word_average_list(self, docs):
        return np.vstack([self.word_average(docs[i],i) for i in range(len(docs))])

# xóa ký tự đặc biệt và chuyển thành chữ thường
def clean_text_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    listpunctuation1=listpunctuation +"0123456789"
    for i in listpunctuation1:
        text = text.replace(i, '')
    return text.lower()

# tách từ
def word_segment(sent):
    sent = word_tokenize(sent, format="text")
    sent = (word_tokenize(sent))
    # sent = ViTokenizer.tokenize(sent)
    return sent

# xóa stopword
# def load_stopwords():
#     # danh sách stopwords
#     data_stopwords = pd.read_csv('stopwords.txt', header=None)
#     list_stopwords = data_stopwords.iloc[:, 0]
#     arr_stopwords = []
#     for i in list_stopwords:
#         arr_stopwords.append(i)
#     return arr_stopwords
#
#
# def remove_stopword(text):  # xóa stopword và kí tự lẻ
#     arr = load_stopwords()
#     list_single = ['ă', 'â', 'ê', 'ô', 'ơ', 'ư', 'ua', 'uô', 'ia', 'yê', 'iê', 'ưa', 'ươ', 'a', 'b', 'c', 'd', 'e', 'f',
#                    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
#                    'ph', 'th', 'tr', 'ch', 'nh', 'ng', 'kh', 'gh', 'ngh']
#     pre_text = []
#     words = text.split()
#     for word in words:
#         if ((word not in arr) and (word not in list_single)):
#             pre_text.append(word)
#         text2 = ' '.join(pre_text)
#     return text2

# xử lý emoji, xóa emoji
def is_emoji(s):
    return s in UNICODE_EMOJI

def remove_emoji(text):
    pre_text = []
    words = text.split()
    for word in words:
        if (not (is_emoji(word))):
            pre_text.append(word)
        text2 = ' '.join(pre_text)
    return text2

def convert_emoji(tokens):
    arr = []
    for i in tokens:
        if (is_emoji(i)):
            emoji = (UNICODE_EMOJI[i])
            emoji = emoji.strip(string.punctuation).lower()
            arr.append(emoji)
        else:
            arr.append(i)
    return arr


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)


def SoSanh(pred):
    if(pred[0]==0):
        print("KQ là: Surprise")
    elif(pred[0]==1):
        print("KQ là: Anger")
    elif (pred[0] == 2):
        print("KQ là: Disgust")
    elif (pred[0] == 3):
        print("KQ là: Enjoyment")
    elif (pred[0] == 4):
        print("KQ là: Fear")
    elif (pred[0] == 5):
        print("KQ là: Other")
    elif (pred[0] == 6):
        print("KQ là: Sadness")

# load dữ liệu
cols = ['Emotion', 'Sentence']

data_train = pd.ExcelFile('train_nor_811.xlsx')
df_train = pd.read_excel(data_train, 'Sheet1')

data_test = pd.ExcelFile('test_nor_811.xlsx')
df_test = pd.read_excel(data_test, 'Sheet1')

data_dev = pd.ExcelFile('valid_nor_811.xlsx')
df_dev = pd.read_excel(data_dev, 'Sheet1')

df_train["Emotion"] = df_train["Emotion"].replace("Anger", 1)
df_train["Emotion"] = df_train["Emotion"].replace("Disgust", 2)
df_train["Emotion"] = df_train["Emotion"].replace("Enjoyment", 3)
df_train["Emotion"] = df_train["Emotion"].replace("Fear", 4)
df_train["Emotion"] = df_train["Emotion"].replace("Other", 5)
df_train["Emotion"] = df_train["Emotion"].replace("Sadness", 6)
df_train["Emotion"] = df_train["Emotion"].replace("Surprise", 0)

df_test["Emotion"] = df_test["Emotion"].replace("Anger", 1)
df_test["Emotion"] = df_test["Emotion"].replace("Disgust", 2)
df_test["Emotion"] = df_test["Emotion"].replace("Enjoyment", 3)
df_test["Emotion"] = df_test["Emotion"].replace("Fear", 4)
df_test["Emotion"] = df_test["Emotion"].replace("Other", 5)
df_test["Emotion"] = df_test["Emotion"].replace("Sadness", 6)
df_test["Emotion"] = df_test["Emotion"].replace("Surprise", 0)

df_dev["Emotion"] = df_dev["Emotion"].replace("Anger", 1)
df_dev["Emotion"] = df_dev["Emotion"].replace("Disgust", 2)
df_dev["Emotion"] = df_dev["Emotion"].replace("Enjoyment", 3)
df_dev["Emotion"] = df_dev["Emotion"].replace("Fear", 4)
df_dev["Emotion"] = df_dev["Emotion"].replace("Other", 5)
df_dev["Emotion"] = df_dev["Emotion"].replace("Sadness", 6)
df_dev["Emotion"] = df_dev["Emotion"].replace("Surprise", 0)


labels_train = to_categorical(df_train.Emotion, num_classes=7)

labels_test = to_categorical(df_test.Emotion, num_classes=7)

labels_dev = to_categorical(df_dev.Emotion, num_classes=7)

tokenize_df = []

# xử lý tập train
def clean_train():
    arr = []
    for i in range(len(df_train)):
        Senten = (df_train['Sentence'][i])
        Senten = clean_text_text(Senten)
        Senten = deEmojify(Senten)
        Senten = word_segment(Senten)
        tokenize_df.append(Senten)
        arr.append(Senten)
    return arr


clean_train = clean_train()

# xử lý tập test
def clean_text_1():
    arr = []

    for i in range(len(df_test)):
        Senten = (df_test['Sentence'][i])
        Senten = clean_text_text(Senten)
        Senten = deEmojify(Senten)
        Senten = word_segment(Senten)
        tokenize_df.append(Senten)
        arr.append(Senten)
    return arr


clean_test = clean_text_1()

# xử lý tập dev
def clean_dev():
    arr = []
    for i in range(len(df_dev)):
        Senten = (df_dev['Sentence'][i])
        Senten = clean_text_text(Senten)
        Senten = deEmojify(Senten)
        Senten = word_segment(Senten)
        tokenize_df.append(Senten)
        arr.append(Senten)
    return arr

clean_dev = clean_dev()

# xử lý dữ liệu nhập vô để test model
def clean_clean(text):
    Senten = clean_text_text(text)
    Senten = deEmojify(Senten)
    Senten = word_segment(Senten)
    return text


#--------------- word2vec-------------------

path_embedding= 'baomoi.vn.model.bin'
from gensim.models import KeyedVectors
word_model = KeyedVectors.load_word2vec_format(path_embedding, binary=True)
mean_vec_tr = MeanEmbeddingVectorizer(word_model)


# word_model= Word2Vec(tokenize_df, min_count=2, size=200, window=5, workers=4, iter=100)
# word_model_1.save('w2v_model')
# word_vectors = word_model.wv
#get keyed Vectors
# word_model= KeyedVectors.load('w2v_model')


# input Sentence: train, test, valid
doc_vec_train = mean_vec_tr.transform(clean_train)
doc_vec_test = mean_vec_tr.transform(clean_test)
doc_vec_dev = mean_vec_tr.transform(clean_dev)

# input Emotion: train, test, valid
# labels_train,labels_test,labels_dev

doc_train=df_train['Emotion']
doc_test=df_test['Emotion']
doc_dev=df_dev['Emotion']

#--------------------------------------------RD---------------------------------------------------------
# clf = RandomForestClassifier(n_estimators=10, random_state=30)
# clf_rd=clf.fit(doc_vec_train, doc_train)
# y_pred = clf_rd.predict(doc_vec_test)
# score = accuracy_score(doc_test,y_pred)
# f1score = f1_score(doc_test, y_pred,average='weighted')
#
# print("Accuracy là: ",score)
# print("Y_PRED là: ",y_pred)
# print(confusion_matrix(doc_test, y_pred))
# print(classification_report(doc_test, y_pred))
#
# print("-----------------------------------TEST-MODEL------------------------")
# A="tao muốn khóc quá rồi"
# A=clean_clean(A)
# B=mean_vec_tr.predict(A)
# #
# pred=clf_rd.predict(np.array([B]))
#
# SoSanh(pred)





#----------------------------------SVM------------------------------------------------------------------------------------------



# parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10, 100, 1000],
#                'C': [0.01, 0.1, 1, 10, 100, 1000]}]
# svm = SVC()
# clf = GridSearchCV(svm,parameters,cv=10)
# clf.fit(doc_vec_train, doc_train)


clf=SVC(kernel='rbf',gamma='scale')
clf_svm=clf.fit(doc_vec_train, doc_train)
y_pred = clf_svm.predict(doc_vec_test)
print("Y_PRED là: ",y_pred)
accuracy = accuracy_score(doc_test, y_pred)
f1score = f1_score(doc_test, y_pred, average='micro')

print("Accuracy là: ",accuracy)

print(confusion_matrix(doc_test, y_pred))
print(classification_report(doc_test,y_pred))

print("-----------------------------------TEST-MODEL------------------------")
A="tao muốn khóc quá rồi"
A=clean_clean(A)
B=mean_vec_tr.predict(A)
pred=clf_svm.predict(np.array([B]))
SoSanh(pred)









