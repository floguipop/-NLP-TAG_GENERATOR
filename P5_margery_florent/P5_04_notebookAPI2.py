#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns


# In[2]:


data = pd.read_csv('QueryResults_2.csv')


# # Data preparation for text analysis

# In[3]:


data = pd.read_csv('QueryResults_2.csv')


# ### Fonctions déstinée à la création du modèle Word2vec. 

# In[4]:


def sentence_to_wordlist(sentence, remove_stopwords=False, remove_ponctuation=False,lemmatize = False):

    # 1. Remove HTML
    sentence = BeautifulSoup(sentence).get_text()
    sentence = re.sub("[^a-zA-Z+#]"," ", sentence)
    # 1. Tokenization des phrases en liste de mots. 
    words = sentence.split()    
    # 2. Lemmatisation des listes de mots(en option)
    if lemmatize:
        words = lemma_fct(words)
    
    # 3. Gestion des stopWords (en option)
    if remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]
        
    # 4. Gestion de la ponctuation (en option)
    if remove_ponctuation:
        ponct = ['[', ']', ',', '.', ':', '?', '(', ')','\'','\"','`']
        words = [w for w in words if not w in ponct]
    
    # 5. Return a list of words
    return(words)


# In[5]:


def question_to_sentences(question, tokenizer):
    # Function to split a question into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    
    question = BeautifulSoup(question).get_text()
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(question.strip())
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(sentence_to_wordlist(raw_sentence, remove_stopwords=False,                                                  remove_ponctuation=False, lemmatize=False))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


# In[6]:


#tokenizer qui décompose en phrases via la ponctuation= punkt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# ## Gestion des TAGS ( cible ) 

# In[7]:


def tag_remover(tag):
    tag = tag.replace('>' , " ").replace('<', " ")
    return tag

def tag_to_word_list(tag):
    tag = BeautifulSoup(tag).get_text()
    tag = tag.split()
    return(tag)


# In[8]:


Tags = []
for tag in data.Tags:
    tag = tag_remover(tag)
    tag = tag_to_word_list(tag)
    Tags.append(tag)
    
tag_bank = [] 
for tag in Tags : 
    for y in tag:
        tag_bank.append(y)


# In[9]:


tags = pd.DataFrame()
tags['Tags'] = tag_bank
sns.barplot(tags['Tags'].value_counts().head(10).index,            tags['Tags'].value_counts().head(10))
top_ten = tags['Tags'].value_counts().head(10).index
top_ten_tags=[]
for i in range(len(top_ten)):
    top_ten_tags.append(top_ten[i])


# In[10]:


tags['Tags'].value_counts().head(10).sum()


# In[11]:


#Création des data que nous utiliserons pour créer notre modèle supervisé
data_to_train = [] #--> X
target_to_train = [] #--> y
counter = 0
for tag in Tags:
    for i in tag:
        if i in top_ten_tags:            
            data_to_train.append(data.Body[counter])
            target_to_train.append(i)
            
            
    counter+=1


# In[12]:


for tag in top_ten:
    print(tag)
    top_ten_tags.append(tag)


# # WORD2VEC Model

# In[13]:


sentences = []
for question in data['Body']:
    sentences += (question_to_sentences(question,tokenizer))


# In[14]:


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 10   # Minimum word count                        
num_workers = 5      # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,             vector_size=num_features, min_count = min_word_count,             window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)


# ## Embedding du modèle Word2Vec

# In[15]:


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index_to_key)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model.wv[word])
    # 
    
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
        if counter%1000. == 0.:
            print ("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       #
       # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


# In[16]:


# ****************************************************************
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.

clean_train_questions = []
# Ici, data_to_train a été défini dans la partie "Gestion des Tags". Cela correspond aux données avec 
# au moins 1 des 10 tags les plus récurents. C'est sur ces données que l'on va entrainer notre modèle supervisé. 
for question in data_to_train:
    clean_train_questions.append(sentence_to_wordlist( question,         remove_stopwords=True ))
    
print ("Creating average feature vecs for test questions")
trainDataVecs = getAvgFeatureVecs(clean_train_questions, model, num_features)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
target_to_train_encoded = label_encoder.fit_transform(target_to_train)

X_train, X_test, y_train, y_test = train_test_split(trainDataVecs,target_to_train_encoded,test_size=0.3)


# In[114]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

model_to_set = OneVsRestClassifier(SVC())

parameters = {
    "estimator__C": [1],
    "estimator__kernel":["poly","rbf"],
    "estimator__degree":[1, 2],
}

model_tunning = GridSearchCV(model_to_set, param_grid=parameters,cv= 5,verbose = 4, scoring = 'f1_micro',n_jobs=-1)
model_tunning.fit(X_train[:10000], y_train[:10000])

print (model_tunning.best_score_)
print (model_tunning.best_params_)


# In[115]:


model_tunning.score(X_test[:1000],y_test[:1000])


# In[69]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#Avec Probas :
clf = OneVsRestClassifier(SVC(probability=False))
clf.fit(X_train[:1000], y_train[:1000])

#Sans Probas : 
#clf = OneVsRestClassifier(SVC(probability=False)).fit(X_train, y_train)


# In[88]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#Avec Probas :
clf = OneVsRestClassifier(SVC(probability=False))
clf.fit(X_train[:10000], y_train[:10000])

#Sans Probas : 
#clf = OneVsRestClassifier(SVC(probability=False)).fit(X_train, y_train)


# In[93]:


clf.get_params()


# In[81]:


clf.score(X_test[:5000],y_test[:5000])


# In[82]:


clf.score(X_train,y_train)


# ### Pipeline

# ### Créons une pipeline afin de faciliter la prédiction de notre notre modèle. 

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
target_to_train_encoded = label_encoder.fit_transform(target_to_train)

X_train, X_test, y_train, y_test = train_test_split(data_to_train,target_to_train_encoded,test_size=0.3)


# In[29]:


len(target_to_train_encoded)


# In[30]:


def cleaning_question_W2V(data_to_train):
    
    clean_train_questions = []
    # Ici, data_to_train a été défini dans la partie "Gestion des Tags". Cela correspond aux données avec 
    # au moins 1 des 10 tags les plus récurents. C'est sur ces données que l'on va entrainer notre modèle supervisé. 
    for question in data_to_train:
        clean_train_questions.append(sentence_to_wordlist( question,             remove_stopwords=True ))
    
    print ("Creating average feature vecs for test questions")
    trainDataVecs = getAvgFeatureVecs(clean_train_questions, model, num_features)
    return trainDataVecs

# Création d'un transformeur, permettant d'intégrer notre fonction dans la pipeline. 

from sklearn.preprocessing import FunctionTransformer
transformer_2 = FunctionTransformer(cleaning_question_W2V)


# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# ### Création de la pipeline 

# In[ ]:





# In[32]:


pipeline = Pipeline([
   ('transformer', transformer_2), ('clf',OneVsRestClassifier(SVC(probability=True))), 
])


# In[ ]:


pipeline.fit(X_train, y_train)


# In[ ]:




import joblib
joblib.dump(pipeline, 'pipeline_W2V.joblib')
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train,y_train)
import mlflow
mlflow.sklearn.save_model(pipeline, 'mlflow_model_W2V', signature=signature)
# # API

# In[ ]:


## from fast_dash import FastDash
from fast_dash.Components import Text

# Step 1: Define your model inference
def text_to_text_function(question):
    result = []
    result_2  = []
    probas = []
    #result = pipeline.predict([question])
    #result = label_encoder.inverse_transform(result)[0]
    result_proba = pipeline.predict_proba([question])
    print(result_proba)
    for idx, prob in enumerate(result_proba[0]) : 
        if prob > 0.1:
            result.append(idx)
            probas.append(probas)
    print(result)
    result = label_encoder.inverse_transform(result)
    
    """for idx,i in enumerate(result) : 
        result_2.append(i)
        result_2.append(probas[idx] )
    print(result)"""
    
    return result

# Step 2: Specify the input and output components
app = FastDash(callback_fn=text_to_text_function, 
                inputs=Text, 
                outputs=Text, 
                title='App title')

# Step 3: Run your app!
app.run()

# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

