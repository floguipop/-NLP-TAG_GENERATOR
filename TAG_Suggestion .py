#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


data = pd.read_csv('QueryResults_2.csv')


# In[4]:


data


# ### Les titres sont tout aussi important que le Body. Nous pouvons regrouper les deux. 

# In[5]:


data.Body = data.Title + " " + data.Body 


# In[6]:


data.Body


# ## Gestion des TAGS ( cible ) 

# #### Définition des fonction permettant de cleaner les TAGS

# In[7]:


def tag_remover(tag):
    tag = tag.replace('>' , " ").replace('<', " ")
    return tag

def tag_to_word_list(tag):
    tag = BeautifulSoup(tag).get_text()
    tag = tag.split()
    return(tag)


# #### Parcourons tous les TAGS pour leur appliquer nos fonctions de clean.

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


# #### Identifions, les TAGS les plus courants, cela pourra certainement nous être utile.

# In[32]:


sns.set(font_scale = 1)


# In[34]:


import matplotlib.pylab as plt
tags = pd.DataFrame()
tags['Tags'] = tag_bank
# for legend text
plt.figure(figsize = (10,5))

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


# # Data preparation for text analysis

# ### Fonctions primaires 

# In[4]:



def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w


# ### Fonction déstinée à la création d'un "Bag of Word"

# In[5]:


def question_to_words_BW(question):
    # On récupère le texte
    question_text = BeautifulSoup(question).get_text()
    
    #On ne garde que les caractères intéressants. 
    question = re.sub("[^a-zA-Z+#]"," ", question_text)
    
    # On tokenize nos questions en mots.
    words = question.lower().split()
    
    # On retire les mots clés plus courants avec peu de valeur ajoutée. 
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
        
    # On lemmatize les mots
    meaningful_words = lemma_fct(meaningful_words)
    
    #On retourne une liste comprenant les mots ainsi récupérés. 
    return(" ".join( meaningful_words))


# ### Fonctions déstinée à la création du modèle Word2vec. 

# In[6]:


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


# In[7]:


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


# In[8]:


#tokenizer qui décompose en phrases via la ponctuation= punkt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# # BAG Of WORDS

# ### Préprocessing

# In[15]:


questions_cleared = []
for question in data.Body:
        cleared_question = question_to_words_BW(question)
        questions_cleared.append(cleared_question)
        
# On créé le vectorizer qui va vectorizer les mots de nos questions. 
#vectorizer = CountVectorizer(analyzer = "word", max_features = 5000) 
vectorizer = TfidfVectorizer(analyzer = "word", max_features = 5000)


# On fit notre vectorizer avec nos questions. 
questions_features = vectorizer.fit_transform(questions_cleared)
#questions_features = questions_features.toarray()


# ### Modèle de prédiction

# In[16]:


from sklearn.decomposition import LatentDirichletAllocation
n_topics = 20

lda = LatentDirichletAllocation(n_components = n_topics, max_iter = 5, learning_method='online',learning_offset=50.,random_state=0)

lda.fit(questions_features)


# #### Ici, notre modèle nous renvoie un vecteur de la taille n_topics, avec pour chacun d'entre eux la probabilité d'appartencance de notre quetioni en input 

# In[17]:


lda.transform(questions_features[1])


# ### Affichage des topics trouvés par notre modèle.
# 

# In[18]:


def display_topics(model, feature_names, no_top_words):
    topic_list = pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

no_top_words = 5
display_topics(lda,vectorizer.get_feature_names(), 5)


# In[19]:


from __future__ import print_function
import pyLDAvis

import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, questions_features, vectorizer)


# ### Création de la Pipeline regroupant le préprocessing et le modèle LDA

# In[20]:


# On créé le vectorizer qui va vectorizer les mots de nos questions. 

# Pour chaque question, on applique la fonction qui récupère les mots clés des questions posées. 
def clearing_questions(Body):
    # On créé une liste vide déstinée à receullir les mots de nos questions. 
    questions_cleared =[]
    for question in Body:
        cleared_question = question_to_words_BW(question)
        questions_cleared.append(cleared_question)
        
    return questions_cleared


# In[21]:


from sklearn import datasets, preprocessing, model_selection, ensemble, pipeline
from sklearn.experimental import enable_hist_gradient_boosting


# In[22]:


from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import LatentDirichletAllocation
n_topics = 20

from sklearn.pipeline import Pipeline
transformer_1 = FunctionTransformer(clearing_questions)
"""
pipeline_BOW = Pipeline([
   ('transformer', transformer_1), ('vectorizer',CountVectorizer(analyzer = "word", max_features = 5000)),
   ('LDA',LatentDirichletAllocation(n_components = n_topics, max_iter = 5,\
                                                                     learning_method='online',learning_offset=50.,random_state=0))
])
"""
pipeline_BOW = Pipeline([
   ('transformer', transformer_1), ('vectorizer',TfidfVectorizer(analyzer = "word", max_features = 5000)),
   ('LDA',LatentDirichletAllocation(n_components = n_topics, max_iter = 5,\
                                                                     learning_method='online',learning_offset=50.,random_state=0))
])


# In[ ]:


pipeline_BOW.fit(data.Body)


# In[ ]:


def create_topic_list(model, feature_names, no_top_words):
    topic_list = pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        topic_list[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topic_list

def tag_generator(questions):
    for question in questions:
        new_document_topic = np.where(question == question.max())
        new_document_topic = int(new_document_topic[0])
        print('La répartition des proba d\'appartenance aux différents TOPICS est \n  ', question)
        print('Les tags suggérés pour ce document sont :  \n',topic_list.T.iloc[new_document_topic,:5])
    return topic_list.T.iloc[new_document_topic,:5].tolist()


# In[ ]:


no_top_words = 5
topic_list = create_topic_list(pipeline_BOW['LDA'],pipeline_BOW['vectorizer'].get_feature_names_out(),no_top_words)


# ### Essai de la pipeline avant API

# In[16]:


test_text = pipeline_BOW.transform([data.Body[10]])
tag_generator(test_text)


# # API Tf-Idf

# In[18]:


from fast_dash import FastDash
from fast_dash.Components import Text

# Step 1: Define your model inference
def text_to_text_function(question):
    
    result = pipeline_BOW.transform([question])
    result = tag_generator(result)
    
    return result

# Step 2: Specify the input and output components
app = FastDash(callback_fn=text_to_text_function, 
                inputs=Text, 
                outputs=Text, 
                title='Unsupervised Question TAG Generator')

# Step 3: Run your app!
app.run()

# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


# # WORD2VEC Model

# In[ ]:


sentences = []
for question in data['Body']:
    sentences += (question_to_sentences(question,tokenizer))


# In[ ]:


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

# In[ ]:


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


# In[ ]:


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


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(trainDataVecs,target_to_train_encoded,test_size=0.3)


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


model_to_set = OneVsRestClassifier(SVC(kernel="poly"))

parameters = {
    "estimator__C": [0.1,1,10],
    "estimator__kernel": ["poly","rbf"],
    "estimator__degree":[1, 2, 3],
}

model_tunning = GridSearchCV(model_to_set, param_grid=parameters)

model_tunning.fit(X_train[0:5000], y_train[0:5000])

print (model_tunning.best_score_)
print (model_tunning.best_params_)


# In[ ]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

#Avec Probas :
clf = OneVsRestClassifier(SVC(probability=True))
clf.fit(X_train[0:1000], y_train[0:1000])

#Sans Probas : 
#clf = OneVsRestClassifier(SVC(probability=False)).fit(X_train, y_train)


# In[ ]:


x = np.sort(clf.predict_proba(X_train[0:1]))
x[x>0.1]


# ### Pipeline

# ### Créons une pipeline afin de faciliter la prédiction de notre notre modèle. 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
target_to_train_encoded = label_encoder.fit_transform(target_to_train)

X_train, X_test, y_train, y_test = train_test_split(data_to_train,target_to_train_encoded,test_size=0.3)


# In[ ]:


len(target_to_train_encoded)


# In[ ]:


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


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# ### Création de la pipeline 

# In[ ]:


pipeline = Pipeline([
   ('transformer', transformer_2), ('clf',OneVsRestClassifier(SVC(probability=True))), 
])


# In[ ]:


pipeline.fit(X_train, y_train)


# ### Prédiction du classifier

# In[36]:


clean_test_questions = []
data_to_test = data.Body[12356:12359]
for question in data_to_test:
    clean_test_questions.append(sentence_to_wordlist( question,         remove_stopwords=True ))
    
print ("Creating average feature vecs for test questions")
testDataVecs = getAvgFeatureVecs(clean_test_questions, model, num_features)

prediction = clf.predict(testDataVecs)
prediction = label_encoder.inverse_transform(prediction)
proba = clf.predict_proba(testDataVecs)
for i in range(len(prediction)):
    print('le texte n°',i,'est : \n', BeautifulSoup(data_to_test.iloc[i]).get_text() )
    print('LA PREDICTION EST :' , prediction[i])
    print('VECTEUR DE PROBA', proba[i])


# ### Optional : Clustering

# In[ ]:


from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.vectors
num_clusters = int(word_vectors.shape[0] / 5)

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")


# In[ ]:


pipeline.predict_proba(X_train[200:201])[0]


# In[ ]:


probas = pipeline.predict_proba(X_train[200:201])[0]
print(probas)

result = []
for idx, prob in enumerate(probas) : 
    if prob > 0.1:
        result.append(idx)
result = label_encoder.inverse_transform(result)
print(result)


# In[ ]:


pipeline.predict(X_train[200:201])[-1]


# In[ ]:


probas = pipeline.predict_proba(X_train[200:201])[0]
print(probas)

result = []
for idx, prob in enumerate(probas) : 
    if prob > 0.1:
        result.append(idx)
result = label_encoder.inverse_transform(result)
print(result)


# In[ ]:


result_2 =[]
for idx,i in enumerate(result) : 
    result_2.append(i)
    result_2.append(probas[idx])
result_2


# # API Word2Vec

# In[ ]:


from fast_dash import FastDash
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
    return result

# Step 2: Specify the input and output components
app = FastDash(callback_fn=text_to_text_function, 
                inputs=Text, 
                outputs=Text, 
                title='App title')

# Step 3: Run your app!
app.run()

# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


# # API BagOfWords

# In[ ]:


from fast_dash import FastDash
from fast_dash.Components import Text

# Step 1: Define your model inference
def text_to_text_function(question):
    
    result = pipeline_BOW.transform([question])
    
    
    return result

# Step 2: Specify the input and output components
app = FastDash(callback_fn=text_to_text_function, 
                inputs=Text, 
                outputs=Text, 
                title='Tag Generator')

# Step 3: Run your app!
app.run()

# * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

