#!/usr/bin/env python
# coding: utf-8

# # Import des packages

# In[ ]:





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


# #### Import de la base de donnée

# In[2]:


data = pd.read_csv('QueryResults_2.csv')


# # Data preparation for text analysis

# ### Fonctions primaires 

# In[3]:


# Lemmatizer, permet de regrouper les mots dont le sens est très liés 
#(différentes conjugaisons d'un même verbe, singulier et pluriel du même nom).
def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w
"""
#Tokenizer, permet de tokenizer les phrases, ou texte, en liste de mot. 
def tokenizer_fct(question) :
    # print(sentence)
    question = BeautifulSoup(question).get_text()
    question = question.lower()
    #sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(question)
    return word_tokens
# stopwords remover, permet de supprimer les mots à faible valeur ajoutée (les mots très courants)
def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2
"""


# ### Fonction déstinée à la création d'un "Bag of Word"

# In[4]:


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


# # BAG Of WORDS

# ### Préprocessing

# In[5]:


questions_cleared = []
for question in data.Body:
        cleared_question = question_to_words_BW(question)
        questions_cleared.append(cleared_question)
    
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# On créé le vectorizer qui va vectorizer les mots de nos questions. 

#vectorizer = CountVectorizer(analyzer = "word", max_features = 5000) 
vectorizer = TfidfVectorizer(analyzer = "word", max_features = 5000)
    
# On fit notre vectorizer avec nos questions. 
questions_features = vectorizer.fit_transform(questions_cleared)
#questions_features = questions_features.toarray()


# ### Modèle de prédiction

# In[6]:


from sklearn.decomposition import LatentDirichletAllocation
n_topics = 20

lda = LatentDirichletAllocation(n_components = n_topics, max_iter = 5, learning_method='online',learning_offset=50.,random_state=0)

lda.fit(questions_features)


# ### Affichage des topics trouvés par notre modèle.
# 

# In[7]:


def display_topics(model, feature_names, no_top_words):
    topic_list = pd.DataFrame()
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list[topic_idx] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

no_top_words = 5
display_topics(lda,vectorizer.get_feature_names(), 5)


# In[8]:


from __future__ import print_function
import pyLDAvis

import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, questions_features, vectorizer)


# ### Création de la Pipeline regroupant le préprocessing et le modèle LDA

# In[9]:


# On créé le vectorizer qui va vectorizer les mots de nos questions. 

# Pour chaque question, on applique la fonction qui récupère les mots clés des questions posées. 
def clearing_questions(Body):
    # On créé une liste vide déstinée à receullir les mots de nos questions. 
    questions_cleared =[]
    for question in Body:
        cleared_question = question_to_words_BW(question)
        questions_cleared.append(cleared_question)
        
    return questions_cleared


# In[10]:


from sklearn import datasets, preprocessing, model_selection, ensemble, pipeline
from sklearn.experimental import enable_hist_gradient_boosting


# In[12]:


from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

n_topics = 20
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


# In[13]:


pipeline_BOW.fit(data.Body)


# In[14]:


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


# In[15]:


no_top_words = 5
topic_list = create_topic_list(pipeline_BOW['LDA'],pipeline_BOW['vectorizer'].get_feature_names_out(),no_top_words)


# ### Essai de la pipeline avant API

# In[16]:


test_text = pipeline_BOW.transform([data.Body[10]])
tag_generator(test_text)


# In[17]:


data.Body[10]


# ### Création de "Topic List'

# # API BagOfWord

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

