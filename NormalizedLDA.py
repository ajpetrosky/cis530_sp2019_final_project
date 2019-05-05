# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:35:53 2019

@author: yuqit
"""
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#import pandas as pd

df = pd.read_csv(r"C:\Users\yuqit\Dropbox\_C04\project\NormalizedStudentGenie_Train.csv",encoding='utf-8')

data = df.messages.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        if isinstance(sentence, float):
            sentence = str(sentence)
        yield(gensim.utils.simple_preprocess(sentence.encode('utf-8'), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[:1])

##########prepocessing
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

######run from here
# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

#import spacy
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)# python3 -m spacy download en
#nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
#data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_lemmatized = data_words_bigrams

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


#Choose N
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

%matplotlib inline
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, 
                                                        start=2, limit=8, step=1)

x = range(2, 8, 1)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    
best_n = 1#FILL THE BEST COHERENCE SCORE
optimal_model = model_list[best_n]
model_copy = model_list[best_n]

#4 topics
model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Select the model and print the topics
model_topics = model.show_topics(formatted=False)
pprint(model.print_topics(num_words=10))


# 1. Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

#topics = lda_model.show_topics(formatted=False)
topics = model.show_topics(formatted=False)

fig, axes = plt.subplots(4, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

from gensim.test.utils import datapath
# Save model to disk.
temp_file = datapath("model_genie_student") ##CHANGE NAME
model.save(temp_file)

# Load a potentially pretrained model from disk.
model = gensim.models.ldamodel.LdaModel.load(temp_file)

# Dev data
#df_dev = pd.read_csv("Data/NLP/GenieMessagesDev.csv",  lineterminator='\n', encoding='utf-8')
df_dev = pd.read_csv(r"C:\Users\yuqit\Dropbox\_C04\project\NormalizedStudentGenie_Dev.csv", encoding='utf-8')

#df_dev = df_dev[['Student_Year','Combined.messages.to.Genie_ALL', 'Combined.messages.from.Genie_ALL']]
df_test = pd.read_csv(r"C:\Users\yuqit\Dropbox\_C04\project\NormalizedStudentGenie_Test.csv", encoding='utf-8')

#df_dev.columns = ['stud', 'content1', 'content2']
#df_dev['content'] = df_dev['content1'] + ' ' + df_dev['content2']
#df_dev['content'].values.astype('U')
    #run model on one test example
#    other_texts = [item for item in df_test['messages']]
    #Run preprocessing for dev and test
def get_lemmatized_text(df_test):
    test_data = df_test.messages.values.tolist()
    test_data_words = list(sent_to_words(test_data))
    # Remove Stop Words
    #test_data_words_nostops = remove_stopwords(test_data_words)
    # Form Bigrams
    test_data_words_bigrams = make_bigrams(test_data_words)
    return test_data_words_bigrams


texts_dev = get_lemmatized_text(df_dev)
texts_test = get_lemmatized_text(df_test)

def get_prediction(other_text, optimal_model):
    #run model on one test example
#    other_texts = [item for item in df_test['messages']]
    #Run preprocessing for dev and test
#    test_data = df_test.messages.values.tolist()
#    test_data_words = list(sent_to_words(test_data))
#    # Remove Stop Words
#    #test_data_words_nostops = remove_stopwords(test_data_words)
#    # Form Bigrams
#    test_data_words_bigrams = make_bigrams(test_data_words)
#    test_data_lemmatized = test_data_words_bigrams
#    
#    other_text = test_data_lemmatized
    other_corpus = [id2word.doc2bow(text) for text in other_text]
    result = [optimal_model.get_document_topics(doc) for doc in other_corpus]
    result_max = [sorted(doc, key=lambda x: x[1], reverse=True)[0][0] for doc in result]
    
    return result_max

result_train = get_prediction(texts, model)
result_dev = get_prediction(texts_dev, model)
result_test = get_prediction(texts_test, model)

with open(r"C:\Users\yuqit\Dropbox\_C04\project\normalized_genie_dev_8.csv", "w") as f:
    for item in result_dev:
        f.write(str(item))
        f.write('\n')
        
with open(r"C:\Users\yuqit\Dropbox\_C04\project\normalized_genie_test_8.csv", "w") as f:
    for item in result_test:
        f.write(str(item))
        f.write('\n')
        
with open(r"C:\Users\yuqit\Dropbox\_C04\project\normalized_genie_train_8.csv", "w") as f:
    for item in result_train:
        f.write(str(item))
        f.write('\n')
        
for i in range(8):
    print(result_test.count(i)/len(result_test))



#get coherence for dev and test

def get_coherence(text, model):
#    #coherencemodel = CoherenceModel(model=model, texts=dev_data_lemmatized, dictionary=id2word, coherence='c_v')
#    #coherence_values_dev.append(coherencemodel.get_coherence())
#    test_data = df.messages.values.tolist()
#    test_data_words = list(sent_to_words(test_data))
#    # Remove Stop Words
#    #test_data_words_nostops = remove_stopwords(test_data_words)
#    # Form Bigrams
#    test_data_words_bigrams = make_bigrams(test_data_words)
#    test_data_lemmatized = test_data_words_bigrams
    coherencemodel = CoherenceModel(model=model, texts=text, dictionary=id2word, coherence='c_v')  
    print(coherencemodel.get_coherence())

get_coherence(texts, model)    
get_coherence(texts_dev, model)
get_coherence(texts_test, model)  

#get sih score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

def get_silhouette_score(text,label,sample_size):
    string_list = [" ".join(item) for item in text]
    vect = TfidfVectorizer(min_df=3, max_features=5000)
    X = vect.fit_transform(string_list) #x is an np.array of messages
    sil = silhouette_score(X, label, sample_size = sample_size) #y is the select topic label 
    print(sil)
    return(sil)
    
get_silhouette_score(texts,result_train,5000)
get_silhouette_score(texts_dev,result_dev,5000)
get_silhouette_score(texts_test,result_test,5000)    
