import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import json,csv
import string
from nltk.tag import pos_tag
import pyLDAvis
from pyLDAvis import gensim
from gensim import corpora, models
from gensim import similarities
from nltk.stem.snowball import SnowballStemmer
import sys
from itertools import chain

reload(sys)
sys.setdefaultencoding('utf8')

##LDA loop time setup
SOME_FIXED_SEED = 42
np.random.seed(SOME_FIXED_SEED)

stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')

head = []
movie_id = []
with open("plot_summaries.txt",'r') as myfile:
    for line in myfile:
        movie_id.append(line.split('\t')[0])
        head.append(line.split('\t')[1])
#document,topic number setup
topic_number_setup = 20
document_setup = 40000

synopses = head[:document_setup]
selected_id = movie_id[:document_setup]

# def tokenize_and_stem(text):
#     #first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
#     tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
#     filtered_tokens = []
#     #filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
#     for token in tokens:
#         if re.search('[a-zA-Z]', token):
#             filtered_tokens.append(token)
#     stems = [stemmer.stem(t) for t in filtered_tokens]
#     return stems

def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text.decode('utf-8')) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS' and pos !='VB' and pos!='VBD'and pos!='VBG'and
    pos!='VBN'and pos!='VBP'and pos!='VBZ' and pos!='CD' and pos!='CC'and pos!='LS' and pos!='MD' and pos!='POS' and pos!='PDT' and pos!='DT' ]
    return non_propernouns

#remove proper names
preprocess = [strip_proppers(doc) for doc in synopses]

#tokenize
tokenized_text = [strip_proppers_POS(text) for text in preprocess]
#tokenized_text = [tokenize_and_stem(text) for text in preprocess]

#remove stop words
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]
#keep words that only contains letters
texts = [[word for word in text if re.match('[a-zA-Z]+', word)]for text in texts]

#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(texts)

#remove extremes no_below 1 appearance, no above 0.8 percent of corpus
dictionary.filter_extremes(no_below=1, no_above=0.8)

#convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(text) for text in texts]

#convert the dictionary to a bag of words corpus for reference
lda = models.LdaModel(corpus, num_topics=topic_number_setup,
                            id2word=dictionary,
                            minimum_probability=0.05
                            # update_every=5,
                            # chunksize=10000,
                            # passes=20
                            )
##### for each movie in corpus, give topic and probability(>0.01)
lda_corpus = lda[corpus]



######doc topic matrix: [document index, [topic,probability],document id, document content]
doc_topic_mat = zip(range(0,document_setup),lda_corpus,selected_id,synopses)

###### matrix of words in topic
topics_matrix = lda.show_topics(num_topics=topic_number_setup,num_words=10, log=False, formatted=True)
topics_matrix = np.array(topics_matrix)

#for each topic and each document, give probability
cluster1 = []
for k,i,j,l in doc_topic_mat:
    for x in i :
        # for each (topic,probability) for each document
        # append [(topic, probability),document id] to cluster1
        cluster1.append((x,j,l))

###### Save topics
with open('lda_topic.txt','w') as file:
    for i in lda.show_topics(num_topics=topic_number_setup):
        file.write(str(i)+'\n')

##### topic cluster visualization
##### topic term relation json save
movies = pyLDAvis.gensim.prepare(lda,corpus, dictionary)
pyLDAvis.save_html(movies, 'LDA_Visualization.html')

######Topic-term relationship matrix
pyLDAvis.save_json(movies, 'topic_term.json')
with open('topic_term.json') as json_data:
    d = json.load(json_data)
mat = np.column_stack(( d['token.table']['Topic'], d['token.table']['Freq'],d['token.table']['Term']))

##### load movie metadata:
meta_dict = {}
with open("movie.metadata.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        meta_dict[line[0]] = line[2]

####Enable topic document search
def enable_search():
    query = raw_input('Enter your search query: ')
    #method one
    test_doc =query.split()
    doc_bow = dictionary.doc2bow(test_doc)
    output = lda[doc_bow]
    #print output
    #show topic words
    print 'Query topic relationship:'
    for n in output:
        i = n[0]
        print n
        for j in topics_matrix[i][1].split('+'):
            print j.split('*')[1],
        print '\n'
    #user choose topic
    topic_number = int(raw_input('Choose your topic: '))
    document_probability = []
    for i in cluster1:
        #print i[0],i[1]
        if int(i[0][0]) == topic_number:
                document_probability.append((i[0][1],i[1]))
    sort_doc = sorted(document_probability, key = lambda x:x[0],reverse = True)
    print 'Top related movies of this topic:'
    for i in sort_doc:
        if float(i[0])>0.5:
            print (i[0],meta_dict[i[1]])
    user_choose = raw_input('New Search(Yes/No): ')
    while user_choose != 'Yes' and user_choose != 'No':
        print 'Please choose Yes/No!'
        user_choose = raw_input('New Search(Yes/No): ')
    if user_choose == 'Yes':
        enable_search()
enable_search()




# ##############################Used methods################################################

# ##### Print topic belonging of documents
# # index = 0
# # for i in corpus[1:10]:
# #     print index
# #     index +=1
# #     print lda[i]

##### New document, topic prediction
# test_doc ='father mother and me.'.split()
# doc_bow = dictionary.doc2bow(test_doc)
# print lda[doc_bow]


# ##### setup threshold ? or use 0.01
# # scores = list(chain(*[[score for topic_id,score in topic] \
# #                       for topic in [doc for doc in lda_corpus]]))
# # threshold = sum(scores)/len(scores)
# # print threshold
#
#
# ##### New movie Classification
# ##### Classification one movie outside
# #transfer to bag-of-word
# # for i in range(101,105):
# #     print i
# #     test_doc = list(head[i].decode('utf-8'))
# #     doc_bow = dictionary.doc2bow(test_doc)
# #     doc_lda = lda[doc_bow]
# #     print doc_lda
#
# ##### Classification cluster of movies, document belonging
# #
# # def get_doc_topics(lda, bow):
# #     gamma, _ = lda.inference([bow]) #gamma is the unnormalised topic scores per document
# #     topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution
# #     print topic_dist
# #     #return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)]
# #
# # for i in range(100,101):
# #     test_doc=list(head[i].decode('utf-8'))
# #     doc_bow = dictionary.doc2bow(test_doc)
# #     print get_doc_topics(lda, doc_bow)
