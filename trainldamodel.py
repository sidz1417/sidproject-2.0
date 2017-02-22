import json
import cPickle as pickle
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
import nltk 
import logging
#from nltk.tag.perceptron import PerceptronTagger
import gensim
from gensim import corpora, models ,similarities	
from gensim.models import LdaModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print "finished importing....reading from file...."

location=r'amazonreviews.csv'
df=pd.read_csv(location,header=None)

print "finished reading from file....loading stopwords list...."

with open("stoplist.pkl","rb") as f1:
	stops=pickle.load(f1)

#tagger = PerceptronTagger()
tokenizer = RegexpTokenizer(r'\w+')

review_count=5000
docset=[]


for review in df[4]:
	review_count-=1
	if review_count>0:
		docset.append(review)
	else:
	    break 

texts=[]

def preprocess(docset,texts):
	for doc in docset:
		raw=str(doc).lower()
		tokens=tokenizer.tokenize(raw)
		tokens=[w for w in tokens if not w in stops]
		texts.append(tokens)

print "started preprocessing...."

preprocess(docset,texts)

print "finished preprocessing....Creating dictionary...."
	
dictionary = corpora.Dictionary(texts)
dictionary.save('reviewsdict2.dict')	

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('reviewcorpus2.mm', corpus)

mm=corpora.MmCorpus('reviewcorpus2.mm')
#dictionary=dictionary.load('reviewsdict.dict')

ldamodel = models.LdaMulticore(corpus=mm, num_topics=10, id2word = dictionary, passes=20,workers=2)
ldamodel.save('reviewlda2.model')

print(ldamodel.print_topics(num_topics=10, num_words=4))


def predict(query):		
    temp=query.lower()	
    tokens=tokenizer.tokenize(temp)
    tokens=[w for w in tokens if not w in stops]
    
    dictionary = corpora.Dictionary.load('reviewsdict2.dict')
    lda=LdaModel.load('reviewlda2.model')

    rev_vec = dictionary.doc2bow(tokens)
    topics = sorted(lda[rev_vec],key=lambda x:x[1],reverse=True)
    print(topics)
    print(topics[0][0])

predict("service and shopping was really good...")


		

    
	





			