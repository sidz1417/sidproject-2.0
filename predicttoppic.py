from gensim import corpora, models ,similarities	
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import RegexpTokenizer
import cPickle as pickle

lda=LdaModel.load('gamereviewlda.model')
dictionary = corpora.Dictionary.load('gamereviews.dict')
tokenizer = RegexpTokenizer(r'\w+')
with open("stoplist.pkl","rb") as f5:
    stops=pickle.load(f5)

def predict(query):		
	temp=query.lower()	
	tokens=tokenizer.tokenize(temp)
	tokens=[w for w in tokens if not w in stops]
    
	rev_vec = dictionary.doc2bow(tokens)
	topics = sorted(lda[rev_vec],key=lambda x:x[1],reverse=True)
	print(topics)
	print "predicted topic from lda is %r with probability %r" %(topics[0][0],topics[0][1])
 
predict("COD has the best controls among shooting games....")

