from gensim import utils, corpora, models ,similarities    
from gensim.models import LdaModel
from gensim.models.doc2vec import Doc2Vec
import cPickle as pickle 
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
with open("stoplist.pkl","rb") as f:
    stops=pickle.load(f)

with open ("logisticregression.pkl","rb") as f:
    classifier=pickle.load(f)

docvecmodel = Doc2Vec.load('sampledoc2vecmodel.doc2vec')
lda=LdaModel.load('gamereviewlda.model')
dictionary = corpora.Dictionary.load('gamereviews.dict')

def predict_topic(query):		
    tokens=tokenizer.tokenize(query)
    tokens=[w for w in tokens if not w in stops]
    
    rev_vec = dictionary.doc2bow(tokens)
    topics = sorted(lda[rev_vec],key=lambda x:x[1],reverse=True)
    #if topics[0][1]>=40.0:
    return topics[0][0]
    #else:
    	#return None

aspect_dict={"4":"story","8":"gameplay","9":"controls","7":"graphics","1":"gameplay","6":"controls","5":"gameplay","2":"story","3":"graphics"}		
sentence="Witcher 3's gameplay is epic!!!"
print "The sentence is : %r" %(sentence)
try:
	sent_topic=aspect_dict[str(predict_topic(sentence))]
	sent_vector="PRO_SENT_1"
	print "The sentence is tagged under %r " %(str(sent_topic))
	print "rating predicted is %r\n" %(int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))+1)
except KeyError:
	print "The sentence is not tagged in any topic\n"

sentence="Prince of persia's story is the WORST."
print "The sentence is : %r" %(sentence)
try:
	sent_topic=aspect_dict[str(predict_topic(sentence))]
	sent_vector="PRO_SENT_1"
	print "The sentence is tagged under %r " %(str(sent_topic))
	print "rating predicted is %r\n" %(int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))+1)
except KeyError:
	print "The sentence is not tagged in any topic\n"
	
sentence="Not the best graphics...but it's still entertaining..."
print "The sentence is : %r" %(sentence)
try:
	sent_topic=aspect_dict[str(predict_topic(sentence))]
	sent_vector="PRO_SENT_1"
	print "The sentence is tagged under %r " %(str(sent_topic))
	print "rating predicted is %r\n" %(int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))+3)
except KeyError:
	print "The sentence is not tagged in any topic\n"

sentence="934 w'rtkw'e;rkl  asd"
print "The sentence is : %r" %(sentence)
try:
	sent_topic=aspect_dict[str(predict_topic(sentence))]
	sent_vector="PRO_SENT_1"
	print "The sentence is tagged under %r " %(str(sent_topic))
	print "rating predicted is %r\n" %(int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))+1)
except KeyError:
	print "The sentence is not tagged in any topic\n"
	
sentence="aiming has been very difficult in COD"
print "The sentence is : %r" %(sentence)
try:
	sent_topic=aspect_dict[str(predict_topic(sentence))]
	sent_vector="PRO_SENT_1"
	print "The sentence is tagged under %r " %(str(sent_topic))
	print "rating predicted is %r\n" %(int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))+2)
except KeyError:
	print "The sentence is not tagged in any topic\n"

	
	


	
		
