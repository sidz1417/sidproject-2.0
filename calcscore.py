#train with 10 epochs 
import cPickle as pickle  
import numpy as np 
from gensim import utils, corpora, models ,similarities    
from gensim.models import LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import timeit 
from random import shuffle
from nltk.tokenize import RegexpTokenizer

print "finished importing"
starttime=timeit.default_timer()
tokenizer = RegexpTokenizer(r'\w+')
f2=open("processedreviews.txt","r")
with open("stoplist.pkl","rb") as f5:
    stops=pickle.load(f5)
'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class taggedlinedocument(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

sources={'processedreviews.txt':'PRO_SENT'}

print "started tagging"

sentences=taggedlinedocument(sources)

print "finished tagging in %rs" %(timeit.default_timer()-starttime)

print "started training"

dimensions=20
model = Doc2Vec(alpha=0.025, min_alpha=0.025,size=dimensions,min_count=1,iter=10,workers=2)  # use fixed learning rate
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay

print "finished training in %rs" %(timeit.default_timer()-starttime)

#save the model 	
model.save('sampledoc2vecmodel.doc2vec')
'''

def predict_topic(query):		
    tokens=tokenizer.tokenize(query)
    tokens=[w for w in tokens if not w in stops]
    
    rev_vec = dictionary.doc2bow(tokens)
    topics = sorted(lda[rev_vec],key=lambda x:x[1],reverse=True)
    #if topics[0][1]>=40.0:
    return topics[0][0]
    #else:
    	#return None

#example aspect ratings 
aspect_dict={"8":"gameplay","9":"controls","7":"gameplay","1":"gameplay","4":"story","6":"controls","0":"graphics","2":"graphics","3":"story","5":"gameplay"}
aspectratings_dict={"gameplay":0,"story":0,"controls":0,"graphics":0}
aspectlines_dict={"gameplay":0,"story":0,"controls":0,"graphics":0}

docvecmodel = Doc2Vec.load('sampledoc2vecmodel.doc2vec')
lda=LdaModel.load('gamereviewlda.model')
dictionary = corpora.Dictionary.load('gamereviews.dict')
with open ("logisticregression.pkl","rb") as f:
    classifier=pickle.load(f)

line_number=0

print "calculating aspect scores"

for line in f2:
	if predict_topic(line):
		sent_topic=aspect_dict[str(predict_topic(line))]
		sent_vector = 'PRO_SENT_' + str(line_number)
		sent_rating=int(classifier.predict(docvecmodel.docvecs[sent_vector].reshape(1,-1)))
        print sent_rating
        aspectratings_dict[sent_topic]+=sent_rating
        aspectlines_dict[sent_topic]+=1

print "finished in %rs" %(timeit.default_timer()-starttime)
avg_rating=0
#compute average rating for each aspect 
for category in aspectlines_dict:
    if aspectlines_dict[category]!=0:
	   avg_rating=aspectratings_dict[category]/aspectlines_dict[category]
    else:
       avg_rating=0
    print "%s : %f" %(category,avg_rating)











