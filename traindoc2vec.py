import gensim
from gensim import utils 
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
from random import shuffle

print "finished importing"

starttimer=timeit.default_timer()

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

#sources={'testset/test1star.txt':'TEST_ONE','trainset/train1star.txt':'TRAIN_ONE','testset/test2star.txt':'TEST_TWO','trainset/train2star.txt':'TRAIN_TWO','testset/test3star.txt':'TEST_THREE','trainset/train3star.txt':'TRAIN_THREE','testset/test4star.txt':'TEST_FOUR','trainset/train4star.txt':'TRAIN_FOUR','testset/test5star.txt':'TEST_FIVE','trainset/train5star.txt':'TRAIN_FIVE','unsupervised.txt':'TRAIN_UNS'}	

sources={'testset/test1star.txt':'TEST_ONE','trainset/train1star.txt':'TRAIN_ONE','testset/test2star.txt':'TEST_TWO','trainset/train2star.txt':'TRAIN_TWO','testset/test3star.txt':'TEST_THREE','trainset/train3star.txt':'TRAIN_THREE','testset/test4star.txt':'TEST_FOUR','trainset/train4star.txt':'TRAIN_FOUR','testset/test5star.txt':'TEST_FIVE','trainset/train5star.txt':'TRAIN_FIVE'}   

print "started tagging sentences"

sentences=taggedlinedocument(sources)

print "finished in %rs" %(default_timer()-starttimer)

print "started training"

dimensions=20
model = Doc2Vec(alpha=0.025, min_alpha=0.025,size=dimensions,min_count=1,iter=10,workers=2)  # use fixed learning rate
no_of_epochs=0
model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())
    model.alpha -= 0.002  # decrease the learning rate
    model.min_alpha = model.alpha  # fix the learning rate, no decay
    no_of_epochs+=1
    print "%d epochs completed" %(no_of_epochs)

	
print "finished training in %rs" %(default_timer()-starttimer)

#save the model 	
model.save('doc2vecmodel.doc2vec')
