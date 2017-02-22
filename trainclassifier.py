from gensim.models import Doc2Vec
#import logging
import numpy
from sklearn.linear_model import LogisticRegression
import timeit 
#from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle 

print "Finished importing"

starttimer=timeit.default_timer()

model = Doc2Vec.load('doc2vecmodel.doc2vec')

'''
f=open("doc2veclabels.txt","w")
for label,vec in sorted(model.docvecs.doctags.items()):
    print >>f,label 
'''

def linecount(filename):
    lines=0
    with open(filename,"r") as f:
        for line in f:
            lines+=1
    return lines 

test1=linecount("testset/test1star.txt")
test2=linecount("testset/test2star.txt")
test3=linecount("testset/test3star.txt")
test4=linecount("testset/test4star.txt")
test5=linecount("testset/test5star.txt")

train1=linecount("trainset/train1star.txt")
train2=linecount("trainset/train1star.txt")
train3=linecount("trainset/train1star.txt")
train4=linecount("trainset/train1star.txt")
train5=linecount("trainset/train1star.txt")

dimensions=20

trainsize=train1+train2+train3+train4+train5
testsize=test1+test2+test3+test4+test5

train_arrays = numpy.zeros((trainsize, dimensions))
train_labels = numpy.zeros(trainsize)
test_arrays = numpy.zeros((testsize, dimensions))
test_labels = numpy.zeros(testsize)

print "forming train array...."

for i in xrange(train1):
    prefix_train_1star = 'TRAIN_ONE_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_1star]
    train_labels[i] = 1

for i in xrange(train2):
    prefix_train_2star = 'TRAIN_TWO_' + str(i)
    train_arrays[i+train1] = model.docvecs[prefix_train_2star]
    train_labels[i+train1] = 2

for i in xrange(train3):
    prefix_train_3star = 'TRAIN_THREE_' + str(i)
    train_arrays[i+train2] = model.docvecs[prefix_train_3star]
    train_labels[i+train2] = 3

for i in xrange(train4):
    prefix_train_4star = 'TRAIN_FOUR_' + str(i)
    train_arrays[i+train3] = model.docvecs[prefix_train_4star]
    train_labels[i+train3] = 4

for i in xrange(train5):
    prefix_train_5star = 'TRAIN_FIVE_' + str(i)
    train_arrays[i+train4] = model.docvecs[prefix_train_5star]
    train_labels[i+train4] = 5

print "finished in %rs" %(default_timer()-starttimer)

print "forming test array...."

for i in xrange(test1):
    prefix_test_1star = 'TEST_ONE_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_1star]
    test_labels[i] = 1

for i in xrange(test2):
    prefix_test_2star = 'TEST_TWO_' + str(i)
    test_arrays[i+test1] = model.docvecs[prefix_test_2star]
    test_labels[i+test1] = 2

for i in xrange(test3):
    prefix_test_3star = 'TEST_THREE_' + str(i)
    test_arrays[i+test2] = model.docvecs[prefix_test_3star]
    test_labels[i+test2] = 3

for i in xrange(test4):
    prefix_test_4star = 'TEST_FOUR_' + str(i)
    test_arrays[i+test3] = model.docvecs[prefix_test_4star]
    test_labels[i+test3] = 4

for i in xrange(test5):
    prefix_test_5star = 'TEST_FIVE_' + str(i)
    test_arrays[i+test4] = model.docvecs[prefix_test_5star]
    test_labels[i+test4] = 5

print "finished in %rs" %(default_timer()-starttimer)

print "training classifier...." 
#classifier = RandomForestClassifier(n_estimators=100,n_jobs=2)
classifier = LogisticRegression()

classifier.fit(train_arrays, train_labels)

print "finished in %rs" %(default_timer()-starttimer)

with open("logisticregression.pkl","wb") as f:
    pickle.dump(classifier,f)

print classifier.score(test_arrays, test_labels)
