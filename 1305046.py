
import numpy
import csv

def mat_factorization(R, P, Q, K, record, record2, steps=100, alpha=0.02, beta=0.02, ):
    Q = Q.T
    for step in xrange(steps):
        print step
        for ii in xrange(len(record)):
            i=record[ii]
            j=record2[ii]
            eij = R[i,j] - numpy.dot(P[i,:],Q[:,j])
            for k in xrange(K):
                P[i][k] = P[i][k] + alpha * (eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for ii in xrange(len(record)):
            i=record[ii]
            j=record2[ii]
            e = e + pow(R[i,j] - numpy.dot(P[i,:],Q[:,j]), 2)
            #for k in xrange(K):
                #e = e + (beta) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        e=e/len(record)
        e=e**(0.5)
        if e < 0.001:
            break


    return P, Q.T,e

def validation(R, P, Q, K, record, record2, steps=1, alpha=0.02, beta=0.02 ):
    Q = Q.T
    for step in xrange(steps):
        print step
        for ii in xrange(len(record)):
            i=record[ii]
            j=record2[ii]
            eij = R[i,j] - numpy.dot(P[i,:],Q[:,j])
            for k in xrange(K):
                P[i][k] = P[i][k] + alpha * (eij * Q[k][j] - beta * P[i][k])
        eR = numpy.dot(P,Q)
        e = 0
        for ii in xrange(len(record)):
            i=record[ii]
            j=record2[ii]
            e = e + pow(R[i,j] - numpy.dot(P[i,:],Q[:,j]), 2)
            #for k in xrange(K):
                #e = e + (beta) * ( pow(P[i][k],2) + pow(Q[k][j],2) )

        e=e/len(record)
        e=e**(0.5)
        if e < 0.001:
            break

    return P, Q.T,e






train_no=6000
step=10
tot_item=10000
filename = "train.json"
i = 0
reviewers = {}
items = {}
found = -1
found2 = -1

record = []
record2 = []
ratings=[]
index = 0;

#file = open("testfile.txt", "w")
#R = numpy.zeros([90600, 37336])

with open('train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        i+=1

        if(i==1):
            continue


        rid = (row[0])
        iid = (row[1])
        rat = (float)(row[2])



        if rid in reviewers:
            found = reviewers[rid]
        else:
            reviewers[rid] = len(reviewers)
            found = len(reviewers) - 1

        if iid in items:
            found2 = items[iid]
        else:
            items[iid] = len(items)
            found2 = len(items) - 1


        record.append(found)
        record2.append(found2)
        ratings.append(rat)

        if(i>=train_no):
            break

#        print found, found2,rat

#        i += 1;

from scipy import sparse

print record[0],record2[0],ratings[0]
print len(reviewers)+1

R = sparse.coo_matrix((ratings, (record, record2)), shape=(len(reviewers)+1,tot_item))
R = R.tocsc()








i = 0
reviewersValid = {}
found = -1
found2 = -1

recordValid = []
record2Valid = []
ratingsValid=[]
index = 0;

#file = open("testfile.txt", "w")
#R = numpy.zeros([90600, 37336])

with open('validation.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        i+=1

        if(i==1):
            continue


        rid = (row[0])
        iid = (row[1])
        rat = (float)(row[2])



        if rid in reviewersValid:
            found = reviewersValid[rid]
        else:
            reviewersValid[rid] = len(reviewersValid)
            found = len(reviewersValid) - 1

        if iid in items:
            found2 = items[iid]
        else:
            items[iid] = len(items)
            found2 = len(items) - 1


        recordValid.append(found)
        record2Valid.append(found2)
        ratingsValid.append(rat)

        if(i>=train_no/3):
            break


#        print found, found2,rat

#        i += 1;

from scipy import sparse

#print record[0],record2[0],ratings[0]
#print len(reviewers)+1

RValid = sparse.coo_matrix((ratingsValid, (recordValid, record2Valid)), shape=(len(reviewersValid)+1,tot_item))
RValid = RValid.tocsc()

#lamda = [0.01, 0.1, 1, 10]
#K = [10, 20, 30, 40, 50]

lamda = [ 0.1]
K = [10]

mine=9999
minQ=[]
minK=999
minLamda=999


for i in range(len(lamda)):
    for j in range(len(K)):
        N = len(reviewers)+1
        M = tot_item
        P = numpy.random.rand(N, K[j])
        Q = numpy.random.rand(M, K[j])

        nP, nQ, e = mat_factorization(R, P, Q, K[j], record, record2,step,0.02,lamda[i])

        print e



        N = len(reviewersValid)+1
        M = tot_item
        P = numpy.random.rand(N, K[j])
        Q = numpy.random.rand(M, K[j])

        nP, nQ, e = validation(RValid, P, nQ, K[j], recordValid, record2Valid,1,0.02,lamda[i])

        if(e<mine):
            minQ=nQ
            minLamda=lamda[i]
            minK=K[j]


i = 0
reviewersTest = {}
found = -1
found2 = -1

recordTest = []
record2Test = []
ratingsTest=[]
index = 0;

#file = open("testfile.txt", "w")
#R = numpy.zeros([90600, 37336])

with open('test.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        i+=1

        if(i==1):
            continue


        rid = (row[0])
        iid = (row[1])
        rat = (float)(row[2])




        if rid in reviewersTest:
            found = reviewersTest[rid]
        else:
            reviewersTest[rid] = len(reviewersTest)
            found = len(reviewersTest) - 1

        if iid in items:
            found2 = items[iid]
        else:
            items[iid] = len(items)
            found2 = len(items) - 1


        recordTest.append(found)
        record2Test.append(found2)
        ratingsTest.append(rat)

        if(i>=train_no/3):
            break


#        print found, found2,rat

#        i += 1;

from scipy import sparse

#print record[0],record2[0],ratings[0]
#print len(reviewers)+1

RTest = sparse.coo_matrix((ratingsTest, (recordTest, record2Test)), shape=(len(reviewersTest)+1,tot_item))
RTest = RTest.tocsc()

N = len(reviewersTest) + 1
M = tot_item
P = numpy.random.rand(N, minK)
Q = numpy.random.rand(M, minK)

nP, nQ, e = validation(RTest, P, minQ, minK, recordTest, record2Test, 1, 0.02, minLamda)

print reviewersTest

print minK,minLamda,e



i = 0
reviewersTest = {}
found = -1
found2 = -1

recordTest = []
record2Test = []
ratingsTest=[]
index = 0;

#file = open("testfile.txt", "w")
#R = numpy.zeros([90600, 37336])
j=0
with open('test.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        i+=1

        if(i<=(train_no/3)+1):
            continue


        rid = "0"
        iid = (row[1])
        rat = (float)(row[2])

        print iid, rat




        if rid in reviewersTest:
            found = reviewersTest[rid]
        else:
            reviewersTest[rid] = len(reviewersTest)
            found = len(reviewersTest) - 1

        if iid in items:
            found2 = items[iid]
        else:
            items[iid] = len(items)
            found2 = len(items) - 1


        recordTest.append(found)
        record2Test.append(found2)
        ratingsTest.append(rat)

        j+=1
        if(j==10):
            break




#        print found, found2,rat

#        i += 1;

print len(items)

from scipy import sparse

#print record[0],record2[0],ratings[0]
#print len(reviewers)+1

RTest = sparse.coo_matrix((ratingsTest, (recordTest, record2Test)), shape=(len(reviewersTest)+1,tot_item))

print(RTest)

RTest = RTest.tocsc()

N = len(reviewersTest) + 1
M = tot_item
P = numpy.random.rand(N, minK)
Q = numpy.random.rand(M, minK)

nP, nQ, e = validation(RTest, P, minQ, minK, recordTest, record2Test, 1, 0.02, minLamda)


nR = numpy.dot(nP, nQ.T)
print nR.shape

print "Recommendation"
check=0
while(1):
    rec=numpy.argmax(nR[0])

    print rec

    for name,val in items.items():
        if val == rec:
            print name
            check+=1

    if(check==5):
        break
    nR[0][rec]=0
