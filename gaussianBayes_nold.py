#Casey Nold; nold@pdx.edu; ML HW4

import scipy as sp
import random as r

def main():

    
    trainPos = '/Users/caseynold/Desktop/CS545/ML HW 4/train.pos'
    trainNeg = '/Users/caseynold/Desktop/CS545/ML HW 4/train.neg'
    test = '/Users/caseynold/Desktop/CS545/ML HW 4/test.data'

    # read files
    spamFeatures = read(trainPos)
    emailFeatures = read(trainNeg)
    testFeatures = read(test)

    # find the prior probability of each class, should be 40/60 spam/email
    spamTrain_pos, emailTrain_pos = probability(spamFeatures)
    spamTrain_neg, emailTrain_neg = probability(emailFeatures)

    #calculating the prior probability
    emailProbability = emailTrain_neg/(emailTrain_neg + spamTrain_pos)
    spamProbability = spamTrain_pos/(emailTrain_neg + spamTrain_pos)
    
    # calculate the standard deviation and mean of both classes
    meanEmail,stdEmail = probablisticModel(emailFeatures)
    meanSpam,stdSpam = probablisticModel(spamFeatures)

    # print stdDev and mean out for both classes; exclude the last element which is the label
    print("Email Class Mean:", meanEmail[:-1])
    print("Email Class Standard Deviation:", stdEmail[:-1])
    print("Spam Class Mean: ", meanSpam[:-1])
    print("Spam Class Standard Deviation: ", stdSpam[:-1])

    
    # running naive bayes on the test set
    i,j = testFeatures.shape
    j= j-1 # so we skip the labels
    email = [];spam=[];

    # move through the matrix of test features and perform naive bayes
    # this first nested for loop does this for the email training examples
    for a in range(0,i):
        feature_e = testFeatures[a][:-1]
        total = 0
        for b in range(0,j):
            fp = naiveBayes(meanEmail[b], stdEmail[b],feature_e[b])
            if fp == 0:
                fp = r.randint(1,2)
            total = sp.log10(fp) + total
            classChoice_email = sp.log10(emailProbability)+total
        email.append(classChoice_email)
        
    # this nested loop performs naive bayes for the spam training examples
    for a in range(0,i):
        feature_s = testFeatures[a][:-1]
        total = 0
        for b in range(0,j):
            gp = naiveBayes(meanSpam[b], stdSpam[b], feature_s[b])
            if gp == 0:
                fp = r.randint(1,2)
            total = sp.log10(gp) + total
            classChoice_spam = sp.log10(spamProbability)+total
        spam.append(classChoice_spam)

    # calculate the arg max for each feature
    classTest = []
    for each in range(len(spam)):
        if spam[each] > email[each]:
            classTest.append(1)
        elif spam[each] < email[each]:
            classTest.append(0)

    # see how the model performed by calculating the elements of a confusion matrix
    # testFeatures labels are scaled by 1, so testFeature == 2 is a 1 and testFeature ==1 is a 0
    k,l = testFeatures.shape; fp = 0; tp= 0; fn = 0;tn = 0
    for a in range(0,k):
        if testFeatures[a][57] == 2 and classTest[a] == 1:
            tp += 1
        elif testFeatures[a][57] == 1 and classTest[a] == 1:
            fp+=1
        elif testFeatures[a][57] == 2 and classTest[a] == 0:
            fn+=1
        elif testFeatures[a][57] == 1 and classTest[a] == 0:
            tn +=1

    # print the results
    accuracy = (tp+tn)/2300
    precision = tp /(tp+fp)
    recall = tp/(tp+fn)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    print(" TP:", tp, "FP:", fp, "FN:", fp, "TN:", tn)



def naiveBayes(mean,stdDev,feature):
    """this function takes a mean, a standard deviation and a feature.
        it then runs the naive bayes algorithm returning the resultant value"""

    N = (1/((sp.sqrt(2*sp.pi)*float(stdDev))))*sp.exp(- (((float(feature) - float(mean))**2)/(2*(float(stdDev)**2))))

    return N


def probablisticModel(features):
    """ take a matrix of features and calculate the mean and standard deviation of each column.
        return the standard deviation and mean"""

    classMean = sp.mean(features, axis =0)
    classStd = sp.std(features, axis=0)

    return classMean, classStd

def probability(matrix):
    """ take a matrix and calculate the prior probability of each class in the matrix.
         returning count of each class as two lists."""
    positive = 0; negative = 0
    i,j = matrix.shape

    for a in range(0,i):
        if(matrix[a][57] == 1.0):
            negative+= 1
        else: positive +=1

    return positive, negative 
    

def read(file):
    """read a file and turn store in a nxm matrix. Add one to each value to scale the data.
        return a nxm matrix"""

    featVect = []

    with open(file,'r') as f:
        line = f.read().splitlines()

    for each in line:
        splitline = each.split(",")
        for element in splitline:
            featVect.append(float(element)+1)

    column = int(len(featVect)/58)
    matrix = sp.zeros(column*58).reshape(column,58)
    i,j =  matrix.shape

    for a in range(0,i):
        for b in range(0,j):
            matrix[a][b] = float(featVect.pop(0))            

    return matrix
