from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy
from numpy import vstack
from numpy import array
from sklearn import svm

#Some possible helper methods
import helper

helper.statement("Model", 0)

# Load the Model
model = Doc2Vec.load("d2v.model")

helper.statement("Model\n", 1)

#Load CSV Files
dummyData = pd.read_csv("./data/P1_Dummy.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
labeledData = pd.read_csv("./data/P1_Labelled_dataset.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
testingData = pd.read_csv("./data/P1_testing_set.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
trainingData = pd.read_csv("./data/P1_training_set.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])

# Constants for data to be run
TRAINING_DATA = trainingData
TESTING_DATA = testingData

# Lists to run through training
listOfData = []
listOfLabels = []

# Word Vectors ------------------------------------------------------------------------------------
helper.statement("Word Vectors", 0)

#Loop through the data and generate the word vectors
for index, row in TRAINING_DATA.iterrows():
    # Run Doc2Vec's NLP on the sentence under E1 and E2
    sent1 = word_tokenize(str(row['E1']))
    sent2 = word_tokenize(str(row['E2']))

    vector1 = model.infer_vector(sent1)
    vector2 = model.infer_vector(sent2)
    # Get the two word vectors and concatenate them
    # nlp.vector on the string will return the average word vector for the sentences in an 1D Numpy array
    # from there, we use concatenate from numpy to combine into a single vector
    combination = numpy.array([])
    combination = numpy.concatenate([vector1, vector2])

    #print(combination.shape)
    # Append to a list of all the data points
    listOfData.append(combination)

    # Append the label to a list of labels
    listOfLabels.append(row['Label'])

helper.statement("Word Vectors\n", 1)

# Training ----------------------------------------------------------------------------------------
helper.statement("Training Data", 0)

# Based on scikit-learn site: https://scikit-learn.org/stable/modules/svm.html#classification
clf = svm.SVC(gamma='scale')

# Train and fit the data
clf.fit(listOfData, listOfLabels)

helper.statement("Training Data\n", 1)

# Test case using only a single entry
# testList = helper.testData()

#Counters for correct and total 
counter = 0
trueCounter = 0

helper.statement("Testing Data", 0)

# Run agaisnt testing data
for index, row in TESTING_DATA.iterrows():

    # Grab the sentences
    sent1 = word_tokenize(str(row['E1']))
    sent2 = word_tokenize(str(row['E2']))

    vector1 = model.infer_vector(sent1)
    vector2 = model.infer_vector(sent2)
    # Build the vectors and store them in a numpy array (default)
    # then convert to a normal array
    testString = []
    combineTests = numpy.array([])
    combineTests = numpy.concatenate([vector1, vector2])
    testString.append(combineTests.tolist())

    #print(row['Label'], "-", clf.predict(testString)[0])

    # Run the prediction, If the prediction was accurate, then add to the counter
    if (str(row['Label']) == str(clf.predict(testString)[0])):
        trueCounter += 1
    #print()

    # Add to the counter
    counter += 1

helper.statement("Testing Data\n", 1)

# The Results printing
print("Results:\n")
print("Accurate Predictions:", trueCounter)
print("Total Predictions:", counter)
print("Percentage Correct", round((trueCounter/counter), 2))