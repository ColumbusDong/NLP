import pandas as pd
import numpy
from numpy import vstack
from numpy import array
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Based on Tutorial from https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5

#Load CSV Files
labeledData = pd.read_csv("./data/P1_Labelled_dataset.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])

# Generate the training data with a list of strings/sentences
testStrings = []
for index, row in labeledData.iterrows():
    sent1 = str(row['E1'])
    sent2 = str(row['E2'])


    testStrings.append(sent1)
    testStrings.append(sent2)

# Tag the data
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(testStrings)]

#Train the Model
max_epochs = 100
vec_size = 20
alpha = 0.025

# dm = 1 refers to distributed memory (PV-DM), while dm = 0 means distrubted bag of words (PV-DBOW)
# PV-DM preserves word order
model = Doc2Vec(size = vec_size, alpha = alpha, min_alpha = .00025, min_count = 1, dm = 1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    # Decrease the learning rate on each iteration
    model.alpha -= .0002
    # Fix the learn rate with no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model has been saved")