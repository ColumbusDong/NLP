# Import NLP Library
import spacy
import pandas as pd

#Load NLP models
nlp = spacy.load('en_core_web_sm')

#Load CSV Files
dummyData = pd.read_csv("./data/P1_Dummy.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
labeledData = pd.read_csv("./data/P1_Labelled_dataset.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
testingData = pd.read_csv("./data/P1_testing_set.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])
trainingData = pd.read_csv("./data/P1_training_set.csv", sep = ",", header = 0, names = ["EID", "E1", "E2", "Label"])

for index, row in dummyData.iterrows():
    doc = nlp(row['E1'])

    for token in doc:
        print(token.text, end = "+")
    print("")