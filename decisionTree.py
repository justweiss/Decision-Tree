"""
    Created by Justin Weiss on 3/1/20.
    Copyright © 2020 Justin Weiss. All rights reserved.Justin Weiss


    DECISION TREE LEARNING Algorithm
    This program implements the decision tree learning algorithm from Artificial Intelligence, A Modern Approach
        chapter 18.

    NOTE: This program requires python3, pandas, numpy, and pprint, without these libraries the program will not run.

    RUN: python3 decisionTree.py
        It will then ask you a set of questions, it will give you a set of options to the right. Based off the questions
            it will tell you to wait or leave
        Example: Patrons?  ['Full', 'None', 'Some']
            You would enter Full or full (the first letter is not case sensitive)
"""

from pprint import pprint
import pandas as pd
import numpy as np
import math

# This function takes in the target (WillWait) column and returns the entropy of that column
def entropyTarget (example):
    count = 0
    split = [0, 0, 0]

    # Loops through all the yes/no out comes and addes them up
    for x in example:
        if x == "Yes":
            split[0] += 1
        else:
            split[1] += 1
        count += 1

    # calls entropy and returns value
    return entropy(split[0], split[1])

# Calculates the entropy, takes in the number of yes and no, returns entropy value
def entropy (yes, no):
    # finds total yes and no
    total = yes + no

    # if yes or no are 0 then the entropy is 0
    if (yes == 0) or (no == 0):
        return 0

    # else calculate the entropy, using ID3
    else:
        return -(yes / total) * math.log((yes / total), 2) - (no / total) * math.log((no / total), 2)

# This function calculates the importance gain, it takes in the data, the feature to calculate importance on, and the
#   the target column
def importance (data, feature, target):
    # creates a local dictionary to hold the values
    split = {}
    count = 0
    index = data.index.values

    # Loops through the data for that type of feature to calculate entropy
    for x in data[feature]:

        if x not in split:
            split[x] = [0, 0, 0]
        if data[target][index[count]] == "Yes":
            split[x][0] = split[x][0] + 1
        else:
            split[x][1] = split[x][1] + 1
        count += 1

    # Finds the weighted entropy for each option under feature
    weightedEntropy = 0
    for x in split:
        # gets entropy for the elements in split
        split[x][2] = entropy(split[x][0], split[x][1])
        subTotal = split[x][0] + split[x][1]
        # weights the entropy for each item with a percent of total amount
        weightedEntropy += split[x][2]*(subTotal/count)

    # Finds target entropy
    targetEntropy = entropyTarget(data[data.columns[-1]].tolist())

    # Returns gain
    return targetEntropy-weightedEntropy

"""
This is the main function that creates the tree, it takes in 5 paramters. data = is the input data but it gets smaller
everytime something is removed. originaldata = is the input data, it is ever edited or reduced. attributes = is the column
headings, it is used to help remove attributes from data, targetAttribute = is the target output column. parent_examples =
this is the last attribute removed
"""
def decisionTree(data, originaldata, attributes, targetAttribute="WillWait", parent_examples=None):

    #  if all examples have the same classification then return the classification
    if len(np.unique(data[targetAttribute])) <= 1:
        return np.unique(data[targetAttribute])[0]

    # if data is empty then return the most common output value among a set of examples
    elif len(data) == 0:
        return np.unique(originaldata[targetAttribute])[
            np.argmax(np.unique(originaldata[targetAttribute], return_counts=True)[1])]

    # if attributes is empty then return most common output value among a set of examples
    elif len(attributes) == 0:
        return parent_examples

    # else, grow the tree!
    else:
        # Set the default value for this node
        parent_examples = np.unique(data[targetAttribute])[np.argmax(np.unique(data[targetAttribute], return_counts=True)[1])]

        # Loops through the attributes calculating info gain
        gainValues = [importance(data, feature, targetAttribute) for feature in attributes]

        # Finds the highest info gain
        bestFeatureIndex = np.argmax(gainValues)
        best_feature = attributes[bestFeatureIndex]

        # Create the tree structure
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain
        attributes = [i for i in attributes if i != best_feature]

        # Grow a branch for each possible value of the root feature
        for value in np.unique(data[best_feature]):
            value = value

            # creates a smaller data set with the features of the largest information gain
            sub_data = data.where(data[best_feature] == value).dropna()

            # Calls decisionTree to create a subtree
            subtree = decisionTree(sub_data, originaldata, attributes, targetAttribute, parent_examples)

            # add a branch to tree with label best_feature and subtree subtree
            tree[best_feature][value] = subtree

        return (tree)

# This function traverses through the tree asking the user questions
def treeTraversal (tree):
    # loops until an answer is reached
    while True:

        # finds first location in tree
        current = list(tree.keys())

        # Prints the question and the options, then reads in user input
        print("%s?  %s" % (current[0], list(tree[current[0]].keys())))
        user = str(input()).title()

        # if an answer is reached output answer
        if (tree[current[0]][user] == 'Yes'):
            print("Yes, you will wait")
            return 0
        elif (tree[current[0]][user] == 'No'):
            print("No, you will not wait")
            return 0

        # else set tree to new current place and repeat
        else:
            tree = tree[current[0]][user]



def main():
    # Read in data and clean whitespace char
    dataFile = pd.read_csv('restaurant.csv', header=None)
    dataFile.columns = ["Alt", "Bar", "Friday", "Hungry", "Patrons", "Price", "Rain", "Res", "Type", "Est", "WillWait"]
    dataFile = dataFile.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Creates and prints the tree
    tree = decisionTree(dataFile, dataFile, dataFile.columns[:-1])

    # Prints tree nicely
    pprint(tree)

    # Traverses tree
    treeTraversal(tree)


main()
