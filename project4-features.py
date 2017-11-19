# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.
    Args:
        datum: 2-dimensional numpy.array representing a single image.
    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.
    ## DESCRIBE YOUR ENHANCED FEATURES HERE...
        add an extra feature that shows the number of continuous white regions in the graph.
    ##
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1

    row = datum.shape[0]
    col = datum.shape[1]

    count = 0
    visited = [[False for y in range(col)] for x in range(row)]
    r_itr =0
    c_itr =0



    ti=0
    tj=0
    flg=0

    check = True
    while ti < row:
        if flg==1:
            break
        while tj < col:
            if features[r_itr][c_itr] == 0 and visited[r_itr][c_itr]==False:
                visited[r_itr] [c_itr] = True
                flg =1
                break
            tj = tj+1
        ti = ti+1

    if flg==0:
        check = False


    traverse_list = []
    while check==True:
        traverse_list.append((r_itr, c_itr))
        count += 1

        while traverse_list:
            x, y = traverse_list[0]
            traverse_list.remove((x,y))
            if edge1(y,features,x,visited):
                visited[x] [y - 1] = True
                traverse_list.append((x, y - 1))

            if edge2(y,features,x,visited,col):
                visited[x] [y + 1] = True
                traverse_list.append((x, y + 1))

            if edge3(y,features,x,visited,row):
                visited[x+1][ y ] = True
                traverse_list.append((x + 1, y))

            if edge4(y,features,x,visited):
                visited[x-1][ y ] = True
                traverse_list.append((x - 1, y))


        flg1=0
        i=0
        j=0
        for i in range(row):
            for j in range(col):
                if features[i][j] == 0 and visited[i][j] == False:
                    visited[i][j] = True
                    r_itr = i
                    c_itr=j
                    flg1 = 1
                    break

        if flg1 == 0:
            check = False


    extra_features = helper(count)
    return np.concatenate((features.flatten(), extra_features), axis = 0)

def edge1(y,features,x,visited):
    return True if y > 0 and visited[x][y - 1]== False and features[x][y - 1] == 0 else False

def edge2(y,features,x,visited,col):
    return True if y + 1 < col and visited[x] [y + 1]== False and features[x][y + 1] == 0 else False

def edge3(y,features,x,visited,row):
    return True if x + 1 < row and visited[x+1][ y ] == False and features[x + 1][y] == 0  else False

def edge4(y,features,x,visited):
    return True if x > 0 and visited[x-1][ y ] ==False and features[x - 1][y] == 0  else False


def helper(count):
    if count == 1:
        return np.array([1, 0, 0])
    elif count == 2:
        return np.array([0, 1, 0])
    elif count > 2:
        return np.array([0, 0, 1])



def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
