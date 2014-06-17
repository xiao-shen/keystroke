import os
from scipy.io import wavfile
import numpy

"""
This module manages the test sets. There should be two test sets. One for 
training purpose and one for testing. 

Typically the train set is located in the root folder train whereas
the test set is located in the folder test
"""

def getClassLabel(file):
    """
    This method extracts the class label from a given file name or file path. 
    
    The class label should either be the file name without ending numbers
    or the folder in which the file is located
    
    @param file: The file name or the file path 
    @returns the class label as string
	"""
    return file.split('.')[0].strip("0123456789")

def getDataSamples(path):
    """
    Returns the available test or train data in a given root directory.
    The directory structure should either be:
    <root>/<class>/<file1... n>.wav
    <root>/<class><nr1...n>.wav
    @param path: The root path to the directory structure
    @returns a dictionary. The key of the dictionary is the actual class 
        and the value of the dictionary is a list of files in that class
    """
    testData = [(getClassLabel(f),path+"/"+f) for f in os.listdir(path)]

    testDict = {}
    for key,val in testData:
        if not key in testDict:
            testDict[key] = []
        testDict[key].append(val)
        
    return testDict
    
def getTestData(path="test"):
    return getDataSamples(path)

def getTrainData(path="train"):
    return getDataSamples(path)
    
class DataSet:
    """
    This class holds all features and their corresponding classes in a so called 
    feature vector. The feature vector is a 2d array with
        - features for each instance in the x dimension
        - instances in the y dimension
    """
    def __init__(self, featureExtractor):
        self.featureExtractor = featureExtractor
        self.featureVector = None
        self.classes = []
    
    def storeFeatures(self, featureArray, feature_class):
        if self.featureVector == None:
            self.featureVector = featureArray
        else:
            self.featureVector = numpy.concatenate((self.featureVector, featureArray))
        
            
        self.classes.extend([feature_class]*featureArray.shape[0])
    
    def addFile(self,feature_class, file):
        """
        Add a new audio file to this dataset. This class extracts features
        from the audio files and stores them in the feature vector.
        
        @param feature_class: The class of the audio file(e.g. walking, wc, toothbrush)
        @param file: The audio file
        """
    
        sampleRate, data = wavfile.read(file)
        
        if len(data.shape) != 1:
            print "File {0} is not mono!".format(file)
            data = data[:,0]
        
        features = self.featureExtractor(data)
        self.storeFeatures(features, feature_class)
