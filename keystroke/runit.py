# -*- coding: utf-8 -*-

from classifier import Classifier
from dataset import getTestData, DataSet
from extract import FeatureExtractor
from evaluation import Evaluation
import argparse
import numpy
from sklearn.cross_validation import train_test_split
import os
import sys
import time



def print_class_stats(classes):
    keys = set(classes)
    for k in keys:
        print "%s: %s" % (k,sum(1 for s in classes if s == k))  

def eval_classifier(classifierToUse, featuresToUse, testOrTrain="train"):

    print("Chosen feature: {0}".format(featuresToUse) )
    print("Chosen classifier: {0}".format(classifierToUse))

    fe = FeatureExtractor(featuresToUse)
    dataset = DataSet(fe)
    classifier = Classifier()
    evaluate = Evaluation()

    print "test or Train %s" % testOrTrain
    for feature_class, files in getTestData(testOrTrain).items():
        print "%s" % testOrTrain
        for f in files:
            dataset.addFile(feature_class, f)

    print "Dataset initialized"
    print_class_stats(dataset.classes)

    print "Test set created."
    a_train, a_test, c_train, c_test = train_test_split(dataset.featureVector, dataset.classes, test_size=0.9)
    
    c_pred = classifier.classification(a_train,a_test,c_train,c_test,classifierToUse)
    
    evaluate.evaluate(c_pred,c_test,featuresToUse,classifierToUse)
    
def fixPathOnWindowsAndMacOs():
    if os.name=="nt":
        workdir= os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(workdir)
    if os.name=="posix":
        workdir= os.path.dirname(os.path.abspath(sys.argv[0]))
        os.chdir(workdir)

if __name__=="__main__":
    fixPathOnWindowsAndMacOs()
    parser = argparse.ArgumentParser(description="Keystroke-Detctor script to predict the key press event which caused a certain sound")
    parser.add_argument("-c", "--classifier", default="KNN", choices=["GNB", "DT", "KNN", "RF", "SVC", "DC"], help="The Classification Algorithm to use")
    parser.add_argument("-f", "--features", default="MFCC", choices=["MFCC", "DMFCC","PS", "ZCR", "BE"], help="The features to extract from the audio input files")

    args = parser.parse_args()

    start = time.clock()
    print time.ctime()
    eval_classifier(args.classifier, args.features)
    end = time.clock()
    print "eval_classifier() needs %1.2f seconds to calculate" % (end-start)

