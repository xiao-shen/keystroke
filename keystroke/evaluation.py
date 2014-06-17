import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class Evaluation():
    """
    This class is used to evaluate the performance of our machine learning tool chain.
    """

    def evaluate(self, c_pred, c_test,feature,classifier):
        """
        This function calculates a whole set of statistics from the classification results.
        It also plots the confusion matrix.
        
        @param c_pred: The class labels of the test set predicted by our classifier
        @param c_test: The acutal (correct) class labels
        @param feature: The used feature set
        @param classifier: The used classifier  
        """
        #creates confusion matrix with scikit-learn which needs class-labels as numbers
        le=LabelEncoder()
        le.fit(c_test)
        c_test = le.transform(c_test)
        c_pred = le.transform(c_pred)
        cm = confusion_matrix(c_test, c_pred)
        print "Results of %s with %s \n\n" % (feature,classifier)
        
        print(classification_report(c_test,c_pred))
        print "\n\n"
        print cm
        print "\n\n"
        
        c_test = le.inverse_transform(c_test)
        c_pred = le.inverse_transform(c_pred)
        
        #prints existing labels
        labels = sorted(set(c_test))
        for k in range(len(labels)):
            labels.insert(2*k,k)
        print "labels: " + str(labels) 
        print "\n"
        
       
        
        # normalize confusion matrix
        norm_conf = []
        for i in cm:
            a=0
            cache=[]
            a=sum(i,0)
            for j in i:
                cache.append(j/float(a))
            norm_conf.append(cache)
        # create figure
        fig = plt.figure()
        cm_plot = fig.add_subplot(111)
        cmconfig = cm_plot.imshow(np.array(norm_conf), cmap=plt.cm.PuBuGn,interpolation='nearest')
        length=len(cm)
        # put number of samples in each field
        for x in xrange(length):
            for y in xrange(length):
                cm_plot.annotate(str(cm[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center')
        
        fig.colorbar(cmconfig)
        # describe axes, rows and columns with labels
        xlabel= sorted(set(c_test))
        ylabel= sorted(set(c_test))
        for i in xrange(np.size(xlabel)):
            xlabel[i]=xlabel[i][:3]
        plt.xticks(range(length),xlabel[:length])
        plt.yticks(range(length),ylabel[:length])
        plt.title('Confusion matrix of %s with %s' % (feature,classifier))
        plt.xlabel('predicted class')
        plt.ylabel('true class')
        plt.show()
        
        # plot with pylab
        #pl.matshow(cm)
        #pl.title('Confusion matrix of %s with %s' % (feature,classifier))
        #pl.xlabel('predicted class')
        #pl.colorbar()
        #pl.ylabel('true class')
        #pl.show()
