from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.mixture import GMM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Classifier:

    def classification(self,a_train,a_test,c_train,c_test,classifier):
        """ 
        All classifier are fitted with matrix a_train. a_train contains feature values and c_train contains
        corresponding classes.
        
        Then, in a second step, this fitted algorithm is used to predict the classes for the test set a_test. This 
        results are stored in c_pred.
        
        LabelEncoder encodes the symbolic class names to numbers 1,2,3... because of compatibility problems
        
        @param a_train: The feature vectors to fit the classifier
        @param a_test: The feature vectors to test the fittet classifier
        @param c_train: The actual classes for the training set
        @param c_test: The actual classes for the test set
        """
        le =LabelEncoder()
        le.fit(c_train)
        c_train = le.transform(c_train)
        c_test = le.transform(c_test)
        if classifier=="GNB": #Gaussian Naive Bayes
            gnb = GaussianNB()
            gnb.fit(a_train, c_train)
            c_pred = gnb.predict(a_test)
        elif classifier=="DT": #Decision Tree
            dt=DecisionTreeClassifier()
            dt.fit(a_train, c_train)
            c_pred = dt.predict(a_test)
        elif classifier=="KNN": #K-Next-Neighbors
            kn=KNeighborsClassifier(n_neighbors=5)
            kn.fit(a_train, c_train)
            c_pred = kn.predict(a_test)
        elif classifier=="RF": #Random Forest
            rf=RandomForestClassifier()
            rf.fit(a_train, c_train)
            c_pred = rf.predict(a_test)
        elif classifier=="SVC": # Support Vector Classifier
            """
            SVC needs normalisation of Feature Values to scale of [-1,1] or [0,1] depending on sign of them
            """
            if a_train.min()<0:
                mms = MinMaxScaler(feature_range=(-1,1))
            else:
                mms = MinMaxScaler()
            mms.fit(a_train)
            a_train = mms.transform(a_train)
            a_test = mms.transform(a_test)
            svc=SVC(cache_size=2000,C=1, probability=True,kernel='rbf')
            svc.fit(a_train,c_train)
            #c_pred = svc.predict(a_test) did not work, that's why it is predicted manual
            new_prob = svc.predict_proba(a_test)
            samples=new_prob.shape[0]
            c_pred= np.array
            for k in range(samples):
                c_pred=np.append(c_pred,new_prob[k].argmax())
            c_pred = c_pred[1:samples+1]
        elif classifier=="DC": #Dummy Classifier
            dc=DummyClassifier(strategy="uniform")
            dc.fit(a_train, c_train)
            c_pred = dc.predict(a_test)
        elif classifier=="GMM": #Gaussian Mixture Modell
            #number of existing classes get passed to the GMM (n_classes)
            n_classes_train = len(np.unique(c_train))
            n_classes_test = len(np.unique(c_test))
            if n_classes_train>n_classes_test:
                n_classes = n_classes_train
            else:
                n_classes = n_classes_test
            #init_params='', because initial values get calculated manual
            gmm = GMM(n_components=n_classes,init_params='')
            #array of feature values of class i get extracted for further process
            gmm.means_=np.array([a_train[c_train==i,:].mean(axis=0) for i in xrange(n_classes)])
            gmm.weights_=np.array([a_train[c_train==i,:].shape[0]/float(c_train.shape[0]) for i in xrange(n_classes)])
            
            gmm_covars = np.zeros((a_train.shape[1]))
            for i in xrange(n_classes):
                valuesOfClassi = a_train[c_train==i,:]
                valuesOfClassi = np.asarray(valuesOfClassi).T
                matrixOfCov = np.cov(valuesOfClassi)+gmm.min_covar*np.eye(valuesOfClassi.shape[0])
                variance = np.array([matrixOfCov[j,j] for j in xrange(matrixOfCov.shape[0])])
                gmm_covars=np.vstack((gmm_covars,variance))
            gmm_covars=gmm_covars[1:,:] #deletes initial row with zeros
            
            gmm.covars_=gmm_covars
            c_pred = gmm.predict(a_test)
        
        c_pred=le.inverse_transform(c_pred)
        return c_pred
