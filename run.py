from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import backend as K
from time import time
from keras.models import load_model
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
import csv
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
import pprint
from livelossplot import PlotLossesKeras 
from keras.utils import np_utils
#class Data():
np.set_printoptions(threshold=np.nan)

#%matplotlib inline

class Classifier():

	def __init__(self, Data=None, Labels=None, test_size=0.30):
		self.Data = Data
		self.Labels = Labels
		self.test_size = test_size
	
	def preProcessData(self):
		return preprocessing.scale(self.Data)
	
	def splitData(self):
		x_train, x_test, y_train, y_test = train_test_split(self.Data,
		                                                  self.Labels,
		                                                  test_size=self.test_size,
		                                                  random_state=42)
		return x_train, x_test, y_train, y_test

	def bayes(self, x_train, x_test, y_train, y_test):
		print("Bayes Classifier: "),
		gnb = GaussianNB()
		model = gnb.fit(x_train, y_train)
		y_pred = gnb.predict(x_test)
		y_pred_train = gnb.predict(x_train)
		print("train:")
		print(perf_measure(y_train, y_pred_train))
		print("test:")
		print(perf_measure(y_test, y_pred))
		print("train:")
		print(accuracy_score(y_train, y_pred_train))
		print("test:")
		print(accuracy_score(y_test, y_pred))


	def logesticReg(self, x_train, x_test, y_train, y_test):
		print("Logestic Regression: "),
		logisticRegr = LogisticRegression()
		logisticRegr.fit(x_train, y_train)
		y_pred = logisticRegr.predict(x_test)
		y_pred_train=logisticRegr.predict(x_train)
		score = logisticRegr.score(x_test, y_test)
		score_train=logisticRegr.score(x_train, y_train)
		print(score)
		print("train:")
		print(score_train) 
		print("train:")
		print(perf_measure(y_pred_train, y_train))
		print("test:")
		print(perf_measure(y_pred, y_test))

	def decisionTree(self, x_train, x_test, y_train, y_test):
		print("Decision Tree: "),
		clf = tree.DecisionTreeClassifier()
		clf.fit(X=x_train, y=y_train)
		clf.feature_importances_
		y_pred = clf.predict(x_test)
		y_pred_train=clf.predict(x_train)
		acc = clf.score(X=x_test, y=y_test)
		acc_train=clf.score(X=x_train,y=y_train)
		print("train:")
		print(acc_train)
		print("test:")
		print(acc)
		print("train:")
		print(perf_measure(y_pred_train, y_train))
		print("test:")
		print(perf_measure(y_pred, y_test))

	def svmRadial(self, x_train, x_test, y_train, y_test):
		print("SVM Radial: "),
		svclassifier = SVC(kernel='rbf')  
		svclassifier.fit(x_train, y_train)  
		y_pred = svclassifier.predict(x_test) 
		y_pred_train = svclassifier.predict(x_train) 
		print(svclassifier.score(x_test, y_test))
		print("train:")
		print(svclassifier.score(x_train, y_train))
		print("train:")
		print(perf_measure(y_pred_train, y_train))
		print("test:")
		print(perf_measure(y_pred, y_test))

	def svmLinear(self, x_train, x_test, y_train, y_test):
		print("SVM Linar: "),
		svclassifier = SVC(kernel='linear')  
		svclassifier.fit(x_train, y_train)  
		y_pred = svclassifier.predict(x_test) 
		y_pred_train = svclassifier.predict(x_train) 
		print(svclassifier.score(x_test, y_test))
		print("train:")
		print(svclassifier.score(x_train, y_train))
		print("train:")
		print(perf_measure(y_pred_train, y_train))
		print("test:")
		print(perf_measure(y_pred, y_test))

	def classify(self):
		x_train, x_test, y_train, y_test = self.splitData()
		self.bayes(x_train, x_test, y_train, y_test)
		self.logesticReg(x_train, x_test, y_train, y_test)
		self.decisionTree(x_train, x_test, y_train, y_test)
		self.svmRadial(x_train, x_test, y_train, y_test)
		self.svmLinear(x_train, x_test, y_train, y_test)
        
def pretty_print_linear(coefs, names = None, sort = False):
	if names == None:
		names = ["X%s" % x for x in range(len(coefs))]
	lst = zip(coefs, names)
	if sort:
		lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
	return " + ".join("%s * %s" % (round(coef, 3), name)
		                   for coef, name in lst)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

class Autoencoder:
	def __init__(self, isRelational=0, alpha=0.5, epochs=50, batch_size=8):
		self.isRelational = isRelational
		self.alpha = alpha
		self.model = None
		self.history = None
		self.epochs = epochs
		self.batch_size = batch_size

	def custom_loss(self, y_true, y_pred):
		if self.ifRelational==1 :
			reconstruction_loss = mse(y_true, y_pred)
			reconstruction_loss *= 128*128
			reconstruction_loss *= self.alpha
			relation_loss = mse(K.dot(K.transpose(y_true), y_true),K.dot(K.transpose(y_pred), y_pred))
			relation_loss *= 128*128
			relation_loss *= (1-self.alpha)
			auto_loss = reconstruction_loss + relation_loss
			return reconstruction_loss
		else:
			return binary_crossentropy(y_true, y_pred)

	def non_shuffling_train_test_split(self, X, y, test_size=0.3):
		i = int((1 - test_size) * X.shape[0]) + 1
		X_train, X_test = np.split(X, [i])
		y_train, y_test = np.split(y, [i])
		return X_train, X_test, y_train, y_test

	def fitmodel(self, x_train, x_test, y_train, y_test): 
		'''self.history = self.model.fit(x_train, [x_train, np_utils.to_categorical(y_train)],
			epochs=self.epochs,
			batch_size=self.batch_size,
			shuffle=True,
			validation_data=(x_test, [x_test, np_utils.to_categorical(y_test)]),
			callbacks=[PlotLossesKeras()])'''
		self.history = self.model.fit(x_train, x_train,
			epochs=self.epochs,
			batch_size=self.batch_size,
			shuffle=True,
			validation_data=(x_test, x_test),
			callbacks=[PlotLossesKeras()])

	def saveWeight(self, modelName):
		if self.isRelational == 1:
			modelName = modelName + 'Relational'
		self.model.save_weights(modelName + "h5")
        
class DenseAuto(Autoencoder):
	def createModel(self):
		input = Input(shape=(15075,), name='input')
		reduced = Dropout(0.8)(input)
		encoded = Dense(8024, activation='relu', name='encoded')(input)
		reduced = Dropout(0.8)(encoded)
		encoded = Dense(1024, activation='relu')(encoded)
		reduced = Dropout(0.8)(encoded)
		reduced = Dense(9,  activation='relu', name='reduced')(reduced)
		reduced = Dropout(0.8)(reduced)
		encoded = Dense(1024, activation='relu', name='encoded2')(reduced)
		reduced = Dropout(0.8)(encoded)
		encoded = Dense(8024, activation='relu')(encoded)
		reduced = Dropout(0.8)(encoded)
		decoded = Dense(15075, activation='softmax', name='decoded')(reduced)
		reduced = Dropout(0.8)(reduced)
		classifier = Dense(3, activation='softmax', name='classifier')(reduced)

		self.model = Model(input, [decoded, classifier])
		self.model.summary()
		plot_model(self.model, to_file='Autoencoder.png', show_shapes=True)

		opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		if self.isRelational == 0:
			self.model.compile(optimizer=opt, loss={'decoded':'binary_crossentropy', 'classifier':'categorical_crossentropy'}, loss_weights=[10, 1], metrics=['accuracy'])
		else:
			self.model.compile(optimizer=opt, loss={'decoded':self.custom_loss, 'classifier':'categorical_crossentropy'}, loss_weights=[10, 1], metrics=['accuracy'])

class CNNAuto(Autoencoder):

	def createModel(self):
		input = Input(shape=(128*128, 1), name='input')
       #reduced = Dropout(0.8)(input)
		x = Conv1D(16, 3, activation='relu', padding='same')(input)
		#encoded = Dense(1024, activation='relu', name='encoded2')(reduced)
		maxPool = MaxPooling1D()(x)
		x = Conv1D(1, 3, activation='relu', padding='same')(input)
		flat = Flatten()(maxPool)
		encoded = Dense(1024, activation='relu', name='encoded1')(flat)
		hidden = Dense(9, activation='relu', name='hiddens')(encoded)
		encoded = Dense(1024, activation='relu', name='encoded2')(hidden)
		reshape =Reshape((1024, 1))(encoded)
		#reduced = Dropout(0.8)(reduced)
		upSample = UpSampling1D(16)(reshape)
		x = Conv1D(16, 3, activation='relu', padding='same')(upSample)
		decoded = Conv1D(1, 3, activation='softmax', padding='same', name='decoded')(x)

		#classifier = Dense(3, activation='softmax', name='classifier')(hidden)

		self.model = Model(input, decoded)
		self.model.summary()
		plot_model(self.model, to_file='Autoencoder.png', show_shapes=True)

		opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		if self.isRelational == 0:
			#self.model.compile(optimizer=opt, loss={'decoded':'binary_crossentropy', 'classifier':'categorical_crossentropy'}, loss_weights=[10, 1], metrics=['accuracy'])
			self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
		else:
			self.model.compile(optimizer=opt, loss={'decoded':self.custom_loss, 'classifier':'categorical_crossentropy'}, loss_weights=[10, 1], metrics=['loss'])


																																																																																																																																								

class Reduction():

	def __init__(self, startRow=1, endRow=420, autoencoder=0,  exFeatureStart=-6, exFeatureEnd=-2, ispreprocess=1, modelType='dense', path=None, typeofclass=None, lassoC=0.0027, principleCompo=11):
		self.startRow = startRow
		self.endRow = endRow
		self.lassoC = lassoC
		self.path = path
		self.principleCompo = principleCompo
		self.exFeatureStart = exFeatureStart
		self.exFeatureEnd = exFeatureEnd
		self.typeofclass = typeofclass
		self.ispreprocess = ispreprocess
		self.autoencoder = autoencoder
		self.modelType = modelType
																																

	def splitExtraFeature(self, x):
		return  x[0:, 1: self.exFeatureStart], x[0:, self.exFeatureStart: self.exFeatureEnd]

	def scaleData(self, x):
		scaler = StandardScaler()
		scaler.fit(x_train)
		return scaler.transform(x)

	def loadData(self):
		print("Loading Processed Data")
		reader = csv.reader(open("modifiedfile"+str(self.typeofclass)+"class"+".csv", "r"), delimiter=",")
		data = list(reader)
		print("Loaded")
		#return np.array(data[self.startRow: self.endRow]).astype(np.float)    #read data in string form
		npdata = np.array(data[self.startRow: self.endRow])
		return npdata[:, :-1].astype(np.float)

	def preprocess(self):
		reader = csv.reader(open(self.path+"abc"+str(self.typeofclass)+"classwoenrolid.csv", "r"), delimiter=",")
		print("Data Loaded")
		data = list(reader)
		print("Processing Data...")
		result = np.array(data[self.startRow: self.endRow]).astype("str")    #read data in string form
		for j in range(0, len(result)):
			result[j][0] = '0'
			for i in range(0, len(result[j])):
				if(result[j][i] == "butrans"):
					result[j][i] = '0'
				if(result[j][i] == "opana"):
					result[j][i] = '1'
				if(result[j][i] == ''):
					result[j][i] = '0'
				if(result[j][i] == "Butrans and Opana"):
					result[j][i] = '2' 
				if(result[j][i] == 'Frequent'):
					result[j][i] = '0' 
				if(result[j][i] == 'Non Frequent'):
					result[j][i] = '1'
		print("Done Processing")
		df = pd.DataFrame(result)
		df.to_csv("HighDimDataClass"+str(self.typeofclass)+".csv")
		print("Data Saved")

	def joinFeatures(self, x, x_extra):
		return np.concatenate((x, x_extra), axis=1)
	
	def finalData(self):
		if self.ispreprocess == 1:
			self.preprocess()
		x = self.loadData()
		labels = x[self.startRow-1:, -1]   
		x, x_extra = self.splitExtraFeature(x)
		if self.modelType == 'CNN':
			x_train = np.empty((len(x), 128*128, 1))            
			for j in range(0, len(x)):                         #reshaping to 2d array for convolutional autoencoder
				x_train[j] = np.reshape(np.pad(x[j], (0, 1311), 'constant'),(128*128,1))
			x = x_train
		return x, x_extra, labels

	def skPCA(self, x):
		pca =PCA(n_components=self.principleCompo)
		pca.fit(x)
		print("PCA Components: ")
		#print(pca.components_)
		print("PCA Variance: ")
		print(pca.explained_variance_ratio_)
		print("PCA singular values: ")
		print(pca.singular_values_)
		X = x - np.mean(x, axis=0)
		cov_matrix = np.dot(X.T, X) / x.shape[0]
		print("Projected Data")
		proj = pca.transform(x)
		#print(proj)
		print("Eigen Vector and Eigen Values")
		eigenvalues = pca.explained_variance_
		for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):  
			print("eigenvector & eignevalue")
			print(eigenvector.shape)  
			print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector))),
			print(eigenvalue)
		return proj
	
	def Lasso(self, x, y):
		print("Lasso Regression :")
		lsvc = LinearSVC(C=self.lassoC, penalty="l1", dual=False).fit(x, y)
		print("Lasso coeff")
		print((lsvc.coef_))
		#print(pretty_print_linear(lsvc.coef_))
		print(lsvc.coef_.shape)
		model = SelectFromModel(lsvc, prefit=True)
		X_new = model.transform(x)
		print(X_new.shape)
		print("Reduced")
		return X_new

	def npPCA(self, x):
		M = mean(x.T, axis=1)
		print("Mean: ")
		print(M)
		Cov = x - M
		print("Center Column: ")
		print(Cov)
		V = cov(Cov.T)
		print("Convariance Matrix: ")
		print(V)
		values, vectors = eig(V)
		print("Eigen Vectors: ")
		print(vectors)
		print("Eigen Values: ")
		print(values)
		P = vectors.T.dot(Cov.T)
		print("Projected Data:")
		print(P.T)
		return P

	def ridge(self, x, labels, alpha):
		X, y = x, labels
		clf = Ridge(alpha)
		clf.fit(X, y) 
		#print(pretty_print_linear(clf.coef_))
		model = SelectFromModel(clf, prefit=True)
		X_new = model.transform(X)
		print(X_new.shape)
		return X_new

	def reduce(self):
		if self.autoencoder == 0:
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print(str(self.typeofclass) +" Classification")
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")																	
			x, x_extra, labels = self.finalData()
			print("Staring to Reduce Dimension:")
			print("Lasso: ")
			print("==============================================================")
			lassoxReduced = self.Lasso(x, labels)
			lassofinalX = self.joinFeatures(lassoxReduced, x_extra)
			print("Dimension Reduced")
			print("Classification")
			classifier = Classifier(lassofinalX, labels)
			classifier.classify()
			print("==============================================================")
			print(" ")

			print("Pca using sklearn")
			print("==============================================================")
			pcaxReduced = self.skPCA(x)			
			pcafinalX = self.joinFeatures(pcaxReduced, x_extra)
			print("Dimension Reduced")
			print("Classification")
			classifier = Classifier(pcafinalX, labels)
			classifier.classify()
			print("==============================================================")
			print(" ")
			"""print("Pca using numpy")
			print("==============================================================")
			nppcaxReduced = self.npPCA(x)
			nppcafinalX = self.joinFeatures(pcaxReduced, x_extra)
			print("Dimension Reduced")
			print("Classification")
			classifier = Classifier(nppcafinalX, labels)
			classifier.classify()
			print("==============================================================")
			print(" ")"""
			print("Ridge alpha=lasso")
			print("==============================================================")
			ridhexReduced = self.ridge(x, labels, self.lassoC)
			ridgefinalX = self.joinFeatures(ridhexReduced, x_extra)
			print("Dimension Reduced")
			print("Classification")												
			classifier = Classifier(ridgefinalX, labels)
			classifier.classify()
			print("==============================================================")
			"""print(" ")
			print("Ridge alpha=0")
			print("==============================================================")
			ridhexReduced = self.npPCA(x)
			ridgefinalX = self.joinFeatures(ridhexReduced, x_extra, 0)
			print("Dimension Reduced")
			print("Classification")
			classifier = Classifier(ridgefinalX, labels)
			classifier.classify()
			print("==============================================================")"""
		else:
			if self.modelType == 'CNN':
				auto = CNNAuto(isRelational=0, epochs=6)
			else:
				auto = DenseAuto(isRelational=0)
			auto.createModel()
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			print(str(self.typeofclass) +" Autoencoder")
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
			x, x_extra, labels = self.finalData()
			print(x.shape)
			#X_train, X_test, y_train, y_test = auto.non_shuffling_train_test_split(x, labels)
			X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.3)            
			#auto.history = auto.fitmodel(X_train, X_test, y_train, y_test)
			auto.model.load_weights("CNN.h5")
			#auto.saveWeight(self.modelType)
			getlayer_output = K.function([auto.model.layers[0].input],
                           			       [auto.model.layers[2].output])
			layer_output = np.reshape(np.array(getlayer_output([x])), (41999, 9))
			print(layer_output.shape)
			autofinalX = self.joinFeatures(layer_output, x_extra)
			print("Dimension Reduced")
			print("Classification")												
			classifier = Classifier(autofinalX, labels)
			classifier.classify()
			print("==============================================================")

"To run CNN autoencoder just give modelType='CNN' else it will run dense autoencoder in the reduction argument given below"


if __name__=="__main__":
	data = Reduction(endRow=500, path = "/media/ghost/DATA/Dataset/", autoencoder=1,
                     typeofclass=3, ispreprocess=0, lassoC=0.00008, modelType='CNN')
	data.reduce()
