import os
import warnings

import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle

from sklearn .utils import shuffle
from sklearn import neighbors, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
warnings.filterwarnings("ignore", category=ConvergenceWarning)

########################
### Helper Functions ###
########################
def str_to_intlist(x):
	w = x.rstrip("\n ")
	s = w.split(" ")
	int_list = []
	for item in s:
		new_item = int(item)
		int_list.append(new_item)
	return int_list

def data_to_list(dataset_obj):
	data = []
	for i in dataset_obj:
		data.append(i)
	data_list = []
	for i in data:
		new = str_to_intlist(i)
		data_list .append(new)
	return data_list

def shuffle_data(x,y):
	x, y = shuffle(x, y, random_state = 0)
	return x,y

def plot_ConfusionMatrix(conf_mat, num_of_classes, model_name):
	df_cm = pd.DataFrame(conf_mat, range(num_of_classes), range(num_of_classes))
	plt.figure(figsize=(12, 9))
	plt.title(model_name)
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	sn.set(font_scale=1.4) # for label size
	sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues") # font size
	# plt.show()
	plt.savefig(model_name + ".png", format = 'png')


###################################
### Final Boards Classification ###
###################################
def FinalBoard_KNN(x, y,num_neighbors, extraFileNameInfo):
	print("\n--------KNN--------")
	NumFolds   = 10
	NumClasses = 2

	x,y = shuffle_data(x,y)
	knn_cross = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)

	y_pred    = cross_val_predict(knn_cross, x, y, cv = NumFolds)
	conf_mat  = confusion_matrix(y, y_pred, normalize = 'true')

	plot_ConfusionMatrix(conf_mat, NumClasses, "KNN_Final" + extraFileNameInfo)

	cv_scores =  cross_val_score(knn_cross, x, y, cv = NumFolds)
	print("Cross Validation Mean Accuracy scores: {:.2f}".format(np.mean(cv_scores)))

def FinalBoard_SVM(x, y, extraFileNameInfo):
	print("\n--------SVM--------")
	NumFolds   = 10
	NumClasses = 2
	x,y = shuffle_data(x,y)

	svm_cross = svm.SVC()
	y_pred    = cross_val_predict(svm_cross, x, y, cv = NumFolds)
	conf_mat  = confusion_matrix(y, y_pred, normalize = 'true')
	plot_ConfusionMatrix(conf_mat, NumClasses, "SVM_Final" + extraFileNameInfo)

	cv_scores = cross_val_score(svm_cross, x, y, cv = NumFolds)
	print('Cross Validation Mean Accuracy scores {:.2f}'.format(np.mean(cv_scores)))

def FinalBoard_MLP(x, y, HiddenLayerSizes, trainBool, extraFileNameInfo):
	NumClasses = 2
	NumFolds   = 10
	print("\n--------MLP--------")
	x,y = shuffle_data(x,y)

	if trainBool:
		mlp_cross = MLPClassifier(hidden_layer_sizes = HiddenLayerSizes, random_state = 0)
		y_pred    = cross_val_predict(mlp_cross, x, y, cv = NumFolds)
		np.save('Saved_Models/MLP_FinalBoard_y_pred' + extraFileNameInfo, y_pred)

	else:
		y_pred = np.load('Saved_Models/MLP_FinalBoard_y_pred.npy')

	Accuracy = accuracy_score(y, y_pred)
	conf_mat = confusion_matrix(y, y_pred,normalize='true')
	plot_ConfusionMatrix(conf_mat, NumClasses, "MLP_Final" + extraFileNameInfo)
	print("Accuracy for MLP: {:.2f}".format(Accuracy))



###################
### MULTI-CLASS ###
###################
def multiclass_KNN(x,y, num_neighbors, extraFileNameInfo):
	print("\n--------MultiClass KNN--------")
	NumClasses = 9
	NumFolds   = 10
	x,y = shuffle_data(x,y)
	knn_multiclass = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=num_neighbors))

	y_pred   = cross_val_predict(knn_multiclass, x, y, cv = NumFolds)
	conf_mat = confusion_matrix(y, y_pred, normalize = 'true')
	plot_ConfusionMatrix(conf_mat, NumClasses, "KNN_SingleLabel" + extraFileNameInfo)

	cv_scores =  cross_val_score(knn_multiclass, x, y, cv = NumFolds)
	print('Cross Validation Mean Accuracy scores {:.2f}'.format(np.mean(cv_scores)))


def multiclass_SVM(x,y, trainBool, extraFileNameInfo):
	print("\n--------MultiClass SVM--------")
	NumClasses = 9
	NumFolds   = 10
	x,y = shuffle_data(x,y)

	if trainBool: #Train classifier
		svm_multiclass = OneVsRestClassifier(svm.SVC(kernel='rbf'))

		y_pred   = cross_val_predict(svm_multiclass, x, y, cv = NumFolds)
		np.save('Saved_Models/SVM_MultiClass_y_pred' + extraFileNameInfo, y_pred)

	else: #evaluate
		#Read in y_pred
		y_pred = np.load('Saved_Models/SVM_MultiClass_y_pred.npy')

	Accuracy = accuracy_score(y, y_pred)
	conf_mat = confusion_matrix(y, y_pred, normalize = 'true')
	plot_ConfusionMatrix(conf_mat, NumClasses, "SVM_SingleLabel" + extraFileNameInfo)
	print("Accuracy for SVM: {:.2f}".format(np.asarray(Accuracy)))


def multiclass_MLP(x, y, HiddenLayerSizes, trainBool, extraFileNameInfo):
	NumClasses = 9
	NumFolds   = 10
	print("\n--------MultiClass MLP--------")
	x,y = shuffle_data(x,y)

	if trainBool: #Train classifier
		mlp_multiclass = MLPClassifier(hidden_layer_sizes = HiddenLayerSizes)
		y_pred         = cross_val_predict(mlp_multiclass, x, y, cv = NumFolds)
		mlp_multiclass.fit(x,y)
		pickle.dump(mlp_multiclass, open('Saved Models/multiclass_MLP' + extraFileNameInfo+'.sav',"wb"))
		np.save('Saved_Models/MLP_MultiClass_y_pred' + extraFileNameInfo, y_pred)

	else: #evaluate
		#Read in y_pred
		y_pred = np.load('Saved_Models/MLP_MultiClass_y_pred.npy')

	Accuracy = accuracy_score(y, y_pred)
	conf_mat = confusion_matrix(y, y_pred, normalize = 'true')
	plot_ConfusionMatrix(conf_mat, NumClasses, "MLP_SingleLabel" + extraFileNameInfo)
	print("Accuracy for MLP: {:.2f}".format(np.asarray(Accuracy)))


###################
### MULTI-LABEL ###
###################
def k_FoldCV_Multilabel(multi_x, Model, CurrentYToRegress, NumFolds, classIndex, ModelString):
	multi_x, CurrentYToRegress = shuffle_data(multi_x, CurrentYToRegress)

	kf = KFold(n_splits = NumFolds)
	Accuracies = [] #store accuracies per fold
	y_preds    = []
	y_tests    = []
	for train_index, test_index in kf.split(multi_x): #split data into NumFolds
		X_train, X_test = multi_x[train_index], multi_x[test_index]
		y_train, y_test = CurrentYToRegress[train_index], CurrentYToRegress[test_index]
		
		if ModelString == 'Linear Regressor':
			#Fit model on training set for current fold
			w 	   = np.dot(np.dot(np.linalg.pinv(np.dot(X_train.T,X_train)),X_train.T),y_train)
			y_pred = np.dot(X_test,w)
			np.save("Saved_Models/LinearRegressor_weights"+str(classIndex),w)
		else:
			#Fit model on training set for current fold
			Model.fit(X_train, y_train)
			if ModelString == "MLP Regressor":
				pickle.dump(Model, open("Saved_Models/MLPRegressor"+str(classIndex)+".sav", 'wb'))
			elif ModelString == "KNN Regressor":
				pickle.dump(Model, open("Saved_Models/KNNRegressor"+str(classIndex)+".sav", 'wb'))

			#Predict on current X_Test fold
			y_pred = Model.predict(X_test)

		#round to make regression be 0 or 1
		y_pred  = np.round(y_pred)
		y_preds.append(y_pred)
		y_tests.append(y_test)

		#calculate Accuracy as a classifer (per class)
		Accuracy = accuracy_score(y_test, y_pred)
		Accuracies.append(Accuracy)
	y_preds = [y for x in y_preds for y in x] #flatten arrays
	y_tests = [y for x in y_tests for y in x]

	meanAccs = np.mean(Accuracies) #Accuracy for 1 class
	# print("Accuracy for folds\n", Accuracies)
	# print('Cross Validation Accuracy Per Class:', meanAccs)
	return meanAccs, y_preds, y_tests

def multilabel_KNN(multi_x, multi_y, K):
	print("\n--------Multi-Label KNN--------")
	NumFolds   = 10
	NumClasses = 9

	AccuracyPerClass = []
	#Do a single regressor per class
	for i in range(NumClasses):
		CurrentYToRegress = multi_y[:, i]

		#Set up KNN Regressor object
		KNN_Regress = neighbors.KNeighborsRegressor(n_neighbors = K)

		#k-Fold cross validation (per single class)
		AvgAcc, y_p, y_t = k_FoldCV_Multilabel(multi_x, KNN_Regress, CurrentYToRegress, NumFolds, i, "KNN Regressor")
		AccuracyPerClass.append(AvgAcc)

	print('Cross Validation Accuracy Per Class:', np.asarray(AccuracyPerClass)) #array

	return AccuracyPerClass

def multilabel_MLP(multi_x, multi_y, HiddenLayerSizes, trainBool):
	print("\n--------Multi-Label MLP--------")
	NumFolds   = 10
	NumClasses = 9

	AccuracyPerClass = []
	#Do a single regressor per class
	for i in range(NumClasses):
		CurrentYToRegress = multi_y[:, i]

		#Set up MLP Regressor object
		MLP_Regress = MLPRegressor(solver='sgd',activation='relu', alpha=1e-5, hidden_layer_sizes = HiddenLayerSizes, random_state=0, batch_size=200,max_iter=100)

		#k-Fold cross validation (per single class)
		if trainBool:
			AvgAcc, y_pred, y_test = k_FoldCV_Multilabel(multi_x, MLP_Regress, CurrentYToRegress, NumFolds, i, "MLP Regressor")
			np.save('Saved_Models/MLP_MultiLabel_y_pred', y_pred)
			np.save('Saved_Models/MLP_MultiLabel_y_test', y_test)
		else:
			y_pred = np.load('Saved_Models/MLP_MultiLabel_y_pred.npy', allow_pickle = True)
			y_test = np.load('Saved_Models/MLP_MultiLabel_y_test.npy', allow_pickle = True)
			AvgAcc = accuracy_score(y_test, y_pred)
		AccuracyPerClass.append(AvgAcc)

	print('Cross Validation Accuracy Per Class:', np.asarray(AccuracyPerClass)) #array

	return AccuracyPerClass

def multilabel_LinearReg(multi_x, multi_y):
	print("\n--------Multi-Label Linear Regression--------")
	NumFolds   = 10
	NumClasses = 9

	AccuracyPerClass = []
	#Do a single regressor per class
	for i in range(NumClasses):
		CurrentYToRegress = multi_y[:, i]

		#k-Fold cross validation (per single class)
		AvgAcc, y_p, y_t = k_FoldCV_Multilabel(multi_x, "Linear Regressor",CurrentYToRegress, NumFolds,i, "Linear Regressor")
		AccuracyPerClass.append(AvgAcc)

	print('Cross Validation Accuracy Per Class:', np.asarray(AccuracyPerClass)) #array

	return AccuracyPerClass


#####################
### START PROGRAM ###
#####################
#Train all models or use pre-trained models?
userInput = input("Would you like to train (t) or evaluate (e)? ")
if userInput == 't':
	trainBool = True
elif userInput == 'e':
	trainBool = False
else:
	sys.exit('Enter t or e.')

dataset_final  = open("datasets/tictac_final.txt","r")
dataset_single = open("datasets/tictac_single.txt","r")
dataset_multi  = open("datasets/tictac_multi.txt","r")

print("\n###########################---- Final Boards Classification ---- ############################")
final_listdata = data_to_list(dataset_final)
#print(final_listdata[10]) -> [1, 1, 1, 1, 0, -1, -1, 0, -1, 1]

final_x = []
final_y = []
for item in final_listdata:
	out = item.pop()
	final_y.append(out)
	final_x.append(item)

K = 5
FinalBoard_KNN(final_x, final_y, K, '')
FinalBoard_SVM(final_x, final_y, '')
HiddenLayerSizes = (128, 128, 64, 64)
FinalBoard_MLP(final_x, final_y, HiddenLayerSizes, trainBool, '')

# Train classifiers on 1/10th of the data
# final_x, final_y = shuffle_data(final_x, final_y)
# NumSamples       = len(final_x)
# TenthOfData      = round(NumSamples/10)
# final_xSubset    = final_x[:TenthOfData]
# final_ySubset    = final_y[:TenthOfData]
#
# K = 5
# FinalBoard_KNN(final_xSubset, final_ySubset, K, '_tenth')
# FinalBoard_SVM(final_xSubset, final_ySubset, '_tenth')
# HiddenLayerSizes = (128, 128, 64, 64)
# FinalBoard_MLP(final_xSubset, final_ySubset, HiddenLayerSizes, trainBool, '_tenth')


print("\n##################---- Intermediate boards optimal play [Single Label]:---- ##################")
single_listdata = data_to_list(dataset_single)
#print(single_listdata[10]) -> [-1, 0, 1, 0, 0, 0, 0, 0, 0, 3]

single_x = []
single_y = []
for item in single_listdata:
	out = item.pop()
	single_y.append(out)
	single_x.append(item)

num_neighbors = 5
multiclass_KNN(single_x, single_y, num_neighbors, '')
multiclass_SVM(single_x, single_y, trainBool, '')
HiddenLayerSizes = (128, 128, 64, 64)
multiclass_MLP(single_x, single_y, HiddenLayerSizes, trainBool, '')

# Train classifiers on 1/10th of the data
# single_x, single_y = shuffle_data(single_x, single_y)
# NumSamples         = len(single_x)
# TenthOfData        = round(NumSamples/10)
# single_xSubset     = single_x[:TenthOfData]
# single_ySubset     = single_y[:TenthOfData]
#
# num_neighbors = 5
# multiclass_KNN(single_xSubset, single_ySubset, num_neighbors, '_tenth')
# multiclass_SVM(single_xSubset, single_ySubset, trainBool, '_tenth')
# HiddenLayerSizes = (128, 128, 64, 64)
# multiclass_MLP(single_xSubset, single_ySubset, HiddenLayerSizes, trainBool, '_tenth')

print("\n##################---- Intermediate boards optimal play [Multi Label]:---- ##################")
multi_listdata = data_to_list(dataset_multi)
#print(multi_listdata[15]) -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

#Split dataset into input and target
multi_x = []
multi_y = []
for item in multi_listdata:
	output = item[9:]
	input  = item[:9]
	multi_y.append(output)
	multi_x.append(input)

multi_y = np.asarray(multi_y)
multi_x = np.asarray(multi_x)

K = 5
AvgAccuracyPerClass_KNN = multilabel_KNN(multi_x, multi_y, K)
print("\nAverage Accuracy KNN Multilabel Regressor: {:.2f}".format(np.mean(AvgAccuracyPerClass_KNN)))

HiddenLayerSizes = (128, 128, 64, 64)
AvgAccuracyPerClass_MLP = multilabel_MLP(multi_x, multi_y, HiddenLayerSizes, trainBool)
print("\nAverage Accuracy MLP Multilabel Regressor: {:.2f}".format(np.mean(AvgAccuracyPerClass_MLP)))

AvgAccuracyPerClass_LR = multilabel_LinearReg(multi_x, multi_y)
print("\nAverage Accuracy LR Multilabel Regressor: {:.2f}".format(np.mean(AvgAccuracyPerClass_LR)))
