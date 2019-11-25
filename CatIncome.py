import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from feature_engine import missing_data_imputers as mdi
import re
import lightgbm as lgb
from category_encoders import TargetEncoder
from catboost import CatBoostRegressor





# 	# trainData['Country'] = trainData['Country'].map(trainData.groupby('Country')['Income'].mean())
# 	# trainData['Profession'] = trainData['Profession'].map(trainData.groupby('Profession')['Income'].mean())
# 	# trainData['UniversityDegree'] = trainData['UniversityDegree'].map(trainData.groupby('UniversityDegree')['Income'].mean())
# 	# trainData['Housing'] = trainData['Housing'].map(trainData.groupby('Housing')['Income'].mean())
# 	# trainData['EmployerSatisfaction'] = trainData['EmployerSatisfaction'].map(trainData.groupby('EmployerSatisfaction')['Income'].mean())
# 	# print(trainData.isnull().sum())

# 	# prodData['Country'] = trainData['Country'].map(trainData.groupby('Country')['Income'].mean())
# 	# prodData['Profession'] = trainData['Profession'].map(trainData.groupby('Profession')['Income'].mean())
# 	# prodData['UniversityDegree'] = trainData['UniversityDegree'].map(trainData.groupby('UniversityDegree')['Income'].mean())
# 	# prodData['Housing'] = trainData['Housing'].map(trainData.groupby('Housing')['Income'].mean())
# 	# prodData['EmployerSatisfaction'] = trainData['EmployerSatisfaction'].map(trainData.groupby('EmployerSatisfaction')['Income'].mean())
# 	print(prodData.isnull().sum())

# 	return trainData,prodData

def renameCols(data): 
	data.rename(columns = {'Total Yearly Income [EUR]':'Income'}, inplace = True)
	data.rename(columns = {'Yearly Income in addition to Salary (e.g. Rental Income)':'AdditionalIncome'}, inplace = True)
	data.rename(columns = {'Year of Record':'Year'}, inplace = True)
	data.rename(columns = {'Housing Situation':'Housing'}, inplace = True)
	data.rename(columns = {'Crime Level in the City of Employement':'CrimeLevel'}, inplace = True)
	data.rename(columns = {'Work Experience in Current Job [years]':'WorkEx'}, inplace = True)
	data.rename(columns = {'Satisfation with employer':'EmployerSatisfaction'}, inplace = True)
	data.rename(columns = {'Size of City':'CitySize'}, inplace = True)
	data.rename(columns = {'University Degree':'UniversityDegree'}, inplace = True)
	data.rename(columns = {'Wears Glasses':'WearsGlasses'}, inplace = True)
	data.rename(columns = {'Hair Color':'HairColor'}, inplace = True)
	data.rename(columns = {'Body Height [cm]':'BodyHeight'}, inplace = True)
	for r,map in data.items():
		data['AdditionalIncome'] = [re.sub('[^0-9\.]','', e) for e in data['AdditionalIncome']]
	data['AdditionalIncome'] = data['AdditionalIncome'].astype(float)
	return data

def manipulateData(data):
	#Drop Irrelavant Data
	#data = data.drop(columns = ['WearsGlasses']) 
	data['Age'] = data['Age'].fillna(method = 'bfill')
	data['Year'] = data['Year'].fillna(method = 'bfill')
	data['Age'] = data['Age']**(1/2)
	data['Year'] = data['Year']**(1/2)


	data = data.replace('nA',np.nan)
	print(data.isnull().sum())
	data['Housing'] = data['Housing'].fillna('unknown')
	data['Housing'] = data['Housing'].replace('Large House','Large')
	data['Housing'] = data['Housing'].replace('Medium House','Medium')
	data['Housing'] = data['Housing'].replace('Small House','Small')
	data['Housing'] = data['Housing'].replace('Large Apartment','Large')
	data['Housing'] = data['Housing'].replace('Medium Apartment','Medium')
	data['Housing'] = data['Housing'].replace('Small Apartment','Small')
	data['Housing'] = data['Housing'].replace('0', 'unknown')
	data['Housing'] = data['Housing'].replace('0.0','unknown')
	data['HairColor'] = data['HairColor'].fillna('unknown')
	print(data.isnull().sum())
	print(data['Housing'].value_counts())
	
	data['WorkEx'] = data.replace('#NUM!',np.nan)
	data['WorkEx'] = data.replace(0.0,0)
	data['WorkEx'] = data['WorkEx'].fillna(method = 'bfill')
	data['WorkEx'] = data['WorkEx'].fillna(method = 'bfill')
	#Gender Manipulation
	data['Gender'] = data['Gender'].fillna('other')
	data['Gender'] = data['Gender'].replace('0','other')
	data['Gender'] = data['Gender'].replace('unknown','other')
	#Fill Null values of "Year & Age" with mean of entire Dataset
	data['EmployerSatisfaction'] = data['EmployerSatisfaction'].fillna('Missing')
	# # Fill Null Value for "Profession"
	data['Profession'] = data['Profession'].fillna('missing')
	#Fill Null values & 0 for Degree
	data['UniversityDegree'] = data['UniversityDegree'].replace('0','missing')
	data['UniversityDegree'] = data['UniversityDegree'].fillna('missing')

	#Encoding Categorical Values
	data['Profession'] = data['Profession'].str.upper() # Convert all values to lowercase
	data['Profession'] = data['Profession'].str.strip() # Remove White Space
	data['UniversityDegree'] = data['UniversityDegree'].str.lower() # Convert all values to uppercase
	data['UniversityDegree'] = data['UniversityDegree'].str.strip() # Remove White Space
	data['Country'] = data['Country'].str.strip() # Remove White Space
	data['Country'] = data['Country'].fillna('missing')

	print(data.isnull().sum())
	#Gender Commented
	#data['Gender'] = data['Gender'].astype('category').cat.codes
	#################    Target Mean Encoding 62K    #####################
	
	# data['Country'] = data['Country'].map(data.groupby('Country')['Income'].mean())
	# data['Profession'] = data['Profession'].map(data.groupby('Profession')['Income'].mean())
	# data['UniversityDegree'] = data['UniversityDegree'].map(data.groupby('UniversityDegree')['Income'].mean())
	# data['Housing'] = data['Housing'].map(data.groupby('Housing')['Income'].mean())
	# data['EmployerSatisfaction'] = data['EmployerSatisfaction'].map(data.groupby('EmployerSatisfaction')['Income'].mean())
	# print('After cat')
	# corr = data.corr()
	# g = sns.heatmap(corr,annot=True, fmt = ".2f", cmap = "coolwarm")
	# plt.show()
	
	#data['Profession'] = data['Profession'].fillna(round(data['Profession'].mean()))
	# data = pd.concat([data,dummy_Gender,dummy_Profession,dummy_Country,dummy_Degree],axis = 1)  #Concat all Dummy Data with Original Data
	# data = data.drop(columns = ['Country','Gender','Profession','University Degree']) #Drop all the non-Encoded Categorical Values
	
	return data

#import Training Dataset
data = pd.read_csv('~/MachineLearning/kaggle/Project2/tcd-ml-1920-group-income-train.csv')
data.head(20)
data = renameCols(data)

data = data.drop(data[data['Income'] < 0].index)

#import Test Dataset

dataset = pd.read_csv('~/MachineLearning/kaggle/Project2/tcd-ml-1920-group-income-test.csv')
dataset = renameCols(dataset)

Merged_Dataset = pd.concat([data,dataset]) #Merge both the datasets
 
Merged_Dataset = manipulateData(Merged_Dataset) # Configure Dataset for Regression Model
prodData, trainData = [x for _, x in Merged_Dataset.groupby(Merged_Dataset['Income'] > 0)] # Split Dataset to Prod & Train

x = trainData.loc[:,trainData.columns != 'Income']
y = trainData['Income']

enc = TargetEncoder(cols=['WearsGlasses','HairColor','Gender','UniversityDegree','Profession','Country','EmployerSatisfaction','Housing','WorkEx']).fit(x, y)
ds = enc.transform(x, y)
prod_Test = prodData.loc[:,prodData.columns != 'Income']
y_prod = np.log(prodData['Income'])
ds1 = enc.transform(prod_Test)
print(ds.dtypes)



#xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 10)

# x['Year'] = x['Year'].astype(str)
# x['WorkEx'] = x['WorkEx'].astype(str)
# x['AdditionalIncome'] = x['AdditionalIncome'].astype(str)
# print(x.dtypes)
# ---------------------------Cat Boost ------------------------------------ #
xTrain, xTest, yTrain, yTest = train_test_split(ds, y, test_size = 0.3, random_state = 1234)
model=CatBoostRegressor(iterations=4000, depth=12, use_best_model=True, learning_rate=0.01, loss_function='MAE')
print('Fit Started')
#categorical_features_indices = np.where(x.dtypes != np.float)[0]
cat = model.fit(xTrain, yTrain,eval_set=(xTest, yTest),verbose = False)
# yPred =  cat.predict(xTest)
# errors = abs(yPred - yTest)
# print('Mean Absolute Error:', round(np.mean(errors), 2))
# print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, yPred)))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(yTest, yPred))
# print('Prod_Test Predict')
yProdPrediction = np.exp(cat.predict(prod_Test))
#Configure Dataset and create a csv
prodData['Income'] = yProdPrediction
prodData.rename(columns = {'Income':'Total Yearly Income [EUR]'}, inplace = True)
incomePrediction = prodData[['Instance','Total Yearly Income [EUR]']]
incomePrediction.to_csv('~/MachineLearning/kaggle/anotherNewGroupIncomeKaggle.csv',index = False)
# ---------------------------LightGBM ------------------------------------ #
# xTrain, xTest, yTrain, yTest = train_test_split(ds, y, test_size = 0.1, random_state = 1234)
# params = {
#           'max_depth': 20,
#           'num_leaves': 80,
#           'learning_rate': 0.001,
#           "boosting": "gbdt",
#           "bagging_seed": 11,
#           "metric": 'mae',
#           "verbosity": -1,
#          }
# trainingData = lgb.Dataset(xTrain, label=yTrain)
# validationData = lgb.Dataset(xTest, label=yTest)
# clf = lgb.train(params, trainingData, 100000, valid_sets = [trainingData, validationData], verbose_eval=1000, early_stopping_rounds=500)
# print('Fit Started')
# # yPred =  clf.predict(xTest)
# # errors = abs(yPred - yTest)
# # print('Mean Absolute Error:', round(np.mean(errors), 2))
# # print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, yPred)))
# # # Explained variance score: 1 is perfect prediction
# # print('Variance score: %.2f' % r2_score(yTest, yPred))
# # print('Prod_Test Predict')
# yProdPrediction = np.exp(clf.predict(ds1))
# #Configure Dataset and create a csv
# prodData['Income'] = yProdPrediction
# prodData.rename(columns = {'Income':'Total Yearly Income [EUR]'}, inplace = True)
# incomePrediction = prodData[['Instance','Total Yearly Income [EUR]']]
# incomePrediction.to_csv('~/MachineLearning/kaggle/KagggleGroupIncome.csv',index = False)


#---------------------------Linear Regression Model ------------------------------------ #
# linearRegressor = LinearRegression()
# print('Training started')
# linearRegressor.fit(xTrain, yTrain)
# print('Predict started')
# yPred =  linearRegressor.predict(xTest)
# errors = abs(yPred - yTest)
# print('Mean Absolute Error:', round(np.mean(errors), 2))
# print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, yPred)))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(yTest, yPred))

# yProdPrediction = linearRegressor.predict(prod_Test)

# yProdPrediction = clf.predict(prod_Test)

# #Configure Dataset and create a csv
# prodData['Income'] = yProdPrediction
# prodData.rename(columns = {'Income':'Total Yearly Income [EUR]'}, inplace = True)
# incomePrediction = prodData[['Instance','Total Yearly Income [EUR]']]
# incomePrediction.to_csv('~/MachineLearning/kaggle/NEWGroupIncome5.csv',index = False)

# ---------------------------Random Forest Model ------------------------------------ #
# rf = RandomForestRegressor(n_estimators = 10, random_state = 1234)
# print('Training started')
# rf.fit(xTrain, yTrain);
# print('Predict started')
# predictions = rf.predict(xTest)
# errors = abs(predictions - yTest)
# print('Mean Absolute Error:', round(np.mean(errors), 2))
# mape = 100 * (errors / yTest)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
# print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(yTest, predictions)))

# yProdPrediction = rf.predict(prod_Test)

# #Configure Dataset and create a csv
# prodData['Income'] = yProdPrediction
# prodData.rename(columns = {'Income':'Total Yearly Income [EUR]'}, inplace = True)
# incomePrediction = prodData[['Instance','Total Yearly Income [EUR]']]
# incomePrediction.to_csv('~/MachineLearning/kaggle/GroupIncome3.csv',index = False)