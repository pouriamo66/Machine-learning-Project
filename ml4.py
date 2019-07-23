#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale,normalize,minmax_scale
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[2]:


#read the dataset
df=pd.read_excel("storedata.xlsx")


# In[3]:


####getting information
df.info()
df.shape
df.describe()


# In[4]:


sns.distplot(df['Profit'])


# In[23]:


dfs=df[['Staff','Floor Space','Window','Demographic score','40min population','30 min population','20 min population','10 min population','Store age','Clearance space','Competition number','Competition score']]
sns.set_style("whitegrid")
plt.figure(figsize=(20,20))
flatui = [ "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
sns.pairplot(dfs,palette=flatui)
plt.savefig('data.png')


# In[6]:


plt.boxplot(df["Staff"])


# In[11]:


sns.countplot(x="Car park",data=df)


# In[30]:


####correlation
tc=df.corr()
plt.figure(figsize=(15,9))
sns.heatmap(tc,cmap='coolwarm',annot=True)


# In[ ]:


#################OUTLLIERS


# In[3]:


##  finding outliers function
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
############################


# In[4]:


###finding outliers
outlier_datapoints = detect_outlier(df['Staff'])
print(outlier_datapoints)

df[df['Staff']==600]
df[df['Staff']==300]
df[df['Staff']==-2]


# In[5]:


#####droping ouliers
df1=df.drop(df.index[[2,53,109]])
df1.set_index(df1['Store ID'])
df2=df1.drop('Country', axis=1)


# In[6]:


##  fixing Bad data in car park columns
df2['Car park'].replace('Y','Yes',inplace=True)
df2['Car park'].replace('N','No',inplace=True)
df2['Location'].replace('Village','Shopping Centre',inplace=True)


# In[7]:


df2['Car park'].describe()


# In[145]:


sns.countplot(x="Performance",data=df)


# In[146]:


sns.countplot(x="Location",data=df2)


# In[8]:


########dummies variable 
df2['Car park']=df2['Car park'].map({'Yes':1,'No':0})
df2=df2.iloc[:,3:]
df2['Performance']=df2['Performance'].map({'Excellent':4,'Good':2,'Poor':1,'Reasonable':3})


# In[9]:


##### make the Location as Dummies 
df2=pd.get_dummies(df2)


# In[11]:


df2.head()


# In[10]:


#####dataset without dummies and targets
df3=df2[['Staff','Floor Space','Window','Demographic score','40min population','30 min population','20 min population','10 min population','Store age','Clearance space','Competition number','Competition score']]


# In[11]:


###scalling dataset
from sklearn.preprocessing import scale,normalize,minmax_scale
minmax_s=minmax_scale(df3)
df4=pd.DataFrame(minmax_s,index=df3.index,columns=df3.columns)


# In[19]:


df4.describe()


# In[12]:


df4[['Car park','Location_High Street','Location_Retail Park','Location_Shopping Centre','Profit','Performance']]=df2[['Car park','Location_High Street', 'Location_Retail Park','Location_Shopping Centre','Profit','Performance']]


# In[42]:


#####Feature selection


# In[13]:


X=df4.drop(df4[['Profit','Performance']],axis=1)
y=df4['Profit']

from sklearn.model_selection import RandomizedSearchCV

# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
                'bootstrap': bootstrap}


# In[18]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X,y)


# In[17]:



###Feature selection another 
from sklearn.ensemble import RandomForestRegressor  
random_forest = RandomForestRegressor (42)  
random_forest.fit(X,y)  
feature_importances = random_forest.feature_importances_  
print(feature_importances)  
# feature importances from random forest model  
importances = random_forest.feature_importances_  
# index of greatest to least feature importances  
sorted_index = np.argsort(importances)[::-1]  
x = range(len(importances))  
# create tick labels  
plt.figure(figsize=(15,9))  
labels = np.array(X.columns)[sorted_index]  
plt.bar(x, importances[sorted_index], tick_label=labels)  
# rotate tick labels to vertical  
plt.xticks(rotation=90)  
plt.show()  


# In[19]:




feature_list = list(X.columns)
# Instantiate random forest and train on new features
from sklearn.ensemble import RandomForestClassifier
rf_exp = RandomForestRegressor(n_estimators= 1000, random_state=42)
rf_exp.fit(X,y)


# Get numerical feature importances
importances = list(rf_exp.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# In[191]:


# list of x locations for plotting
x_values = list(range(len(importances)))
plt.figure(figsize=(15,9)) 
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[15]:


#######Targets===Profit
Xs=df4.drop(df4[['Profit','Performance']],axis=1)
ys=df4['Profit'].values.reshape(-1,1)


# In[16]:


######Ridge hyperparameter
search=GridSearchCV(estimator=ridge,param_grid={'alpha':np.logspace(-5,2,8)},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=10)
search.fit(Xs,ys)
search.best_params_


# In[17]:


######Ridge
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.2, random_state=10)
ridge=Ridge(alpha=1,normalize=False)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(" Test Root Mean Squared Error: {}".format(rmse))


# In[30]:


y0_pred = ridge.predict(X_test)
y1_pred = ridge.predict(X_train)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse0 = np.sqrt(mean_squared_error(y_test,y0_pred))
rmse = np.sqrt(mean_squared_error(y_train,y1_pred))
print("Root Mean Squared Error for Test: {}".format(rmse0))
print("Root Mean Squared Error for Train: {}".format(rmse))


# In[164]:


y_pred = ridge.predict(X_train)
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
print(" Train Root Mean Squared Error: {}".format(rmse))


# In[165]:


print (ridge.score(X_test, y_test))


# In[166]:


# Create a scatter plot with train and test actual vs predictions
plt.scatter(y_train, y1_pred, label='train')
plt.scatter(y_test, y0_pred, label='test')
plt.legend()
plt.show()


# In[168]:


#####Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(ridge,Xs, ys,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[169]:


ridgecv=RidgeCV(cv=10)
ridgecv.fit(Xs,ys)


# In[174]:


##Ridge coefficients function
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)


# In[175]:


print ("Ridge model:", pretty_print_coefs(ridge.coef_))


# In[173]:


print(ridge.coef_)


# In[176]:


from sklearn.linear_model import Lasso
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,0.4,1,5,10,20]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(Xs,ys)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[177]:


lasso = Lasso(alpha=20,normalize=False)

# Fit the regressor to the data
lasso.fit(Xs,ys)

y_pred = ridge.predict(X_test)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[178]:


y_pred = lasso.predict(X_test)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[179]:


ya_pred = lasso.predict(X_test)
yb_pred = lasso.predict(X_train)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmsea = np.sqrt(mean_squared_error(y_test,ya_pred))
rmseb = np.sqrt(mean_squared_error(y_train,yb_pred))
print("Root Mean Squared Error for Test: {}".format(rmsea))
print("Root Mean Squared Error for Train: {}".format(rmseb))


# In[180]:


# Create a scatter plot with train and test actual vs predictions
plt.scatter(y_train, yb_pred, label='train')
plt.scatter(y_test, ya_pred, label='test')
plt.legend()
plt.show()


# In[121]:


y1_pred = lasso.predict(X_test)
print (lasso.score(X_test, y_test))


# In[122]:


plt.scatter(y_test,y1_pred)


# In[123]:


#####Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(lasso,Xs, ys,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[124]:


# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
print(lasso_coef)


# In[181]:


# Plot the coefficients
plt.figure(figsize=(15,9))
plt.plot(range(len(Xs.columns)), lasso_coef)
plt.xticks(range(len(Xs.columns)), Xs.columns.values, rotation=60)
plt.margins(0.02)
plt.show()


# In[18]:


mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(1,15)],
             'activation': ['relu'],
               'solver': ['adam'],
               'learning_rate': ['constant'],
               'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
               'max_iter': [1000],
               'early_stopping': [False],
              'warm_start': [False]}
mlp = MLPRegressor()
param_grid = {'hidden_layer_sizes': [i for i in range(1,15)],
             'activation': ['relu'],
               'solver': ['adam'],
               'learning_rate': ['constant'],
               'learning_rate_init': [0.001],
              'power_t': [0.5],
              'alpha': [0.0001],
               'max_iter': [1000],
               'early_stopping': [False],
              'warm_start': [False]}
_GS = GridSearchCV(mlp, param_grid=param_grid, verbose=True, pre_dispatch='2*n_jobs')
_GS.fit(Xs, ys)


# In[45]:


df4.columns


# In[20]:


X1=df4[['Staff','Competition score','Floor Space', 'Car park' ,'Location_High Street', 'Location_Retail Park',
       'Location_Shopping Centre']]
y1=df4['Profit']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.20, random_state=10)


# In[128]:


mlp =MLPRegressor(activation='relu', alpha=0.5, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(1,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=900, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)      
mlp.fit(X1_train,y1_train)
y2_pred = mlp.predict(X1_test)
y3_pred = mlp.predict(X1_train)
# # Compute and print R^2 and RMSE
print("R^2: {}".format(mlp.score(X1_test, y1_test)))
rmse1 = np.sqrt(mean_squared_error(y1_test,y2_pred))
rmse2 = np.sqrt(mean_squared_error(y1_train,y3_pred))
print("Root Mean Squared Error for Test: {}".format(rmse1))
print("Root Mean Squared Error for Train: {}".format(rmse2))


# In[22]:


##MLPregressor with Kfold based on the hyperparameters i got from GridCV
  # Compute 10-fold cross-validation scores: cv_scores 
cv_scores_MLP = cross_val_score(mlp,X1, y1,cv=5) 
  # Print the 10-fold cross-validation scores 
print(cv_scores_MLP) 
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_MLP)))   
# or  np.mean (cv_scores_MLP)


# In[183]:


# Create a scatter plot with train and test actual vs predictions
plt.scatter(y1_train, y3_pred, label='train')
plt.scatter(y1_test, y2_pred, label='test')
plt.legend()
plt.show()


# In[ ]:


#####performance


# In[31]:


X2=df4.drop(df4[['Profit','Performance']],axis=1)
y2=df4['Performance']


# In[43]:


rf1_exp = RandomForestClassifier (n_estimators= 100, random_state=42)
rf1_exp.fit(X2,y2)
# Get numerical feature importances
importances1 = list(rf1_exp.feature_importances_)
# List of tuples with variable and importance
feature_importances1= [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances1)]
# Sort the feature importances by most important first
feature_importances1 = sorted(feature_importances1, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances1]


# In[44]:


# list of x locations for plotting
x_values = list(range(len(importances1)))
# Make a bar chart
plt.bar(x_values, importances1, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# In[14]:


X1s=df4[['Staff','Competition score','Floor Space', 'Car park' ,'Location_High Street', 'Location_Retail Park',
       'Location_Shopping Centre']]
y1s=df4['Performance']
X2_train, X2_test, y2_train, y2_test = train_test_split(X1s, y1s, test_size = 0.20, random_state=10)


# In[74]:


# Grid search cross validation
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=5)
logreg_cv.fit(X1s,y1s)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[80]:


lr_mn = LogisticRegression(multi_class="multinomial", solver="lbfgs",C=1)
lr_mn.fit(X2_train, y2_train)

print(" training accuracy:", lr_mn.score(X2_train, y2_train))
print(" test accuracy    :", lr_mn.score(X2_test, y2_test))


# In[82]:


y1_pred = lr_mn.predict(X2_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y2_test, y1_pred))
print(classification_report(y2_test, y1_pred))


# In[95]:


y4_pred = lr_mn.predict(X2_test)
y5_pred = lr_mn.predict(X2_train)


# In[99]:


y4_pred


# In[98]:


y5_pred


# In[100]:


# Make predictions with our model
# Create a scatter plot with train and test actual vs predictions
plt.scatter(y2_train, y5_pred, label='train')
plt.scatter(y2_test, y4_pred, label='test')
plt.legend()
plt.show()


# In[89]:


####roc carve
import scikitplot as skplt
lr_mn.fit(X2_train,y2_train)

y1_probas = lr_mn.predict_proba(X2_test)
skplt.metrics.plot_roc(y2_test, y1_probas)


# In[92]:


# Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(lr_mn,X1s, y1s,cv=10)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[33]:


dtree =  DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=90)
dtree.fit(X2_train,y2_train)
predictions = dtree.predict(X2_test)
print(classification_report(y2_test,predictions))
print(" training accuracy:", dtree.score(X2_train, y2_train))
print(" test accuracy    :", dtree.score(X2_test, y2_test))


# In[32]:


#GridsearchCv for Decision tree
decision_tree_classifier = DecisionTreeClassifier(random_state=10)
parameter_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  'criterion':['gini','entropy']}
grid_search = GridSearchCV(decision_tree_classifier, param_grid = parameter_grid,
                          cv = 5)
grid_search.fit(X1s, y1s)
print ("Best Score: {}".format(grid_search.best_score_))
print ("Best params: {}".format(grid_search.best_params_))


# In[31]:


from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=7)

# Fit it to the data
tree_cv.fit(X1s, y1s)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[ ]:





# In[37]:


# Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(dtree,X1s, y1s,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[38]:


df4.columns


# In[23]:


X1s=df4[['Staff','Competition score','Floor Space', 'Car park' ,'Location_High Street', 'Location_Retail Park',
       'Location_Shopping Centre','Demographic score','40min population', '20 min population',]]
y1s=df4['Performance']
X2_train, X2_test, y2_train, y2_test = train_test_split(X1s, y1s, test_size = 0.20, random_state=10)


# In[53]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 
features = list(X1s.columns[0:])
features
['Staff',
 'Floor Space',
 'Competition score',
 'Car park',
 'Location_High Street',
 'Location_Retail Park',
 'Location_Shopping Centre']

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png()) 


# In[145]:


# Compute 10-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(dtree,X1s, y1s,cv=10)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 10-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[61]:


from sklearn.ensemble import RandomForestClassifier 

rfc1=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy', 

            max_depth=10, max_features='auto', max_leaf_nodes=4, 

            min_impurity_decrease=0.0, min_impurity_split=None, 

            min_samples_leaf=1, min_samples_split=4, 

            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1, 

            oob_score=False, random_state=10, verbose=1, 

            warm_start=False) 

rfc1.fit(X2_train, y2_train) 

 

rfc1_pred = rfc1.predict(X2_test) 

print(confusion_matrix(y2_test,rfc1_pred)) 

print(classification_report(y2_test,rfc1_pred))
print(rfc1.score(X2_test,y2_test))


# In[24]:


mlp_GridCV = MLPClassifier(max_iter=500, random_state=10)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (13,13,13)],
    'activation': ['tanh', 'relu','identity'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.05, 0.1, 1, 100, 1000],
    'learning_rate': ['constant','adaptive','invscaling'],
}

clf = GridSearchCV(mlp_GridCV, parameter_space, n_jobs=-1, cv=3)
clf.fit(X1s,y1s)
# Best paramete set
print('Best parameters found:\n', clf.best_params_)


# mlp1 =MLPClassifier(activation='relu', alpha=0.5, batch_size='auto', beta_1=0.9,
#        beta_2=0.999, early_stopping=False, epsilon=1e-08,
#        hidden_layer_sizes=(50, 100, 50), learning_rate='constant',
#        learning_rate_init=0.001, max_iter=900, momentum=0.9,
#        n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
#        random_state=None, shuffle=True, solver='adam', tol=0.0001,
#        validation_fraction=0.1, verbose=False, warm_start=False)      
# mlp1.fit(X2_train,y2_train)
# y7_pred = mlp1.predict(X2_test)
# y8_pred = mlp1.predict(X2_train)
# 
# print("R^2: {}".format(mlp1.score(X1_test, y1_test)))
# rmse3 = np.sqrt(mean_squared_error(y2_test,y2_pred))
# rmse4 = np.sqrt(mean_squared_error(y2_train,y3_pred))
# print("Root Mean Squared Error for Test: {}".format(rmse3))
# print("Root Mean Squared Error for Train: {}".format(rmse4))
# 

# In[27]:


mlp1 =MLPClassifier(activation='relu', alpha=0.5, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10,100,100), learning_rate='constant',
       learning_rate_init=0.001, max_iter=900, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=100, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=10, warm_start=False)      
mlp1.fit(X2_train,y2_train)
y7_pred = mlp1.predict(X2_test)
y7_pred
y8_pred = mlp1.predict(X2_train)
accuracy_score(y2_test, y7_pred)
print("  test accuracy:",accuracy_score(y2_test, y7_pred))
print(" training accuracy    :",accuracy_score(y2_train,y8_pred))


# In[69]:


cm = confusion_matrix(y2_test, y7_pred)
cm
sns.heatmap(cm, center=True)
plt.show()


# cm

# In[71]:


print(classification_report(y2_test, y7_pred))


# In[28]:


# hidden_layer_sizes=(13,13,13),max_iter=500 # Compute 5-fold cross-validation scores: cv_scores 
cv_scores_MLP = cross_val_score(mlp1,X1s, y1s,cv=5) 
  # Print the 10-fold cross-validation scores 
print(cv_scores_MLP) 
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores_MLP)))   
# or  np.mean (cv_scores_MLP)

