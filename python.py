#importing the modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
#import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
plt.rc('font', size=20) 
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20) 
plt.rc('legend', fontsize=20)    
plt.rc('figure', titlesize=20)
import warnings
warnings.filterwarnings('ignore')
%config Completer.use_jedi = False


#describing the columns and reading the data from the csv file
col_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "labels"
]
df = pd.read_csv('../project/nsl-kdd/KDDTrain+.txt', names=col_names, index_col=False)


#exploring the dataset
df.head()
df.describe()
#labelling data as attack type and normal type and counting their values in the dataset
df.loc[df.labels != 'normal', 'labels'] = 'attack'
df['labels'].value_counts()

#exploring the dataset some more
print(f"Dataset shape:{df.shape}")
df.isna().sum()

#printing duplicate values and droping the columns with duplicate values
print(f"The number of duplicated records: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
.............................................
#splitting the data
X = df.drop('labels',axis=1)
Y = df['labels']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=41)
print(f"Train-set shape:{X_train.shape}")
print(f"Test-set shape:{X_test.shape}")

#exploring the training data
X_train.head()

#Scaling numeric features using Standard scalar
std_scaler = preprocessing.StandardScaler()
def standardization(df, col):
    for i in col:
        arr = np.array(df[i])
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df
  
 numeric_col = X_train.select_dtypes(include='number').columns
X_train = standardization(X_train, numeric_col)
X_test = standardization(X_test, numeric_col)
X_train.head()

cat_col = X_train.select_dtypes('object').columns
# encoding train set
categorical_train = X_train[cat_col]
categorical_train = pd.get_dummies(categorical_train,columns=cat_col)
categorical_train.head()

X_train.drop(cat_col, axis=1, inplace=True)
X_train = pd.concat([X_train, categorical_train],axis=1)
X_train.head()


cat_col = X_test.select_dtypes('object').columns
# encoding test set
categorical_test = X_test[cat_col]
categorical_test = pd.get_dummies(categorical_test,columns=cat_col)

#Adding missing columns to the test set
fill_list = np.setdiff1d(categorical_train.columns, categorical_test.columns)
for item in fill_list:
    categorical_test[item]=0
categorical_test = categorical_test[categorical_train.columns]

#Combinig the features
X_test.drop(cat_col, axis=1, inplace=True)
X_test = pd.concat([X_test, categorical_test],axis=1)
X_test.head()

#dimensionality reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components= 0.99, random_state=40).fit(X_train)
print(f"The data has been reduced from {X_train.shape[1]} features to -> {len(pca.components_)} features")
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#Feature selection using RFE (Recursive Feature Elimination)
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
model = DecisionTreeClassifier()
rfe = RFECV(model, n_jobs=-1)
rfe.fit(X_train, Y_train)
new_features_count = len(rfe.estimator_.feature_importances_)
print(f"The number of features has been reduced from {X_train.shape[1]} to -> {new_features_count}")

X_train = rfe.transform(X_train)
X_test = rfe.transform(X_test)

#modelling SVM using 3 different kernels

# Fitting a model and saving the results
results = pd.DataFrame(columns=['Kernel Name','Train-set Score','Test-set Score','Recall','Precision','f1-score'])
#function to provide results which will be called everytime a model will be tested   
def fit_model_result(x_train, y_train, x_test, y_test, kernel_name, model):
    # Modelling
    reg = model
    reg.fit(x_train,y_train)
    
    # Getting evaluation results
    y_pred = reg.predict(x_test)
    report = metrics.classification_report(y_test, y_pred, digits=5, output_dict=True)
    recall = round(report['weighted avg']['recall']*100, 2)
    precision = round(report['weighted avg']['precision']*100, 2)
    f1 = round(report['weighted avg']['f1-score']*100,2)
    test_score = round(report['accuracy']*100,2)
    train_score = round(reg.score(x_train,y_train)*100,2)
    
    # Printing results
    results.loc[len(results.index)] = [kernel_name, train_score, test_score, recall, precision, f1]
    print(results.iloc[-1,:3])
    print(metrics.classification_report(y_test, y_pred, digits=5))
    
    # Displaying confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    metrics.plot_confusion_matrix(reg, x_test, y_test, cmap='Greens', normalize='true',ax=ax)
    plt.show()
    
#using Kernel: RBF
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.svm import SVC

param_grid = { 'C':loguniform(1,200),
             'gamma': loguniform(0.001,1)}
  
grid = RandomizedSearchCV(SVC(kernel = 'rbf'), param_grid,n_iter=10, n_jobs=-1, random_state=42)
grid.fit(X_train, Y_train)

best_C = grid.best_estimator_.C
best_gamma = grid.best_estimator_.gamma
print(f"Best C value found from Random search: {best_C}")
print(f"Best gamma value found from Random search: {best_gamma}")

from sklearn.svm import SVC
fit_model_result(X_train, Y_train, X_test, Y_test, 'RBF',SVC(kernel='rbf'))

#using Kernel:Linear
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.svm import LinearSVC

param_grid = { 'C':loguniform(0.01, 200)}
  
grid = RandomizedSearchCV(LinearSVC(), param_grid,n_iter=10, n_jobs=-1, random_state=42)
grid.fit(X_train, Y_train)
best_C = grid.best_estimator_.C
print(f"Best C value found from Random search: {best_C}")
fit_model_result(X_train, Y_train, X_test, Y_test, 'Linear', LinearSVC(C=best_C))

#using Kernel:Poly
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.svm import SVC

param_grid = { 'C':loguniform(0.01, 200),
             'degree' : [1,2,3,4,5]}
  
grid = RandomizedSearchCV(SVC(kernel='poly'), param_grid,n_iter=10, n_jobs=-1, random_state=42)
grid.fit(X_train, Y_train)
best_C = grid.best_estimator_.C
best_degree = grid.best_estimator_.degree
print(f"Best C value found from Random search: {best_C}")
print(f"Best degree value found from Random search: {best_degree}")
fit_model_result(X_train, Y_train, X_test, Y_test, 'Poly',SVC(kernel='poly', C=best_C, degree=best_degree))

#comparing the different Kernels
results.groupby('Kernel Name').first()
