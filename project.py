#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
trainData = ''
testData = ''
columnsAfterCleaning = ['age', 'wage_per_hour', 'capital_gains', 'capital_losses', 'dividend_from_Stocks', 
                        'num_person_Worked_employer', 'weeks_worked_in_year', 'class_of_worker', 
                        'industry_code', 'occupation_code', 'education', 'enrolled_in_edu_inst_lastwk', 
                        'marital_status', 'major_industry_code', 'major_occupation_code', 'race', 
                        'hispanic_origin', 'sex', 'member_of_labor_union', 'reason_for_unemployment', 
                        'full_parttime_employment_stat', 'tax_filer_status', 'region_of_previous_residence', 
                        'state_of_previous_residence', 'd_household_family_stat', 'd_household_summary', 
                        'live_1_year_ago', 'family_members_under_18', 'country_father', 'country_mother', 
                        'country_self', 'citizenship', 'business_or_self_employed', 'fill_questionnaire_veteran_admin',
                        'year', 'veterans_benefits', 'income_level']
factorCols =  ['class_of_worker','industry_code','occupation_code','education','enrolled_in_edu_inst_lastwk',
               'marital_status','major_industry_code','major_occupation_code','race','hispanic_origin','sex',
               'member_of_labor_union','reason_for_unemployment','full_parttime_employment_stat',
               'tax_filer_status','region_of_previous_residence','state_of_previous_residence','d_household_family_stat',
               'd_household_summary','migration_msa','migration_reg','migration_within_reg','live_1_year_ago','migration_sunbelt',
               'family_members_under_18','country_father','country_mother','country_self',
               'citizenship','business_or_self_employed','fill_questionnaire_veteran_admin',
               'year','veterans_benefits','income_level'] #33/41 Cols
            
numCols = ['age','wage_per_hour','capital_gains','capital_losses','dividend_from_Stocks','num_person_Worked_employer','weeks_worked_in_year']

def loadData():
    trainPath='/Users/hang/Documents/rachel/machine learning/project/data/Census Income/train 2.csv'
    testPath='/Users/hang/Documents/rachel/machine learning/project/data/Census Income/test 2.csv'
    trainData = pd.read_csv(trainPath);
    testData = pd.read_csv(testPath);
    print(trainData.head())
    print(testData.head())
    return trainData, testData
def checkData():
    #check missing value
    check_missing(trainData)
    check_missing(testData)
    #check duplicate
    #no duplicated recoreds
    #check datatype
    print(trainData.dtypes)
    print(testData.dtypes)
    
    #check target variable
    check_target_value()
    #check imbalanced
    check_balance(trainData)    
    check_balance(testData)
    return

def check_missing(data):
    print(data.isnull().sum())
    return

def check_target_value():
    le = preprocessing.LabelEncoder()
    le.fit(trainData['income_level'])
    print(list(le.classes_))
    le.fit(testData['income_level'])
    print(list(le.classes_))
    return

def check_balance(dataFrame):
    print(dataFrame['income_level'].value_counts())
    return

def data_exploration(inputData):
    inputData = trainData
    train_cat, train_num = seperateCatNumCols(inputData)
    
    ##########Numerical Data Exploration#######
    #sns.pairplot(trainData);
    sns.set(rc={"figure.figsize": (10,5)})
    #np.random.seed(sum(map(ord, "palettes")))
    #cmap= sns.palplot(sns.color_palette("hls", 8))
    sns.distplot(train_num['age'],color="g") #Earning class is from 0-90
    sns.distplot(train_num['wage_per_hour'],color="g")
    sns.distplot(train_num['capital_gains'] ,color="g")
    sns.distplot(train_num['capital_losses'],color="g")#Highly skewed
    sns.distplot(train_num['dividend_from_Stocks'],color="g")
    sns.distplot(train_num['num_person_Worked_employer'],color="g")
    sns.distplot(train_num['weeks_worked_in_year'],color="g")
    
    #add income to numerical data for better analysis
    train_num =  pd.concat([train_num,train_cat[['income_level']]],axis=1)
    sns.swarmplot(x='age' ,y='wage_per_hour', hue='income_level', data=train_num,palette="Set2")
    
    
    ##########Categorical Data Exploration#######
    classOfWorker_income = sns.countplot(x="class_of_worker", hue="income_level", data=train_cat,palette="Set2")
    classOfWorker_income.figure.set_size_inches(12,4)
    classOfWorker_income.set_xticklabels(classOfWorker_income.get_xticklabels(), rotation=20)
    
    education_income = sns.countplot(x="education", hue="income_level", data=train_cat,palette="Set2")
    education_income.figure.set_size_inches(10,5)
    education_income.set_xticklabels(education_income.get_xticklabels(), rotation=30)
    return
    
def encode_target():
    trainData['income_level'] = [0 if income==-50000 else 1 for income in trainData.income_level]
    testData['income_level'] = [0 if income=='-50000' else 1 for income in testData.income_level]
    return

def typeConversionFactor(data, col):
    #converting datatypes for the categorical columns
    data[col] = data[col].astype(object)    
    return

def encodeCategoricalData(df,column):
    le = preprocessing.LabelEncoder()
    df[column]=le.fit_transform(df[column].astype(str)) 
    return

####separating categorical variables & numerical variables
def seperateCatNumCols(data):    
    data_cat = data[factorCols]
    data_num = data[numCols]
    
    return data_cat,data_num

def combineCategories(data,col):
    for val, cnt in data[col].value_counts().iteritems():
        #print ('value', val, 'was found', cnt, 'times')
        if(cnt/data.shape[0]*100 < 5):
            data[col]=data[col].replace(val, 'Others')
    return

def binningToGroups(df,column,bins,groupNames):
    df[column] = pd.cut(df[column], bins, labels = groupNames)
    return

def checkGtZero(val):
    if(val==0):
        return 'Zero'
    else:
        return 'MoreThanZero'

#remove columns with more than 5% missing data
#and fill with "Unavailable" for the rest of columns with less than 5% missing data
def handle_missing_data(trainData, testData):
    cat_data, num_data = seperateCatNumCols(trainData)
    test_cat, test_num = seperateCatNumCols(testData)
    #remove columns with more than 5% missing data
    cat_data = cat_data.ix[:,cat_data.isnull().mean() <= 0.05]
    #impute missing data with "Unavailable"
    cat_data = cat_data.fillna('Unavailable')
    #drop the removed columns in test dataset as well
    new_factor_cols = list(cat_data)
    diffList = np.setdiff1d(factorCols,new_factor_cols)
    test_cat = test_cat.drop(diffList,axis=1)
    test_cat = test_cat.fillna('Unavailable')
    #combine categories with less than 5% missing data
    for col in new_factor_cols:
        combineCategories(cat_data,col)
        combineCategories(test_cat,col)
    #BINNING THE NUMERICAL COLUMNS
    #bin age to 0-30 31-60 61 - 90
    age_bins = [0, 30, 60, 90]
    age_groupNames = ["young","adult","old"]
    binningToGroups(num_data,'age',age_bins,age_groupNames) 
    binningToGroups(test_num,'age',age_bins,age_groupNames)
    #int->object
    typeConversionFactor(num_data,'age')
    typeConversionFactor(test_num,'age')
    #string-int
    encodeCategoricalData(num_data,'age')
    encodeCategoricalData(test_num,'age') 
    #bin numeric variables with Zero and MoreThanZero
    #because more than 70-80% of the observations are 0
    num_bin_list = ['wage_per_hour', 'capital_gains', 'capital_losses','dividend_from_Stocks']
    for column in num_bin_list:
        num_data[column]=num_data[column].apply(lambda x: checkGtZero(x))
        test_num[column]=test_num[column].apply(lambda x: checkGtZero(x))
        #after binning, int type changed to object, so encode to int type again
        encodeCategoricalData(num_data,column)
        encodeCategoricalData(test_num,column)
    #print(num_data.head())
    #ENCODE CATEGORICAL COLUMNS
    for column in new_factor_cols:
        encodeCategoricalData(cat_data,column)
        encodeCategoricalData(test_cat,column)
    #merge numerical and categorical columns
    trainData = pd.concat([num_data,cat_data], axis=1)
    testData = pd.concat([test_num,test_cat], axis=1)
    return trainData, testData

def cleaning_data(trainData, testData):
    #encode target variable as 0 and 1
    encode_target()
    #change datatype
    for col in factorCols:
       typeConversionFactor(trainData,col)
       typeConversionFactor(testData,col)
    #print(trainData.info()) 
    #handle missing data-only on categrical
    trainData, testData = handle_missing_data(trainData, testData)
    #print("after handle missing data")
    #print(trainData.head())
    #remove duplicate-no duplicate    
    return trainData, testData


#refer:https://elitedatascience.com/imbalanced-classes
#https://beckernick.github.io/oversampling-modeling/
def performace_of_imbalanced_classes(trainData, testData):
    y = trainData.income_level
    X = trainData.drop('income_level',axis=1)
    y_te = testData.income_level
    X_te = testData.drop('income_level',axis=1)
    model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING']
    for model_name in model_names:
        if(model_name == 'LOGISTIC REGRESSION'):
            model = LogisticRegression()
        elif (model_name == 'RANDOM FORESTS'):
            model = RandomForestClassifier(n_estimators = 100, max_features="auto", min_samples_split=2)
        else:
            model = AdaBoostClassifier(n_estimators = 50, learning_rate =1.0)
            
    model.fit(X, y)
    y_pred = model.predict(X_te)

    print("Accuracy", accuracy_score(y_te, y_pred))
    print("AUC", roc_auc_score(y_te, y_pred))
    print(np.unique(y_pred))
    return

def up_sampling(majority, minority):
    minority = resample(minority,replace=True, n_samples=len(majority),random_state=123)
    total = pd.concat([majority, minority])
    check_balance(total)
    y = total.income_level
    X = total.drop('income_level',axis=1)
    return X, y, total

def down_sampling(majority, minority):
    majority = resample(majority,replace=False, n_samples=len(minority), random_state=123)
    total = pd.concat([majority, minority])
    check_balance(total)
    y = total.income_level
    X = total.drop('income_level',axis=1)
    return X, y, total

def up_sampling_SMOTE():
    y_t = trainData.income_level
    X_t = trainData.drop('income_level',axis=1)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X_t, y_t)
    print(Counter(y_res)) 
  
    y_res0 = np.array([y_res])
    total = np.concatenate((X_res,y_res0.T), axis=1)
    
    total = pd.DataFrame(data=total, columns=columnsAfterCleaning)
    return X_res, y_res, total

#method: 1:up_sample, 2:down_sample, 3:smote_sample
def handle_imbalanced_data(trainData,method):
    majority = trainData[trainData.income_level==0]
    minority = trainData[trainData.income_level==1]
    if method == 1:
        return up_sampling(majority, minority)
    elif method == 2:
        return down_sampling(majority, minority)
    else:
        return up_sampling_SMOTE()
    
#feature selection
def feature_selection_FSS(X,y, trainData):
    features = columnsAfterCleaning[0:-1]
    feature_cols = features
    logreg = LogisticRegression()
    max_auc = -1
    max_auc_label = ''
    selected = []
    # Step 1: Select the first variable
    for f in feature_cols:
        X = trainData[f]
        X = X.values.reshape(-1, 1)
        average_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
        if(average_auc > max_auc):
            max_auc = average_auc
            max_auc_label = f;
    print(max_auc_label)
    print(max_auc)
    selected.append(max_auc_label)
    features = np.delete(features, features.index(max_auc_label)).tolist()
    
    #in each iteration, select one feature
    while(len(feature_cols) > 0):   
        max_auc = -1
        for f in feature_cols:
            X = trainData[np.concatenate((selected, [f]))]
            average_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
            if(average_auc > max_auc):
                max_auc = average_auc
                max_auc_label = f;
        
        print(max_auc_label)
        print(max_auc)
        selected.append(max_auc_label)
        features = np.delete(features, features.index(max_auc_label)).tolist()
    return

def feature_selection_BSS(X, y,trainData):
    features = columnsAfterCleaning[0:-1]
    feature_cols = features
    logreg = LogisticRegression()
    max_auc = -1
    max_auc_label = ''
   
    #in each iteration, remove one feature with the max AUC
    while(len(feature_cols) > 1):   
        max_auc = -1
        for f in feature_cols:
            temp_f = feature_cols.copy()
            X = trainData[np.delete(temp_f, temp_f.index(f))]
            average_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
            if(average_auc > max_auc):
                max_auc = average_auc
                max_auc_label = f
        
        print(max_auc_label)
        print(max_auc)
        features = np.delete(features, features.index(max_auc_label)).tolist()
    #one feature left
    X = trainData[feature_cols[0]]
    X = X.values.reshape(-1, 1)
    average_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
    print(feature_cols[0])
    print(average_auc)
    return
def random_forest_feature_selection(X, y):
    rf = RandomForestClassifier(n_estimators = 10, max_features="auto", min_samples_split=2)
    rf.fit(X, y)
    print(rf.feature_importances_.reshape(-1, 1))
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    feature_cols = columnsAfterCleaning[0:-1]
    for f in range(X.shape[1]):
        print(feature_cols[indices[f]], importances[indices[f]])
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return

def random_forest_for_max_feature(X, y):
    max_features_range = np.arange(1, 7, 1)
    scores = []
    max_auc = -1
    for num_features in max_features_range:
        rf = RandomForestClassifier(n_estimators = 10, max_features=num_features, min_samples_split=2)
        cur_auc = cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean()
        scores.append(cur_auc)
        if(cur_auc > max_auc):
            max_auc = cur_auc
    
    print(max_auc)
    # Plot AUC for different values of B
    plt.plot(max_features_range, scores)
    plt.xlim([0.0, 7])
    plt.title('Random Foresct for max features')
    plt.xlabel('max_features')
    plt.ylabel('AUC')
    plt.grid(True)
    return

def boosting_feature_selection(X, y):
    boost = AdaBoostClassifier(n_estimators = 50, learning_rate = 1)
    boost.fit(X, y)
    test_auc = cross_val_score(boost, X, y, cv=5, scoring='roc_auc').mean()
    print("Test AUC:", test_auc)
    print(boost.feature_importances_)
    
    importances = boost.feature_importances_
    std = np.std([tree.feature_importances_ for tree in boost.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    feature_cols = columnsAfterCleaning[0:-1]
    for f in range(X.shape[1]):
        print(feature_cols[indices[f]], importances[indices[f]])
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return

#model validation
def kfold_cross_validation(X, y):
    #X = loanData[feature_cols]
    #y = loanData.Loan_Status
    model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING']
    for model_name in model_names:
        if(model_name == 'LOGISTIC REGRESSION'):
            model = LogisticRegression()
        elif(model_name == 'RANDOM FORESTS'):
            model = RandomForestClassifier(n_estimators = 10, max_features='auto', min_samples_split=2)
        else:
            model = AdaBoostClassifier(n_estimators = 50, learning_rate =1.0)
        a = cross_val_score(model, X, y, cv=10, scoring='roc_auc')        
        mean_score = a.mean()
        print("AUC by 5-fold cross validation:",model_name, mean_score)
        
        print('CV %s AUC_mean: %.4f' % (model_name, mean_score))   
        
        plt.plot(a,label='%s (AUC = %0.3f)' % (model_name,mean_score))
        plt.legend(loc='lower right')
        plt.xlim([0.0, 10])
        plt.title('CV for three models')
        plt.xlabel('Iterations')
        plt.ylabel('AUC')
        plt.grid(True)
    return

def bootstrap(df):    
    # configure bootstrap
    n_iterations = 100
    values = df.values
    model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING']
    #model_names = ['LOGISTIC REGRESSION']
    for model_name in model_names:
        if(model_name == 'LOGISTIC REGRESSION'):
            model = LogisticRegression()
        elif(model_name == 'RANDOM FORESTS'):
            model = RandomForestClassifier(n_estimators = 100, max_features='auto', min_samples_split=2)
        else:
            model = AdaBoostClassifier(n_estimators = 50, learning_rate =1.0)
        # run bootstrap
        stats = list()
        auc_stats = list()
        for i in range(n_iterations):
           #prepare train and test sets
           train_bs = resample(values, n_samples=100)
           test_bs = np.array([x for x in values if x.tolist() not in train_bs.tolist()])
           
           y = train_bs[:,-1]
           X = train_bs[:,:-1]
           y_te = test_bs[:,-1]
           X_te = test_bs[:,:-1]
               
           #fit model       
           model = model.fit(X,y) 
        	
           # evaluate model
           pred = model.predict(X_te)
           #y_pred_prob = model.predict_proba(X_te)[:, 1]
           score = accuracy_score(y_te, pred)
           stats.append(score)
           auc =  roc_auc_score(y_te, pred)
           auc_stats.append(auc)
           print('Accuracy:',score,' AUC:',auc)
        
        logreg_auc_mean = np.mean(auc_stats) 
        logreg_accuracy_mean = np.mean(stats) 
        
        print('Bootstrap %s AUC_mean: %.4f' % (model_name, logreg_auc_mean))   
        print('Bootstrap %s Accuracy_mean: %.4f' % (model_name, logreg_accuracy_mean))
        
        plt.plot(auc_stats,label='%s (AUC = %0.3f)' % (model_name,logreg_auc_mean))
        plt.legend(loc='lower right')
        plt.xlim([0.0, 100])
        plt.title('Bootstrap for three models')
        plt.xlabel('Iterations')
        plt.ylabel('AUC')
        plt.grid(True)
    return

def normalization(data, column):
    data[column] = preprocessing.scale(np.sqrt(data[column]))
    plt.hist(data[column])
    plt.xlabel(column);
    plt.ylabel('Frequency');
    return data

def normalizeData(trainData):
    normalized_data = normalization(trainData, 'num_person_Worked_employer')
    normalized_data = normalization(normalized_data, 'weeks_worked_in_year')
    return normalized_data
    

def model_roc_curve(X,y,testData): 
    #model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING','SVM',"BAGGING"]
    model_names = ['LOGISTIC REGRESSION','RANDOM FORESTS','BOOSTING',"BAGGING"]
    for model_name in model_names:        
        if(model_name == 'LOGISTIC REGRESSION'):
            model = LogisticRegression()
        elif(model_name == 'RANDOM FORESTS'):
            model = RandomForestClassifier(n_estimators = 100, max_features='auto', min_samples_split=2)
        elif(model_name == 'BOOSTING'):
            model = AdaBoostClassifier(n_estimators = 100, learning_rate =1.0)
        elif(model_name == 'SVM'):
            model = SVC(kernel='linear', class_weight='balanced', probability=True)
        else:
            model = BaggingClassifier(n_estimators = 100)
        
        y_te = testData.income_level
        X_te = testData.drop('income_level',axis=1)
        model.fit(X, y)
        
        y_pred = model.predict(X_te)
        # store the predicted probabilities for class 1
        y_pred_prob = model.predict_proba(X_te)[:, 1]
        
        print('---------------  %s  ------------------' % model_name)           
        '''
        AUC is the percentage of the ROC plot that is underneath the curve:
        '''
        print()
        print('AUC FOR ', model_name , roc_auc_score(y_te, y_pred))    
        '''
        ROC Curves and Area Under the Curve (AUC)
        '''
        print()            
        # IMPORTANT: first argument is true values, second argument is predicted probabilities
        fpr, tpr, thresholds = roc_curve(y_te, y_pred)
        
        plt.plot(fpr, tpr, label='%s (AUC = %0.3f)' % (model_name,roc_auc_score(y_te, y_pred)))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC curves for 4 Models')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')    
        plt.legend()
        plt.grid(True)
    return

###############function call#####################
trainData, testData = loadData()
checkData()
data_exploration(trainData)
trainData, testData = cleaning_data(trainData, testData)
trainData = normalizeData(trainData)
testData = normalizeData(testData)
performace_of_imbalanced_classes(trainData, testData)
X, y, trainData = handle_imbalanced_data(trainData, 3)

random_forest_for_max_feature(X, y)
feature_selection_FSS(X,y, trainData)
random_forest_feature_selection(X, y)
boosting_feature_selection(X, y)

kfold_cross_validation(X, y)
bootstrap(trainData)
model_roc_curve(X,y,testData)

