## Football prediction
import pandas as pd
import datetime
#import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
#from skmultilearn.problem_transform import BinaryRelevance


## Split train and test data set
from sklearn.model_selection import train_test_split
def split_data(X,y):
    # test_size ideal 0.2 ; random_state can be any value
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    return X_train,X_test,y_train,y_test
############################################
    
match = pd.read_csv('football_results.csv') #whole dataset
data_team = pd.read_csv('team_skills.csv')
match['match_date'] = pd.to_datetime(match['match_date']) ## Convert datatype to date
match_filtered = match.loc[match['match_date'] > datetime.date(2013,1,1)] # only matches from 2016
country_array = data_team['Country'].values
match_filtered = match_filtered.loc[(match_filtered['home_team'].isin(country_array)) | (match_filtered['away_team'].isin(country_array))]
match_filtered = match_filtered.reset_index(drop=True)


match_filtered['winning_team'] = np.where(match_filtered['home_score'] > match_filtered['away_score'],'Home',np.where(match_filtered['home_score'] < match_filtered['away_score'],'Away','Draw'))

## Adding team skills to main data set
match_filtered['home_attack'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Attacking']
match_filtered['home_defense'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Defence']
match_filtered['home_mid'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Midfield']
match_filtered['home_overall'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Overall']

match_filtered['away_attack'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Attacking']
match_filtered['away_defense'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Defence']
match_filtered['away_mid'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Midfield']
match_filtered['away_overall'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Overall']
match_filtered['home_push'] = match_filtered['home_attack']*3+match_filtered['home_mid']-match_filtered['away_defense']*3
match_filtered['away_push'] = match_filtered['away_attack']*3+match_filtered['away_mid']-match_filtered['home_defense']*3



match_filtered = match_filtered.dropna()
match_filtered = match_filtered.reset_index()
#match_filtered = match_filtered.reset_index(drop=True)
#match_filtered['list_score'] = np.asarray(list(map(list, zip(match_filtered['home_score'], match_filtered['away_score']))))
#y = np.asarray(list(zip(match_filtered['home_score'], match_filtered['away_score'])))
# Step 1: Importing the dataset
X1 = match_filtered.iloc[:,[19,20]].values
X2 = match_filtered.iloc[:,[19,20]].values


#y_home = match_filtered.iloc[:,4].values
#y_away = match_filtered.iloc[:,5].values
#y = match_filtered.iloc[:,-1].values
y1 = match_filtered.iloc[:, 4].values
y2 = match_filtered.iloc[:, 5].values
#mlb = MultiLabelBinarizer()
#y = mlb.fit_transform(y)

corr_matrix = match_filtered.corr()

X_train1,X_test1,y_train1,y_test1 = split_data(X1,y1)
X_train2,X_test2,y_train2,y_test2 = split_data(X2,y2)
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.fit_transform(X_test1)
X_train2 = sc_X.fit_transform(X_train2)
X_test2 = sc_X.fit_transform(X_test2)


match_filtered.corr()

classifier1 = LogisticRegression()
classifier1.fit(X_train1,y_train1)

y_pred1 = classifier1.predict(X_test1)
score1 = accuracy_score(y_test1, y_pred1, normalize=True)


classifier2 = LogisticRegression()
classifier2.fit(X_train2,y_train2)

y_pred2 = classifier2.predict(X_test2)
score2 = accuracy_score(y_test2, y_pred2, normalize=True)


#y_pred = OneVsRestClassifier(SVC()).fit(X_train, y_train).predict(X_test)
#mlb.inverse_transform(y_pred)

data_fixture = pd.read_csv('fifa_2018.csv')
data_team = pd.read_csv('team_skills.csv')
data_fixture['home_attack'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Attacking']
data_fixture['home_defense'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Defence']
data_fixture['home_mid'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Midfield']
data_fixture['home_overall'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Overall']

data_fixture['away_attack'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Attacking']
data_fixture['away_defense'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Defence']
data_fixture['away_mid'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Midfield']
data_fixture['away_overall'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Overall']
data_fixture['home_push'] = data_fixture['home_attack']*3+data_fixture['home_mid']-data_fixture['away_defense']*3
data_fixture['away_push'] = data_fixture['away_attack']*3+data_fixture['away_mid']-data_fixture['home_defense']*3


X1 = data_fixture.iloc[:,[14,15]]
X2 = data_fixture.iloc[:, [14,15]]
X1 = X1.dropna()
X1 = X1.values
X2 = X2.dropna()
X2 = X2.values


X1 = sc_X.fit_transform(X1)
X2 = sc_X.fit_transform(X2)

y_pred1 = classifier1.predict(X1)
y_pred2 = classifier2.predict(X2)

