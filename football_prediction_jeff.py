## Football prediction
# Importing the libraries
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
########################################

## Split train and test data set
from sklearn.model_selection import train_test_split
def split_data(X,y):
    # test_size ideal 0.2 ; random_state can be any value
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    return X_train,X_test,y_train,y_test
############################################
# Function which predicts score and check accuracy of different models
def predict_score(X_train,X_test): 
    ## Fitting model to training set
    classifier_rf = RandomForestClassifier(n_estimators = 10, criterion ='entropy' , random_state = 0)
    classifier_rf.fit(X_train, y_train)
    classifier_lr = LogisticRegression(random_state = 0)
    classifier_lr.fit(X_train, y_train)
    classifier_svc = SVC(kernel = 'rbf', random_state = 0)
    classifier_svc.fit(X_train, y_train)
    classifier_nb = GaussianNB()
    classifier_nb.fit(X_train, y_train)
    ## Predict the result
    y_pred_rf = classifier_rf.predict(X_test)
    score_rf = accuracy_score(y_test, y_pred_rf, normalize=True)
    y_pred_lr = classifier_lr.predict(X_test)
    score_lr = accuracy_score(y_test, y_pred_lr, normalize=True)
    y_pred_svc = classifier_svc.predict(X_test)
    score_svc = accuracy_score(y_test, y_pred_svc, normalize=True)
    y_pred_nb = classifier_nb.predict(X_test)
    score_nb = accuracy_score(y_test, y_pred_nb, normalize=True)
    return y_pred_rf,score_rf,y_pred_lr,score_lr,y_pred_svc,score_svc,y_pred_nb,score_nb,classifier_rf,classifier_lr,classifier_svc,classifier_nb
    #return y_pred,score,classifier
##############################################

### Step 1: Data preparation
match = pd.read_csv('football_results.csv') #whole dataset
data_team = pd.read_csv('team_skills.csv')
country_code = pd.read_csv('country_code.csv') # to fetch 3 letter ISO code
country_code = country_code[['name','alpha-3']]
country_code = dict(zip(country_code['name'],country_code['alpha-3']))
data_team = data_team.replace({"Country": country_code})
match = match.replace({"home_team": country_code})
match = match.replace({"away_team": country_code})
match['match_date'] = pd.to_datetime(match['match_date']) ## Convert datatype to date
match_filtered = match.loc[match['match_date'] > datetime.date(1900,1,1)] # only matches from 2010
country_array = data_team['Country'].values
match_filtered = match_filtered.loc[(match_filtered['home_team'].isin(country_array)) | (match_filtered['away_team'].isin(country_array))]
match_filtered = match_filtered.reset_index(drop=True)
for i,row in match_filtered.iterrows():
    print (i)
    l = sorted([row['home_team'],row['away_team']])
    if row['home_team'] == l[1]:
        match_filtered.loc[i,['home_team','away_team']] = match_filtered.loc[i,['away_team','home_team']].values
        match_filtered.loc[i,['home_score','away_score']] = match_filtered.loc[i,['away_score','home_score']].values
match_filtered['winning_team'] = np.where(match_filtered['home_score'] > match_filtered['away_score'],'Home',np.where(match_filtered['home_score'] < match_filtered['away_score'],'Away','Draw'))
match_filtered = match_filtered.drop(columns = ['tournament','city','country','neutral'])
## Adding team skills to main data set
match_filtered['home_attack'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Attacking']
match_filtered['home_defense'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Defence']
match_filtered['home_mid'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Midfield']
match_filtered['home_overall'] = match_filtered.merge(data_team,left_on = 'home_team',right_on ='Country',how = 'left')['Overall']
match_filtered['away_attack'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Attacking']
match_filtered['away_defense'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Defence']
match_filtered['away_mid'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Midfield']
match_filtered['away_overall'] = match_filtered.merge(data_team,left_on = 'away_team',right_on ='Country',how = 'left')['Overall']
match_filtered = match_filtered.dropna()
tmp_df = match_filtered
tmp_df = tmp_df.reset_index(drop=True)
#####################################
## Calculate win streak, head to head stats from the basic data set###########
from collections import defaultdict
dictt_w = defaultdict(int)
dictt_l = defaultdict(int)
head_to_head = defaultdict(lambda: [0,0,0,0,0,0])
home_win_streak = []
away_win_streak = []
home_lose_streak = []
away_lose_streak = []
value_list = []
#m2 = tmp_df.loc[((tmp_df['home_team'] == 'ENG') | (tmp_df['away_team'] == 'ENG')) & ((tmp_df['home_team'] == 'DEU') | (tmp_df['away_team'] == 'DEU'))]
#m2 = m2.reset_index(drop=True)
m2 = tmp_df
for i, row in m2.iterrows():
    home_win_streak.append(dictt_w[row['home_team']])
    away_win_streak.append(dictt_w[row['away_team']])
    home_lose_streak.append(dictt_l[row['home_team']])
    away_lose_streak.append(dictt_l[row['away_team']])
    l = sorted([row['home_team'],row['away_team']])
    key = l[0]+'_'+l[1]
    value_list.append(head_to_head[key])
    if row['winning_team'] == 'Home':
        dictt_w[row['home_team']] += 1
        dictt_w[row['away_team']] = 0
        dictt_l[row['away_team']] += 1
        dictt_l[row['home_team']] = 0
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] + 1
        v3 = head_to_head[key][2]
        v4 = head_to_head[key][3] 
        v5 = head_to_head[key][4] + row['home_score']
        v6 = head_to_head[key][5] + row['away_score'] 
    elif row['winning_team'] == 'Away':
        dictt_w[row['away_team']] += 1
        dictt_w[row['home_team']] = 0
        dictt_l[row['home_team']] += 1
        dictt_l[row['away_team']] += 0
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] 
        v3 = head_to_head[key][2] + 1
        v4 = head_to_head[key][3] 
        v5 = head_to_head[key][4] + row['home_score']
        v6 = head_to_head[key][5] + row['away_score']
    else: # Draw
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] 
        v3 = head_to_head[key][2]
        v4 = head_to_head[key][3] + 1
        v5 = head_to_head[key][4] + row['home_score']
        v6 = head_to_head[key][5] + row['away_score']
       
    head_to_head[key] = [v1,v2,v3,v4,v5,v6]
m2['home_win_streak']= home_win_streak
m2['away_win_streak']= away_win_streak
m2['home_lose_streak']= home_lose_streak
m2['away_lose_streak']= away_lose_streak#
#m1 = match_filtered.loc[:,['match_date','home_team','away_team', 'home_score' , 'away_score', 'winning_team', 'home_win_streak', 'away_win_streak','home_lose_streak','away_lose_streak']] 
temp_df = pd.concat([m2, pd.DataFrame(value_list, columns=['h2h_matches','h2h_home_win','h2h_away_win','h2h_tie','h2h_home_goals','h2h_away_goals'])], axis=1)
#temp_df = temp_df.drop(columns = ['index','tournament','city','country','neutral']) # dropping unwanted columns
#m2 = temp_df.loc[((temp_df['home_team'] == 'ENG') | (temp_df['away_team'] == 'ENG')) & ((temp_df['home_team'] == 'DEU') | (temp_df['away_team'] == 'DEU'))]
##########################################################################
##########################################################################

### Prediciton to be worked... pending
# Step 1: Importing the dataset
X = match_filtered.iloc[:, 11:19].values # :2 to make it a matrix
y_home = match_filtered.iloc[:,4].values
y_away = match_filtered.iloc[:,5].values

#y = match_filtered.iloc[:,10].values

X_train,X_test,y_train,y_test = split_data(X,y)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#X_train,X_test,y_train,y_test = split_data(X,y_away)
y_pred_rf,score_rf,y_pred_lr,score_lr,y_pred_svc,score_svc,y_pred_nb,score_nb,classifier_rf,classifier_lr,classifier_svc,classifier_nb = predict_score(X_train,X_test)


#Import WC fixture
data_fixture = pd.read_csv('fifa_2018.csv')

## Adding team skills to main data set
data_fixture['home_attack'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Attacking']
data_fixture['home_defense'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Defence']
data_fixture['home_mid'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Midfield']
data_fixture['home_overall'] = data_fixture.merge(data_team,left_on = 'Home Team',right_on ='Country',how = 'left')['Overall']

data_fixture['away_attack'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Attacking']
data_fixture['away_defense'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Defence']
data_fixture['away_mid'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Midfield']
data_fixture['away_overall'] = data_fixture.merge(data_team,left_on = 'Away Team',right_on ='Country',how = 'left')['Overall']
X = data_fixture.iloc[:, 6:14]
X = X.dropna()
X = X.values



sc_X = StandardScaler()
X = sc_X.fit_transform(X)

winning_team = classifier_lr.predict(X)
winning_team = pd.DataFrame(winning_team,columns=['winning_team'])
winning_team['index'] = winning_team.index
data_fixture['index'] = data_fixture.index


first_round_prediction = data_fixture[['index','Home Team','Away Team']].merge(winning_team,on='index',how='inner')
first_round_prediction = first_round_prediction.drop('index',axis = 1)

first_round_prediction.to_csv('first_round_prediction.csv',index=False)


