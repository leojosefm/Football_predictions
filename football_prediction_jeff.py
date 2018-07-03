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
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)
    return X_train,X_test,y_train,y_test
############################################
# Function which predicts score and check accuracy of different models
def predict_score(X_train,X_test,y_train,y_test): 
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
data_team = pd.read_csv('sofifa_team_list.csv')
team_wc = pd.read_csv('team_2018.csv')
year_list = data_team.year_list.unique()
missing_countries = team_wc[team_wc.Country.isin(data_team.countries) == False]['Country']
## Fill missing countries to data_team using min values ###
attack_min = data_team["attack"].min()
mid_min = data_team["mid"].min()
defence_min = data_team["defence"].min()
overall_min = data_team["overall"].min()
for year in year_list:
    for country in missing_countries:
        data_team = data_team.append({'year_list':year,'countries':country,'overall':overall_min,'attack':attack_min,'mid':mid_min,'defence':defence_min},ignore_index=True)
#############################################
## Fill missing year information with mean value
for year in year_list:
    for country in data_team.countries.unique():
        if ((data_team['year_list'] == year) & (data_team['countries'] == country)).any() == False:
            attack = data_team.groupby(['countries'])['attack'].mean()
            mid = data_team.groupby(['countries'])['mid'].mean()
            defence = data_team.groupby(['countries'])['defence'].mean()
            overall = data_team.groupby(['countries'])['overall'].mean()
            data_team = data_team.append({'year_list':year,'countries':country,'overall':overall[country],'attack':attack[country],'mid':mid[country],'defence':defence[country]},ignore_index=True)

#country_code = pd.read_csv('country_code.csv') # to fetch 3 letter ISO code
#country_code = country_code[['name','alpha-3']]
#country_code = dict(zip(country_code['name'],country_code['alpha-3']))
#data_team = data_team.replace({"Country": country_code})
#match = match.replace({"home_team": country_code})
#match = match.replace({"away_team": country_code})
match['match_date'] = pd.to_datetime(match['match_date']) ## Convert datatype to date
match_filtered = match.loc[match['match_date'] > datetime.datetime(1900,1,1)] # only matches from 2010
country_array = data_team['countries'].values
#match_filtered = match_filtered.loc[(match_filtered['home_team'].isin(country_array)) | (match_filtered['away_team'].isin(country_array))]
match_filtered = match_filtered.reset_index(drop=True)
#for i,row in match_filtered.iterrows():
#    l = sorted([row['home_team'],row['away_team']])
#    if row['home_team'] == l[1]:
#        match_filtered.loc[i,['home_team','away_team']] = match_filtered.loc[i,['away_team','home_team']].values
#        match_filtered.loc[i,['home_score','away_score']] = match_filtered.loc[i,['away_score','home_score']].values

match_filtered = match_filtered.drop(columns = ['tournament','city','country','neutral'])
match_filtered = match_filtered.rename(index=str, columns={"home_team": "team_A", "away_team": "team_B","home_score":"A_score","away_score":"B_score"})
match_filtered['year'] = match_filtered.match_date.dt.to_period('Y')
match_filtered['year'] = match_filtered['year'].astype(str).astype(int)
## Switichin home and away team as per ascending order in order to find head to head stats easier
for i,row in match_filtered.iterrows():
    l = sorted([row['team_A'],row['team_B']])
    if row['team_A'] == l[1]:
        match_filtered.loc[i,['team_A','team_B']] = match_filtered.loc[i,['team_B','team_A']].values
        match_filtered.loc[i,['A_score','B_score']] = match_filtered.loc[i,['B_score','A_score']].values
match_filtered['winning_team'] = np.where(match_filtered['A_score'] > match_filtered['B_score'],'team_A',np.where(match_filtered['A_score'] < match_filtered['B_score'],'team_B','Draw'))
#######################################################################
## Adding team skills to main data set
match_filtered['A_attack'] = match_filtered.merge(data_team,left_on = ['team_A','year'],right_on =['countries','year_list'],how = 'left')['attack']
match_filtered['A_defense'] = match_filtered.merge(data_team,left_on = ['team_A','year'],right_on =['countries','year_list'],how = 'left')['defence']
match_filtered['A_mid'] = match_filtered.merge(data_team,left_on = ['team_A','year'],right_on =['countries','year_list'],how = 'left')['mid']
match_filtered['A_overall'] = match_filtered.merge(data_team,left_on = ['team_A','year'],right_on =['countries','year_list'],how = 'left')['overall']
match_filtered['B_attack'] = match_filtered.merge(data_team,left_on = ['team_B','year'],right_on =['countries','year_list'],how = 'left')['attack']
match_filtered['B_defense'] = match_filtered.merge(data_team,left_on = ['team_B','year'],right_on =['countries','year_list'],how = 'left')['defence']
match_filtered['B_mid'] = match_filtered.merge(data_team,left_on = ['team_B','year'],right_on =['countries','year_list'],how = 'left')['mid']
match_filtered['B_overall'] = match_filtered.merge(data_team,left_on = ['team_B','year'],right_on =['countries','year_list'],how = 'left')['overall']
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
m2 = tmp_df
for i, row in m2.iterrows():
    home_win_streak.append(dictt_w[row['team_A']])
    away_win_streak.append(dictt_w[row['team_B']])
    home_lose_streak.append(dictt_l[row['team_A']])
    away_lose_streak.append(dictt_l[row['team_B']])
    l = sorted([row['team_A'],row['team_B']])
    key = l[0]+'_'+l[1]
    value_list.append(head_to_head[key])
    if row['winning_team'] == 'team_A':
        dictt_w[row['team_A']] += 1
        dictt_w[row['team_B']] = 0
        dictt_l[row['team_B']] += 1
        dictt_l[row['team_A']] = 0
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] + 1
        v3 = head_to_head[key][2]
        v4 = head_to_head[key][3] 
        v5 = head_to_head[key][4] + row['A_score']
        v6 = head_to_head[key][5] + row['B_score'] 
    elif row['winning_team'] == 'team_B':
        dictt_w[row['team_B']] += 1
        dictt_w[row['team_A']] = 0
        dictt_l[row['team_A']] += 1
        dictt_l[row['team_B']] = 0
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] 
        v3 = head_to_head[key][2] + 1
        v4 = head_to_head[key][3] 
        v5 = head_to_head[key][4] + row['A_score']
        v6 = head_to_head[key][5] + row['B_score']
    else: # Draw
        v1 = head_to_head[key][0] + 1
        v2 = head_to_head[key][1] 
        v3 = head_to_head[key][2]
        v4 = head_to_head[key][3] + 1
        v5 = head_to_head[key][4] + row['A_score']
        v6 = head_to_head[key][5] + row['B_score']
       
    head_to_head[key] = [v1,v2,v3,v4,v5,v6]
m2['A_win_streak']= home_win_streak
m2['B_win_streak']= away_win_streak
m2['A_lose_streak']= home_lose_streak
m2['B_lose_streak']= away_lose_streak#
#m1 = match_filtered.loc[:,['match_date','home_team','away_team', 'home_score' , 'away_score', 'winning_team', 'home_win_streak', 'away_win_streak','home_lose_streak','away_lose_streak']] 
temp_df = pd.concat([m2, pd.DataFrame(value_list, columns=['h2h_matches','h2h_A_win','h2h_B_win','h2h_tie','h2h_A_goals','h2h_B_goals'])], axis=1)
#m2 = temp_df.loc[((temp_df['team_A'] == 'England') & (temp_df['team_B'] == 'Germany'))]
temp_df = temp_df.dropna()
temp_df = temp_df.reset_index(drop=True)
##########################################################################
##########################################################################

### Prediciton to be worked... pending
# Step 1: Importing the dataset
X = temp_df.iloc[:, 7:25].values # :2 to make it a matrix
y_home = temp_df.iloc[:,3].values
y_away = temp_df.iloc[:,4].values

#y = match_filtered.iloc[:,10].values

X_train,X_test,y_train_home,y_test_home = split_data(X,y_home)
X_train,X_test,y_train_away,y_test_away = split_data(X,y_away)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#corr_matrix = temp_df.corr()
#X_train,X_test,y_train,y_test = split_data(X,y_away)
y_pred_rf,score_rf,y_pred_lr,score_lr,y_pred_svc,score_svc,y_pred_nb,score_nb,classifier_rf,classifier_lr,classifier_svc,classifier_nb = predict_score(X_train,X_test,y_train_home,y_test_home)
y_pred_rf,score_rf,y_pred_lr,score_lr,y_pred_svc,score_svc,y_pred_nb,score_nb,classifier_rf,classifier_lr,classifier_svc,classifier_nb = predict_score(X_train,X_test,y_train_away,y_test_away)

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


