# Football_predictions
## Welcome !
This is a project created by me and my friend ebbygeorge (https://github.com/ebbygeorge) as part of self learning Machine Learning and Python. This is out first project in Python and also in Machine Learning, so we appreciate all suggestions :)
We have a dataset of all international matches played since 1872. The project fetches additional information like team skills (attack,defense,midfield etc)
from sofifa.com and also computes point in time head to head stats and use them as features for predicting the outcome of a match.
Only group stage matches are predicted. 

- Team skills are obtained from http://sofifa.com using Scrapy
- Point in time head to head stats are computed in the Python script "football_prediction.py"
- Main challenge is the availability of Source Data. Only Source Data availabel is results of International fixtures.Features like Head to head stats , winning streaks are synthesized from source data. Team skills are extracted from sofifa.com which may not reflect that actual skills of the team at the time.

## Data preparation
1. football_results.csv :- The file contains all the international matches results since 1872
2. sofifa_team_list.csv :- The output file created from Spider which contains International teams and their skills availabe from sofifa.com
3. team_2018.csv :- List of all teams participating in WC 2018 finals
4. dataset_prepared.csv :- Intermediate file created by football_prediction.py script . This contains all match history with team skills and point in time head to head stats
5. first_round_prediction.csv :- Output file with winner's predicted

## Scripts:
1. football_prediction.py : The main script which computes head to head stats and train the models and predict the outcomes.
2. Football_predictions/football/football/spiders/matchcrawler.py :- The script scraps data from sofifa.com to fetch team skills.

