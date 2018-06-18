import pandas as pd
import numpy as np
import datetime

def writetoOracle(data,table_name,engine):
    data.to_sql(name=table_name,con=engine ,if_exists = 'append', index=False)

match = pd.read_csv('football_results.csv')
data_team = pd.read_csv('team_skills.csv')
country_array = data_team['Country'].values
match['match_date'] = pd.to_datetime(match['match_date']) ## Convert datatype to date
match = match.drop(columns=['city','tournament','country','neutral'])
#match = match.loc[match['match_date'] > datetime.date(2000,1,1)]
match_filtered = match.loc[(match['home_team'].isin(country_array)) | (match['away_team'].isin(country_array))]
match_filtered['winning_team'] = np.where(match_filtered['home_score'] > match_filtered['away_score'],'Home',np.where(match_filtered['home_score'] < match_filtered['away_score'],'Away','Draw'))




schema = 'edw'
password = 'edw'
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
engine = create_engine('oracle+cx_oracle://{0}:{1}@DESKTOP-T9TNRHD:1530/xe?charset=utf8'.format(schema,password), encoding = 'utf-8', echo=False)
Session = sessionmaker(bind=engine)
connection = engine.connect()
session = Session()

session.execute('TRUNCATE TABLE football_results')
writetoOracle(match_filtered,'football_results',engine)
session.commit()