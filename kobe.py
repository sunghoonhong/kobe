import numpy as np
import pandas as pd
raw_df = pd.read_csv('data.csv', sep=',')

raw_df['remaining_time'] = raw_df['minutes_remaining'] * 60 + raw_df['seconds_remaining']
if type(raw_df['season'][0]) is str:
    raw_df['season'] = raw_df['season'].apply(lambda x: int(x.split('-')[0]))

    
raw_df['home'] = raw_df['matchup'].apply(lambda x: int(x.find('@')==-1))

submission = raw_df[pd.isnull(raw_df['shot_made_flag'])]['shot_id']

drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw_df = raw_df.drop(drop, 1)
    
categoric_vars = ['action_type', 'combined_shot_type', 'shot_type', 'season', 'opponent']
for var in categoric_vars:
    raw_df = pd.concat([raw_df, pd.get_dummies(raw_df[var], prefix=var)], 1)
    raw_df = raw_df.drop(var, 1)

notnull_df = raw_df[pd.notnull(raw_df['shot_made_flag'])]
null_df = raw_df[pd.isnull(raw_df['shot_made_flag'])]

train_X = notnull_df.drop('shot_made_flag', 1)
train_y = notnull_df['shot_made_flag']

from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt

test_X = null_df.drop('shot_made_flag', 1)

model = XGBClassifier()
model.fit(train_X, train_y)

plot_importance(model)
plt.show()

prediction = model.predict_proba(test_X)[:, 1]

result = pd.DataFrame({ 
            'shot_id': submission,
            'shot_made_flag': prediction
        })
        
result.to_csv('output.csv', sep=',', index=False)