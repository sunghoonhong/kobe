import numpy as np
import pandas as pd
raw_df = pd.read_csv('data.csv', sep=',')

# data cleansing
raw_df['remaining_time'] = raw_df['minutes_remaining'] * 60 + raw_df['seconds_remaining']
if type(raw_df['season'][0]) is str:
    raw_df['season'] = raw_df['season'].apply(lambda x: int(x.split('-')[0]))
raw_df['home'] = raw_df['matchup'].apply(lambda x: int(x.find('@')==-1))

categoric_vars = ['action_type', 'combined_shot_type', 'shot_type', 'season', 'opponent']
for var in categoric_vars:
    raw_df = pd.concat([raw_df, pd.get_dummies(raw_df[var], prefix=var)], 1)
    raw_df = raw_df.drop(var, 1)

# drop garbage
drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw_df = raw_df.drop(drop, 1)

# seperate data
train_df = raw_df[pd.notnull(raw_df['shot_made_flag'])]
test_df = raw_df[pd.isnull(raw_df['shot_made_flag'])]