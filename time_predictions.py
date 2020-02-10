#!/usr/bin/env python3
import numpy as np 
import pandas as pd

path = '/home/b/Studies/2IOI0/original_files/%s.csv'
time_df = pd.read_csv(path % 'BPI_Challenge_2012-training')

# Leaving only case, timestamp and event
time_df = time_df[['case concept:name', 'event concept:name', 'event time:timestamp']]
# Converting to datetime64 SLOW STEP!
time_df.loc[:, 'event time:timestamp'] = pd.to_datetime(time_df['event time:timestamp'])
# Converting to actual timestamp 
time_df.loc[:, 'event time:timestamp'] = (time_df['event time:timestamp'] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

# sort by case and event
time_df = time_df.sort_values(by=['case concept:name', 'event time:timestamp'])

# Make an empty df with identical columns
t_df = time_df.drop([*time_df.index])

# initiating time difference and previous event
t_df['delta'] = 0
t_df['prev_event'] = 0

# Loop to create time difference by event on a case-by-case basis
# Used as workaround to not mess up the values for the first event in a case
cases = time_df['case concept:name'].unique()
for case in cases:

    # Filter a specific case
    t = time_df.loc[time_df['case concept:name'].values == case, :]

    # Fill in time difference and previous event
    t.loc[: ,'delta'] = t['event time:timestamp'].diff()
    t.loc[: ,'prev_event'] = t['event concept:name'].shift(1)

    # append all events related to one case
    t_df = t_df.append(t)


time_df = t_df

# Fill in the values for first event
time_df[['delta', 'prev_event']] = time_df[['delta', 'prev_event']].fillna(0)

# Calculating average time for every possible event transition
time_pred_df = time_df[[
    'prev_event', 
    'event concept:name', 
    'delta']].time_groupby(['prev_event', 'event concept:name']).mean()

# Get rid of the first event
time_pred_df = time_pred_df[time_pred_df['prev_event'] != 0]

# Save
time_pred_df.to_csv('data/predicted_times.csv')
