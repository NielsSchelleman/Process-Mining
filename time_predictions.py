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

# Fill in time difference and previous event
time_df['delta'] = time_df['event time:timestamp'].diff()
time_df['prev_event'] = time_df['event concept:name'].shift(1)

time_df = time_df.set_index('case concept:name')

# Make an empty df with identical columns
t_df = time_df.drop([*time_df.index])

# Loop to get rid of the first event
cases = np.unique(time_df.index)

for case in np.nditer(cases): # not sure if speeds up
    # Filter a specific case
    t = time_df.loc[case]
    # append all events related to one case
    t_df = t_df.append(t.iloc[1:])

time_df = t_df.reset_index()


# Calculating average time for every possible event transition
time_pred_df = time_df[[
    'prev_event', 
    'event concept:name', 
    'delta']].groupby(['prev_event', 'event concept:name']).mean()

# Get rid of the grouping
time_pred_df = time_pred_df.reset_index()
time_pred_df = time_pred_df.rename({'delta': 'delta_avg'})

# Save
time_pred_df.to_csv('../data/predicted_times.csv')

