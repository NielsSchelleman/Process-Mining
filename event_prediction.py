import pandas as pd
import numpy as np 

path = '/home/b/Studies/2IOI0/original_files/%s.csv'
train_df = pd.read_csv(path % 'BPI_Challenge_2012-training')
test_df = pd.read_csv(path % 'BPI_Challenge_2012-test')

# Keep only the necessary values
train_df = train_df[['eventID ', 'case concept:name', 'event concept:name']]
train_df = train_df.sort_values(by=['case concept:name', 'eventID '])

# Add an event counter for each case
train_df['n_event'] = train_df.groupby(['case concept:name'])['eventID '].cumcount()

# Create a DataFrame that records the full chain of events using case as index
cases_df = train_df.pivot(columns='n_event', index='case concept:name', values='event concept:name')
