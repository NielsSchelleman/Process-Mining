#!/usr/bin/env python3
import numpy as np 
import pandas as pd
from typing import List


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares a pd.DataFrame for time prediction training and testing.

    Asumes that it has the columns:
    + 'case concept:name',
    + 'event concept:name',
    + 'event time:timestamp'
    """

    # Leaving only case, timestamp and event
    df = df[['case concept:name', 'event concept:name', 'event time:timestamp']]
    
    # Converting to datetime64 SLOW STEP!  
    df.loc[:, 'event time:timestamp'] = pd.to_datetime(df['event time:timestamp'])

    # Converting to actual timestamp 
    df.loc[:, 'event time:timestamp'] = (df['event time:timestamp'] 
            - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    
    # sort by case and event
    df = df.sort_values(by=['case concept:name', 'event time:timestamp'])

    # Fill in time difference and previous event
    group_by_case_df = df.groupby('case concept:name')
    df['prev_event'] = group_by_case_df['event concept:name'].shift(1)
    df['delta'] = group_by_case_df['event time:timestamp'].diff()

    return df


def trainTime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the average time between all event transitions
    and prints training acurracy measures.

    Asumes that the pd.DataFrame has gone trough the preprocess function.
    """

    # Calculating average time for every possible event transition
    time_pred_df = df[[
        'prev_event', 
        'event concept:name', 
        'delta']].groupby(['prev_event', 'event concept:name']).mean()

    # Get rid of the grouping
    time_pred_df = time_pred_df.reset_index()
    time_pred_df = time_pred_df.rename(columns={'delta': 'delta_pred'})

    return time_pred_df


def trainAcc(df: pd.DataFrame, time_pred_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(time_pred_df, on=['prev_event', 'event concept:name'], how='left')
    df['Training Acurracy'] =  np.round((100 * (df['delta'] - df['delta_pred'])) / df['delta'], 2)
    return df['Training Acurracy'].describe()


def testAcc(df: pd.DataFrame) -> pd.DataFrame:
    df['Testing Acurracy'] =  np.round((100 * (df['delta'] - df['delta_pred'])) / df['delta'], 2)
    return df['Testing Acurracy'].describe()


def predTime(df: pd.DataFrame, time_pred_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(time_pred_df, on=['prev_event', 'event concept:name'], how='left')
    return df


path = '/home/b/Studies/2IOI0/original_files/%s.csv'

train_df = pd.read_csv(path % 'BPI_Challenge_2012-training')
test_df = pd.read_csv(path % 'BPI_Challenge_2012-test')
train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Model and its training acurracy scores
model_df = trainTime(train_df)
print(trainAcc(train_df, model_df))

# Testing and scores
results_df = predTime(test_df, model_df)
print(testAcc(results_df))
