from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.log import EventLog
from pm4py.algo.prediction import factory as prediction_factory


# Function takes three arguments 
#   path= string of filepath to dataset in .xes format
#   testsize = float, size of the testset as percentage of entire dataset (counted on traces)
#   demo = boolean, if the entire dataset gets run or only the first 100 traces
def get_timestamp_rnn(path="Road_Traffic_Fine_Management_Process.xes.gz",testsize=0.2,demo=False):
    # reads the data as a log type (a list of events with their cases)
    print('constructing log')
    log = xes_importer.apply(path)
    
    #uses only 100 events for the demo to keep runtimes short
    if demo == True:
        log = log[0:100]
    # Splitting the data into training and testing data
    # If statements because the RNN can only evaluate cases with at most a length equal to the maximum length within the training data
    # Therefore the longest case needs to be in the training data
    
    print("splitting data")
    train_log_size = int(len(log) * (1-testsize))
    print(train_log_size)
    training_log = EventLog(log[0:train_log_size])
    test_log = EventLog(log[train_log_size:len(log)])
    print('validity check')
    if max(log, key=len) == max(test_log,key=len):
            print('reformatting data')
            train_log_size = int(len(log) * testsize)
            test_log = EventLog(log[0:train_log_size])
            training_log =EventLog(log[train_log_size:len(log)])
    if max(log, key=len) == max(test_log,key=len):
        print("TestsizeError, a problem occured with the maximum tracelengths of the test & train sets")
        return "no model, please try again"
    
    #this actually constructs the model
    print('constructing model (slow)')
    model = prediction_factory.train(training_log, variant="keras_rnn")
    
    # applies the test set to the model
    print('testing the model')
    rnn_results = prediction_factory.test(model, test_log)
    return rnn_results  



# does a demo and prints summary statistics of the results
import pandas as pd
print(pd.Series(get_timestamp_rnn(demo=True)).describe())