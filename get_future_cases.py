# required libraries
import pandas as pd


# read your files here
# example
# data_fine_test = pd.read_csv('Road_Traffic_Fine_Management_Process-test.csv')
# data_fine_train = pd.read_csv('Road_Traffic_Fine_Management_Process-training.csv')

def Make_eventDB(data, columnNameCases=None, columnNameEvents=None):
    # if no case column has been defined, find the most probable column
    if columnNameCases is None:
        caselengths = {}
        for item in list(data.columns[data.columns.str.find('case') == 0]):
            caselengths.update({len(set(data[item])): item})
        columnNameCases = caselengths[max(caselengths.keys())]

    # if no event column has been defined check if it is a known column, otherwise print error
    if columnNameEvents is None:
        if 'event org:resource' in list(data.columns[data.columns.str.find('event') == 0]):
            columnNameEvents = 'event org:resource'
        elif 'event concept:name' in list(data.columns[data.columns.str.find('event') == 0]):
            columnNameEvents = 'event concept:name'
        else:
            print('Events column could not be found automatically please insert an Events column into function')
            return None

    # dict for storage
    events = {}
    # for loop gets a key(the case) and value(list of events for every case) for the dictionary
    for key, value in zip(data[columnNameCases], data[columnNameEvents]):
        if key in events.keys():
            events[key].append(value)
        else:
            events.update({key: [value]})

    # get the longest list of events and set all lists of events to be that length
    maxLen = len(max(list(events.values()), key=len))
    for key in events.keys():
        addList = (maxLen - len(events[key])) * [None]
        events[key].extend(addList)

    # make a dataframe for all the events
    eventDB = pd.DataFrame(events).transpose()
    print("database initialised")
    return eventDB


def analyse_test(test, data) -> list:
    # test is a list of all events before the event to be predicted
    # if it is the first element it will just look at the value with the highest probability within the data
    if test == [] or test == None:
        freq = data[0].value_counts()

    else:
        # for one element, the previous function could return it outside a list in that case it is put inside a list
        if type(test) != list:
            test = [test]
        # slices the dataframe of training data to only include data with the same starting pattern as test
        for i in range(len(test)):
            data_slice = data[data[i] == test[i]]
            # at the last element it predicts the next(future) element by getting the mode
            if test[i] == test[-1]:
                freq = data_slice[i + 1].value_counts()
    # if the dataframe slice is empty it just takes the mode of all values
    if len(freq.values) == 0:
        freq = data[len(test)].value_counts()
    # function for getting the mode
    return freq[freq == max(freq)].index[0]


def getBaselineTest(event_trainDB, event_testDB):
    results = {'correct': 0, 'false': 0}
    for event in event_testDB.iterrows():
        event = list(filter(None, event[1].values))
        for case in range(len(event)):
            # performs analyse_test for each subset of test [0:i-1] where i is the case to be predicted
            if case == 0:
                prediction = analyse_test([], event_trainDB)
            else:
                prediction = analyse_test(event[0:case - 1], event_trainDB)
            # checks prediction result
            if event[case] == prediction:
                results['correct'] += 1
            else:
                results['false'] += 1
        # prints a line for a status update
        print('\rcorrect:{}          false:{}          percentage correct:{}%'.format(results['correct'],
                                                                                      results['false'], round(
                (results['correct'] / (results['false'] + results['correct']) * 100), 0), end=''))
    return results

# example of getting future cases from function
# results = getBaselineTest(Make_eventDB(data_fine_train),Make_eventDB(data_fine_test))
