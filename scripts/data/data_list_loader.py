import os.path

def load_data_list(path):
    ''' Parses the file in path and returns a list of the datasets in the file '''
    if not os.path.isfile(path):
        print('Given data list is not a file!')
        return

    list_file = open(path, 'r')
    data_list = [x.split('\n')[0] for x in list_file.readlines()]
    # clear out potentially empty lines
    data_list = [x for x in data_list if x != '']
    return data_list

def load_dataset(path):
    ''' Parses the train, val, and test dataset lists from a path '''
    if not os.path.exists(path):
        print('Given datalist set path does not exist!')
        return

    train = load_data_list(os.path.join(path, 'train.txt'))
    val = load_data_list(os.path.join(path, 'val.txt'))
    test = load_data_list(os.path.join(path, 'test.txt'))

    return train, val, test

if __name__=='__main__':
    print(load_data_list('./data/sim/dataset_lists/Cube/all.txt'))
    print(load_dataset('./data/sim/dataset_lists/Cube'))
