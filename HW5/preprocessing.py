
# training
data = []
with open('data/X_train.csv') as f_data, open('data/T_train.csv') as f_label:
    for data_line, label_line in zip(f_data.readlines(), f_label.readlines()):
        tmp_line = data_line.strip().split(',')
        each_data = ''
        for i, tmp_str in enumerate(tmp_line):
            each_data += ' '+str(i+1)+':'+tmp_str
        each_data = str(int(label_line)) + each_data+'\n'
        data.append(each_data)
with open('data/training.csv', 'w') as fout:
    for each_data in data:
        fout.write(each_data)
# testing
data = []
with open('data/X_test.csv') as f_data, open('data/T_test.csv') as f_label:
    for data_line, label_line in zip(f_data.readlines(), f_label.readlines()):
        tmp_line = data_line.strip().split(',')
        each_data = ''
        for i, tmp_str in enumerate(tmp_line):
            each_data += ' '+str(i+1)+':'+tmp_str
        each_data = str(int(label_line)) + each_data+'\n'
        data.append(each_data)
with open('data/testing.csv', 'w') as fout:
    for each_data in data:
        fout.write(each_data)