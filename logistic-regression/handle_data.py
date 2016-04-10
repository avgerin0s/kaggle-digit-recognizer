import numpy as np
import csv

def load_data(csvpath, test=False):
    data = []
    labels = []
    with open(csvpath, 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='/')
        i = 0
        for row in csvreader:
            i += 1

            if i == 1:
                continue

            if not test:
                labels.append(int(row[0]))
                row = row[1:]

            data.append(np.array(np.int64(row)))

    return [data, labels]

def dump_results():
    f = open("results.csv", "w")
    writer = csv.writer(f)
    n = len(y_predict)
    writer.writerow(["ImageId", "Label"])
    for i in range(n):
        writer.writerow([i+1, y_predict[i]])

    f.close()
