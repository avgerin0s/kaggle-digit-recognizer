import numpy as np
import csv

from sklearn import linear_model

if __name__ == "__main__":
    from handle_data import load_data, dump_results

    x, y = load_data("../train.csv")

    print "Loaded training set"

    x_test, y_test = load_data("../test.csv", test=True)

    print "Loaded testing set"

    logreg = linear_model.LogisticRegression(solver='lbfgs', n_jobs=3, max_iter=200)
    logreg.fit(x, y)

    print "Classifier trained"

    y_predict = logreg.predict(x_test)

    print "Predictions ready"

    dump_results()
