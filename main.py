#!python

from model.planner import Planner
from model.reflex import Reflex
from task import maze

import apollocaffe
import numpy as np

N_ITERS = 1000000
N_BATCH = 100

def main():
    task = maze
    model = Planner()
    total_loss = 0
    for i_iter in range(N_ITERS):
        train_data = task.load_batch(N_BATCH)
        test_data = train_data

        loss, acc = do_iter(train_data, model)
        total_loss += loss

        if i_iter % 1000 == 0:
            preds = model.predict(test_data)
            accs = [task.evaluate(preds[i], test_data[i]) for i in range(N_BATCH)]
            acc = np.mean(1. * np.asarray(accs))

            print "%8.3f   %5.3f" % (total_loss, acc)
            total_loss = 0

            if 1 in accs:
                print
                print "SUCCESS"
                i = accs.index(1)
                print task.visualize(preds[i], test_data[i])

            if 0 in accs:
                print
                print "FAILURE"
                i = accs.index(0)
                print preds[i]
                print task.visualize(preds[i], test_data[i])
                print test_data[i].demonstration
                print task.visualize(test_data[i].demonstration, test_data[i])

            print "\n"


def do_iter(data, model):
    loss = model.demonstrate(data)
    return loss, 0.

if __name__ == "__main__":
    #apollocaffe.set_device(0)
    main()
