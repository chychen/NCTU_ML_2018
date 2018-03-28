"""
2. Online learning 
Use online learning to learn the beta distribution of the parameter p (chance to see 1) of the coin tossing trails in batch. 
    INPUT: 
            i. A file contains many lines of binary outcomes: 
                  0101010111011011010101 
                  0110101 
                  010110101101 
                   ... 
             ii. parameter a for the initial beta prior 
             iii. parameter b for the initial beta prior 
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


def online_learning(all_data, prior_a, prior_b):
    """ Use Beta-Binomial conjugation to perform online learning.
    Print out the Binomial likelihood (based on MLE, of course), Beta prior and posterior probability (parameters only) for each line. 
    """
    prior = prior_a/(prior_a+prior_b)
    all_successes = 0
    all_failures = 0
    all_n = 0
    for i, data in enumerate(all_data):
        n = len(data)
        successes = 0
        for v in data:
            if v == 1:
                successes += 1
        failures = n - successes
        all_successes += successes
        all_failures += failures
        all_n += n
        p = all_successes/all_n
        mle = p**all_successes*(1-p)**all_failures
        posterior = (all_successes+1)/(all_successes+1+all_failures+1)
        print('DATA INDEX: {}\n\tPROB: {}\n\tMLE: {}\n\tPRIOR: {}\n\tPOSTERIOR: {}\n'.format(
            i, p, mle, prior, posterior))
        prior = posterior


def main():
    all_data = []
    with open('online_learning_data.txt', 'r') as file_:
        for line in file_.readlines():
            temp = []
            for c in line:
                if c == '\n':
                    break
                else:
                    temp.append(int(c))
            all_data.append(temp)

    online_learning(all_data, prior_a=1.0, prior_b=1.0)


if __name__ == '__main__':
    main()
