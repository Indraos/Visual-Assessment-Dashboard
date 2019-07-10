"""
To use this script, place the files training_set_rel3.tsv, test_set.tsv and valid_set.tsv
downloaded from https://www.kaggle.com/c/asap-aes/data in a subdirectory data/ of the
calling directory.

This generates a file conformant with the input structure of our web application.
"""
import pandas as pd
import numpy as np


def main():
    """
    Reads in data from asap-aes kaggle challenge and writes in conformant style


    This function takes training, validation and test supplied by the user from
    kaggle to a style conformant with the input of our dashboard.
    """
    training = pd.read_csv('./data/training_set_rel3.tsv',
                           sep="\t",
                           usecols=["essay_set", "essay", "domain1_score"])
    test = pd.read_csv('./data/test_set.tsv',
                       sep="\t",
                       usecols=["essay_set", "essay"])
    validation = pd.read_csv('./data/valid_set.tsv',
                             sep="\t",
                             usecols=["essay_set", "essay"])
    test[
        'domain1_score'] = np.nan  # scores for test and validation set unknown
    validation['domain1_score'] = np.nan
    complete = pd.concat([training, test, validation])
    complete.columns = ['essay_set', 'Essay', 'Grade']
    complete['Grade'].astype("Int64")  # Int64 allows nan
    for essay_set in complete['essay_set'].unique():
        essay_questions = complete[complete['essay_set'] == essay_set]
        essay_questions.drop(['essay_set'], axis=1)
        essay_questions.to_csv("./data/prompt_{}.tsv".format(essay_set),
                               sep="\t",
                               index=False)


if __name__ == "__main__":
    main()
