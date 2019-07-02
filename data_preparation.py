import pandas as pd
import numpy as np

training = pd.read_csv('training_set_rel3.tsv',sep="\t",usecols=["essay_set","essay","domain1_score"])
test = pd.read_csv('test_set.tsv',sep="\t",usecols=["essay_set","essay"])
validation = pd.read_csv('valid_set.tsv',sep="\t",usecols=["essay_set","essay"])
test['domain1_score'] = np.nan
validation['domain1_score'] = np.nan
complete = pd.concat([training,test,validation])
complete.columns = ['essay_set','Essay','Grade']
complete['Grade'].astype("Int64")
for essay_set in complete['essay_set'].unique():
    complete[complete['essay_set']==essay_set].drop(['essay_set'],axis=1).to_csv(
        "prompt_{}.tsv".format(essay_set),sep="\t",index=False)