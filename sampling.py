from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


#upsampling
def upsampling(X_train,Y_train):
    ros = RandomOverSampler(random_state=42)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    return X_train,Y_train

#downsampling
def downsampling(X_train,Y_train):
    rds = RandomUnderSampler(random_state=42)
    X_train, Y_train = rds.fit_resample(X_train, Y_train)
    return X_train,Y_train

## nosampling
def nosampling(X_train,Y_train):
    return X_train,Y_train

##SOMTE
def SMOTEsampling(X_train,Y_train):
    sm=SMOTE(random_state=42)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)
    return X_train,Y_train
