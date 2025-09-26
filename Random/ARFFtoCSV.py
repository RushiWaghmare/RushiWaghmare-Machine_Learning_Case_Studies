import pandas as pd
from scipy.io import arff
data = arff.loadarff('MIP.arff')
train= pd.DataFrame(data[0])
train.head()