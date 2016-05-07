import numpy as np
import xgboost as xgb

dtrain = xgb.DMatrix('train.txt')
dtest = xgb.DMatrix('test.txt')

# specify parameters via map, definition are same as c++ version
param = {'max_depth':5, 'eta':.1, 'silent':1, 'objective':'binary:logistic' }

# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 5
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))