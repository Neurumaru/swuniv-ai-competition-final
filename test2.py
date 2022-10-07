import pandas as pd
from pycaret.classification import *

train = pd.read_csv('outputs/prediction.csv')
clf = setup(data = train, target = 'horizontal')
best_3 = compare_models(sort='AUC', n_select=3)
blended=blend_models(estimator_list=best_3, fold=5, method='soft')
pred_holdout = predict_model(blended)
final_model = finalize_model(blended)
predictions = predict_model(final_model, data=train)
predictions