from hyperopt import fmin, tpe, hp, Trials
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

# Define objective function
def objective_function(params):
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return {'loss': -score, 'status': 'ok'}

# Define parameter space
space = {
    'learning_rate': hp.loguniform('learning_rate', -2, 0),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'n_estimators': hp.quniform('n_estimators', 50, 200, 1),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
}

# Run optimization
trials = Trials()
best_params = fmin(fn=objective_function, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print(best_params)
