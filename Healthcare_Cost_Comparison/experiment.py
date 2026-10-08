"""Reproducible insurance-cost comparison. Run before interpreting test scores."""
import hashlib, io, json, os, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
DATA_BYTES = (HERE / 'insurance.csv').read_bytes()
raw = pd.read_csv(io.BytesIO(DATA_BYTES))
df = raw.drop_duplicates().copy()
X, y = df.drop(columns='charges'), df.charges
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
assert set(X_train.index).isdisjoint(X_test.index)
assert not df.isna().any().any()
CV = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
SEEDS = [42, 43, 44, 45, 46]

def engineer_features(frame):
    # Prespecified functional forms, not selected using held-out test results.
    out = frame.copy()
    out['bmi_squared'] = out.bmi ** 2
    out['smoker_bmi'] = out.bmi * (out.smoker == 'yes').astype(float)
    return out

def make_model(estimator, engineered=False):
    numeric = ['age', 'bmi', 'children']
    if engineered:
        numeric += ['bmi_squared', 'smoker_bmi']
    pre = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('category', OneHotEncoder(drop='first', handle_unknown='ignore',
                                   sparse_output=False), ['sex', 'smoker', 'region'])
    ], verbose_feature_names_out=False)
    steps = []
    if engineered:
        steps.append(('features', FunctionTransformer(engineer_features, validate=False)))
    steps += [('preprocess', pre), ('model', estimator)]
    # Scaling y is useful for Adam. It does NOT log-transform y or change the
    # objective from squared error on charges; inverse transform restores units.
    return TransformedTargetRegressor(regressor=Pipeline(steps), transformer=StandardScaler())

def metrics(actual, predicted):
    return dict(RMSE=float(root_mean_squared_error(actual, predicted)),
                MAE=float(mean_absolute_error(actual, predicted)),
                R2=float(r2_score(actual, predicted)))

specs = {
    'Ridge (L2)': (make_model(Ridge()),
        {'regressor__model__alpha': [0.01, 0.1, 1, 10, 100]}),
    'Lasso (L1)': (make_model(Lasso(max_iter=10000)),
        {'regressor__model__alpha': [0.0001, 0.001, 0.01, 0.1]}),
    'Ridge + engineered features': (make_model(Ridge(), engineered=True),
        {'regressor__model__alpha': [0.01, 0.1, 1, 10, 100]}),
    'MLP (32, 16)': (make_model(MLPRegressor(
        hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=40, max_iter=1500, tol=1e-5,
        batch_size=64, random_state=42)),
        {'regressor__model__learning_rate_init': [0.0003, 0.001, 0.003],
         'regressor__model__alpha': [0.0001, 0.01]})
}
searches, tuning_records = {}, []
for name, (model, grid) in specs.items():
    print('Tuning:', name, flush=True)
    search = GridSearchCV(model, grid, cv=CV, scoring='neg_root_mean_squared_error',
                          n_jobs=1, refit=True, error_score='raise')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        search.fit(X_train, y_train)
    searches[name] = search
    print('CV RMSE:', round(-search.best_score_, 2), search.best_params_, flush=True)
    for i, p in enumerate(search.cv_results_['params']):
        tuning_records.append(dict(model=name, parameters=json.dumps(p),
            CV_RMSE=-float(search.cv_results_['mean_test_score'][i]),
            CV_fold_SD=float(search.cv_results_['std_test_score'][i])))
    if caught:
        print('Warnings:', sorted(set(str(w.message) for w in caught)), flush=True)

# All tuning is now finished. Only now evaluate held-out observations.
predictions = pd.DataFrame({'original_row_id': X_test.index, 'actual_charges': y_test.values})
score_records = []
models = {}
threshold = float(y_train.quantile(0.9))
tail = y_test.to_numpy() >= threshold
for name in ['Ridge (L2)', 'Lasso (L1)', 'Ridge + engineered features']:
    model = searches[name].best_estimator_
    pred = model.predict(X_test)
    models[name] = model
    predictions[name] = pred
    score_records.append(dict(model=name, seed=None, **metrics(y_test, pred),
        tail_MAE=float(mean_absolute_error(y_test.to_numpy()[tail], pred[tail])),
        negative_predictions=int((pred < 0).sum()), epochs=None))
for seed in SEEDS:
    model = clone(searches['MLP (32, 16)'].best_estimator_)
    model.set_params(regressor__model__random_state=seed)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    models[f'MLP seed {seed}'] = model
    predictions[f'MLP seed {seed}'] = pred
    score_records.append(dict(model='MLP (32, 16)', seed=seed, **metrics(y_test, pred),
        tail_MAE=float(mean_absolute_error(y_test.to_numpy()[tail], pred[tail])),
        negative_predictions=int((pred < 0).sum()),
        epochs=int(model.regressor_.named_steps['model'].n_iter_)))
scores = pd.DataFrame(score_records)
summary = scores.groupby('model', sort=False).agg(
    RMSE=('RMSE', 'mean'), RMSE_seed_SD=('RMSE', 'std'),
    MAE=('MAE', 'mean'), MAE_seed_SD=('MAE', 'std'), R2=('R2', 'mean'),
    R2_seed_SD=('R2', 'std'), tail_MAE=('tail_MAE', 'mean'))
summary[['RMSE_seed_SD','MAE_seed_SD','R2_seed_SD']] = summary[
    ['RMSE_seed_SD','MAE_seed_SD','R2_seed_SD']].fillna(0)

mlp_rmse = summary.loc['MLP (32, 16)', 'RMSE']
improvements = {name: 100 * (summary.loc[name, 'RMSE'] - mlp_rmse) /
                         summary.loc[name, 'RMSE'] for name in summary.index if name != 'MLP (32, 16)'}

# Inspect a defined profile, NOT a test case chosen to flatter a model.
# This is sensitivity of predictions, not causal effects or clinical evidence.
profiles = pd.DataFrame([
    dict(age=40, sex='female', bmi=30., children=1, smoker=s, region='southwest')
    for s in ['no', 'yes']])
plus = profiles.copy(); plus['bmi'] += 1
eps = 1e-5
tiny_plus = profiles.copy(); tiny_plus['bmi'] += eps
tiny_minus = profiles.copy(); tiny_minus['bmi'] -= eps

def mlp_bmi_input_gradient(fitted, profile):
    """Chain rule through frozen ReLU network; units: charges / raw BMI unit."""
    pipe = fitted.regressor_
    pre = pipe.named_steps['preprocess']
    nn = pipe.named_steps['model']
    z = pre.transform(profile)
    activations = [z]
    gates = []
    for weights, bias in zip(nn.coefs_[:-1], nn.intercepts_[:-1]):
        h = activations[-1] @ weights + bias
        gates.append((h > 0).astype(float))
        activations.append(np.maximum(h, 0))
    grad = np.tile(nn.coefs_[-1].T, (len(profile), 1))
    for k in range(len(gates)-1, -1, -1):
        grad = (grad * gates[k]) @ nn.coefs_[k].T
    bmi_index = list(pre.get_feature_names_out()).index('bmi')
    x_scale = pre.named_transformers_['numeric'].scale_[1]
    y_scale = fitted.transformer_.scale_[0]
    return grad[:, bmi_index] * y_scale / x_scale

sensitivity_records = []
for name in ['Ridge (L2)', 'Ridge + engineered features', 'MLP seed 42']:
    model = models[name]
    base_pred, step_pred = model.predict(profiles), model.predict(plus)
    derivative = (model.predict(tiny_plus) - model.predict(tiny_minus)) / (2*eps)
    if name == 'MLP seed 42':
        chain = mlp_bmi_input_gradient(model, profiles)
        assert np.allclose(chain, derivative, rtol=1e-5, atol=0.01)
    else:
        chain = [None, None]
    for i, row in profiles.iterrows():
        sensitivity_records.append(dict(model=name, smoker=row.smoker, bmi=30,
            prediction=float(base_pred[i]), BMI_plus_1=float(step_pred[i]-base_pred[i]),
            local_BMI_gradient=float(derivative[i]), chain_rule_gradient=chain[i]))
sensitivity = pd.DataFrame(sensitivity_records)

# Independent recalculation from saved prediction vectors.
for rec in score_records:
    key = rec['model'] if rec['seed'] is None else f"MLP seed {rec['seed']}"
    errors = predictions[key].to_numpy() - predictions.actual_charges.to_numpy()
    assert np.isclose(np.sqrt(np.mean(errors ** 2)), rec['RMSE'])
    assert np.isclose(np.mean(np.abs(errors)), rec['MAE'])

metadata = dict(source_url='https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    kaggle_reference='https://www.kaggle.com/datasets/mirichoi0218/insurance',
    retrieved_date='2026-10-08', SHA256=hashlib.sha256(DATA_BYTES).hexdigest(),
    raw_rows=len(raw), exact_duplicates_removed=len(raw)-len(df), rows=len(df),
    train_rows=len(X_train), test_rows=len(X_test), split_seed=42,
    CV='5-fold shuffled KFold, seed 42, on training data only; same folds for all models',
    MLP_seeds=SEEDS, tail_threshold=threshold, tail_test_count=int(tail.sum()),
    best_parameters={name: s.best_params_ for name, s in searches.items()},
    CV_RMSE={name: -s.best_score_ for name, s in searches.items()},
    RMSE_reduction_percent=improvements,
    versions=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                  numpy=np.__version__, pandas=pd.__version__))
if __name__ == '__main__':
    scores.to_csv(HERE/'scores.csv', index=False)
    summary.to_csv(HERE/'summary.csv')
    predictions.to_csv(HERE/'predictions.csv', index=False)
    sensitivity.to_csv(HERE/'sensitivity.csv', index=False)
    pd.DataFrame(tuning_records).to_csv(HERE/'tuning.csv', index=False)
    (HERE/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(summary.round(3).to_string(), flush=True)
    print('RMSE reduction % versus baselines:', improvements, flush=True)
    print(sensitivity.round(3).to_string(index=False), flush=True)
