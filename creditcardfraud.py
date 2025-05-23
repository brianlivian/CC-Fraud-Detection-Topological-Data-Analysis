# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# print("Path to dataset files:", path)

# %%
import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from tqdm.auto import tqdm  
from mpl_toolkits.mplot3d import Axes3D  
from gtda.time_series import SlidingWindow  
from gtda.homology import VietorisRipsPersistence  
from gtda.diagrams import PersistenceLandscape  
from sklearn import preprocessing  
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score  
from imblearn.combine import SMOTETomek  
from imblearn.pipeline import Pipeline  
import lightgbm as lgbm
from lightgbm import LGBMClassifier, early_stopping  
from xgboost import XGBClassifier  
import optuna
import optuna.visualization as vis
from optuna.importance import get_param_importances
from scipy.stats import ttest_ind, levene

# %%
os.listdir('/Users/brianlivian/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3')

# %%
df = pd.read_csv('/Users/brianlivian/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv')

# %%
pd.set_option('display.max_columns', None)
df

# %% [markdown]
# ## EDA

# %%
(df.isna().sum() >0).sum()

# %%
df['Class'].value_counts()

# %%
df

# %% [markdown]
# ## Apply sliding window embeddings (of size 2-10) to each PCA vector (column wise) to create the point clouds 

# %%
for e in tqdm(range(2,10), desc = "Window Sizes", unit="e"):
    window_size = e
    stride = 1
    X_sws = []
    for i in tqdm(range(len(df)), desc="Sliding windows", unit="row"):
        X = np.asarray(df.iloc[i, 1:29])
        y = [1]
        SW = SlidingWindow(size=window_size, stride=stride)
        X_sw, y_sw = SW.fit_transform_resample(X, y)
        X_sws.append(X_sw)
    df['E{}'.format(e)] = X_sws


# %%
plt.figure()
plt.scatter(df['E2'][0][:,0], df['E2'][0][:,1])
plt.title('2D Point Cloud')
plt.show()

# %%
# Plot the persistence diagram and landscape for a random point cloud sliding window
pointcloud = df['E2'][0]

vrp = VietorisRipsPersistence()
plt.figure()
vrp.fit_transform_plot(pointcloud.reshape(1, *pointcloud.shape))
plt.show()

pl = PersistenceLandscape(layers)
persistencediagram = vrp.fit_transform(pointcloud.reshape(1, *pointcloud.shape))
landscapedata = pl.fit_transform(persistencediagram)
fig = pl.plot(landscapedata, 
    homology_dimensions = [0], 
    plotly_params=None)
fig.show()

# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['E3'][0][:, 0], df['E3'][0][:, 1], df['E3'][0][:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Point Cloud')

plt.show()

# %%
pointcloud = df['E3'][0]

vrp = VietorisRipsPersistence()
vrp.fit_transform_plot(pointcloud.reshape(1, *pointcloud.shape))
plt.show()

pl = PersistenceLandscape(layers)
persistencediagram = vrp.fit_transform(pointcloud.reshape(1, *pointcloud.shape))
landscapedata = pl.fit_transform(persistencediagram)
fig = pl.plot(landscapedata, 
    homology_dimensions = [0], 
    plotly_params=None)
fig.show()


# %%
# Functions to compute the Lp norms
# Find the range of x values from the persistence diagram:
def Ftseq(diagram):
    births =[]
    deaths =[]
    for pair in diagram:
        if pair[2] == 1:
            births.append(pair[0])
            deaths.append(pair[1])
    return np.linspace(min(births), max(deaths), 100)

# Calculate Lp norm:
def Lpnorm(tseq, landscapevalues, p = 1):
    norms = []
    if p == 'aucoriginal':
        layervalues = landscapevalues[layers]
        for point in zip(tseq,layervalues):
            norms.append(np.trapz(layervalues, tseq))
        return(sum(norms))
    elif p == 'aucUpdated':
        layervalues = landscapevalues[layers]
        return (np.trapz(layervalues))
    else:
        normvalues = []
        for layer in range(layers, 2*layers):
            layervalues = landscapevalues[layer]
            normvalue = np.linalg.norm(layervalues,p)**p
            if normvalue == 0:
                break
            else: 
                normvalues.append(normvalue)
        return (np.sum(normvalues)**(1/p))


# %% [markdown]
# ## Compute the L1 Norms

# %%
layers = 10
vrp = VietorisRipsPersistence()
pl = PersistenceLandscape(layers)

for e in tqdm(range(2,10), desc = "Window Sizes", unit="e"):
    Norms = []
    for pointcloud in tqdm(df['E{}'.format(e)], desc="Computing norms", unit="cloud"):
        persistencediagram = vrp.fit_transform(pointcloud.reshape(1, *pointcloud.shape))
        landscapedata = pl.fit_transform(persistencediagram)
        tseq = Ftseq(persistencediagram[0])
        Norm = Lpnorm(tseq, landscapedata[0], p = 1)
        Norms.append(Norm)
    df['N{}'.format(e)] = Norms


# %%
df.to_csv('data.csv')

# %%
df = pd.read_csv('data.csv')

# %%
df['Class'] = df['Class'].astype(str)


# %% [markdown]
# ## (Welch's) t test on whether difference in means of each variable is significant

# %%
def t_test(df, col, class_col='Class', group_labels=('0', '1'), alpha=0.05):
    group_0 = df[df[class_col] == group_labels[0]][col]
    group_1 = df[df[class_col] == group_labels[1]][col]

    # Test for equal variances
    _, p_var = levene(group_0, group_1)
    equal_var = p_var > alpha  # If p > alpha, variances are similar

    # Perform t-test with appropriate assumption
    t_stat, p_val = ttest_ind(group_0, group_1, equal_var=equal_var)

    return {
        'column': col,
        'equal_var_assumed': equal_var,
        'variance_test_p_value': p_var,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < alpha
    }

# Example usage:
result = t_test(df, 'N2')
print(result['p_value'])


# %% [markdown]
# ## L1 Norms obtained via sliding window embeddings of PCA vectors surprisingly have significant effect on response variable

# %%
# continuous variables
continuous = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']
for col in continuous:
    plt.figure(figsize = (18, 8))
    plt.suptitle(col + "\n t test p value: " + str(t_test(df, col)['p_value']) , size = 20, fontweight = 'bold')
    plt.subplot(1,2,1)
    sns.histplot(x = col, data=df[df[col] < np.quantile(df[col], .98)], hue = 'Class', kde = True)
    plt.subplot(1,2,2)
    sns.boxplot(x= col, data = df[df[col] < np.quantile(df[col], .98)], hue = 'Class')
    plt.show()

# %%
plt.figure(figsize = (20,12))
sns.heatmap(df[['N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']].corr(), annot = True)

# %% [markdown]
# ## SMOTETomek to balance the dataset

# %%
X = df[['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
       'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9']]
y = df['Class'].astype(int)

# 3. Split once for train / test, once again for hold-out valid
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.10, stratify=y_train, random_state=42
)

# 4. Balance *only* the training data with SMOTE-Tomek
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# %% [markdown]
# ## Initial run to obtain feature importances

# %%
lgb = LGBMClassifier()
lgb.fit(X_train_balanced,y_train_balanced,
        # early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid), (X_train_balanced, y_train_balanced)])
preds_train = lgb.predict(X_train_balanced)
accuracy_train, recall_train, precision_train, f1_train, auc_train= accuracy_score(y_train_balanced, preds_train), recall_score(y_train_balanced, preds_train), precision_score(y_train_balanced, preds_train), f1_score(y_train_balanced, preds_train), roc_auc_score(y_train_balanced, preds_train)

preds_test = lgb.predict(X_test)
accuracy_test, recall_test, precision_test, f1_test, auc_test = accuracy_score(y_test, preds_test), recall_score(y_test, preds_test), precision_score(y_test, preds_test), f1_score(y_test, preds_test), roc_auc_score(y_test, preds_test)


pd.DataFrame({
    'Metric':['Accuracy', 'Reccall', 'Precision', 'F1', 'AUC'],
    'Train Score' : [accuracy_train,recall_train,precision_train, f1_train,auc_train],
    'Test Score' : [accuracy_test,recall_test,precision_test, f1_test,auc_test],
})

# Get feature importances (by split count) and combine with column names
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb.booster_.feature_importance(importance_type='split')
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(
    data=feature_importance,
    y='Feature',
    x='Importance',
    palette='viridis'
)
plt.title('LightGBM Feature Importance (by Split Count)', fontsize=14)
plt.xlabel('Importance (Number of Splits)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print(*feature_importance['Feature'].tolist(), sep=', ')

# %% [markdown]
# ## Feature selection from feature importances
# - L1 Norms are not as strong as I was hoping, I'm including the strongest norm (N2) along with the other top features

# %%
X_train = X_train[['V14', 'Time', 'V4', 'V12','Amount', 'N2']]
X_test = X_test[['V14', 'Time', 'V4', 'V12','Amount', 'N2']]
X_valid = X_valid[['V14', 'Time', 'V4', 'V12','Amount', 'N2']]
X_train_balanced = X_train_balanced[['V14', 'Time', 'V4', 'V12','Amount', 'N2']]

# %%
X_train.shape

# %% [markdown]
# ## Hyperparameter tune with Optuna using cross validation and f_beta = 2 for more weight on recall and to reduce false negatives

# %%
from sklearn.metrics import make_scorer, fbeta_score


fbeta = make_scorer(fbeta_score, beta=2)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 20, 250),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1
    }

    pipeline = Pipeline([
        ('smote', SMOTETomek(random_state=42)),
        ('model', LGBMClassifier(**params))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=fbeta, n_jobs=-1)

    return scores.mean()

# Create and run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=25)

# Best result
print("Best trial:")
print(study.best_trial.params)


# %%
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
vis.plot_parallel_coordinate(study).show()
vis.plot_slice(study).show()
vis.plot_contour(study).show()

# %%
# -- Get optimization history values
values = [t.value for t in study.trials if t.value is not None]

# -- Get param importances
importances = get_param_importances(study)
params = list(importances.keys())
scores = list(importances.values())

# -- For contour plot, extract values for two chosen params
param_x = 'learning_rate'
param_y = 'n_estimators'
x, y, z = [], [], []

for t in study.trials:
    if t.value is None:
        continue
    if param_x in t.params and param_y in t.params:
        x.append(float(t.params[param_x]))
        y.append(int(t.params[param_y]))
        z.append(t.value)

# -- Create contour grid if enough points
has_contour = len(x) > 10

# -- Plot all in one figure
fig = plt.figure(figsize=(18, 5))

# 1. Optimization history
plt.subplot(1, 3, 1)
plt.plot(range(len(values)), values, marker='o')
plt.xlabel("Trial")
plt.ylabel("Objective Value")
plt.title("Optimization History")
plt.grid(True)

# 2. Parameter importances
plt.subplot(1, 3, 2)
plt.barh(params[::-1], scores[::-1])
plt.xlabel("Importance")
plt.title("Hyperparameter Importances")

# 3. Contour plot: learning_rate vs n_estimators
plt.subplot(1, 3, 3)
if has_contour:
    contour = plt.tricontourf(x, y, z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title("Contour: {} vs. {}".format(param_x, param_y))
else:
    plt.text(0.5, 0.5, "Not enough data for contour plot", ha='center', va='center')
    plt.axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# - Apply SmoteTomek in cross validation as shown below to avoid data leakage

# %%
# Rebuild pipeline with best params from Optuna
pipeline = Pipeline([
    ('smote', SMOTETomek(random_state=42)),
    ('model', LGBMClassifier(**study.best_trial.params, verbose=-1))
])

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = {
    'accuracy': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='accuracy').mean(),
    'recall': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='recall').mean(),
    'precision': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='precision').mean(),
    'f1': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='f1').mean(),
    'f2': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring=fbeta).mean(),
    'roc_auc': cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='roc_auc').mean(),
}

print(scores)

# %% [markdown]
# ## Run on test set

# %%
lgb = LGBMClassifier(**study.best_trial.params)
lgb.fit(X_train_balanced, y_train_balanced,
        eval_set=[(X_valid, y_valid), (X_train_balanced, y_train_balanced)])

preds_train = lgb.predict(X_train_balanced)
accuracy_train = accuracy_score(y_train_balanced, preds_train)
recall_train = recall_score(y_train_balanced, preds_train)
precision_train = precision_score(y_train_balanced, preds_train)
f1_train = f1_score(y_train_balanced, preds_train)
auc_train = roc_auc_score(y_train_balanced, preds_train)
fbeta_train = fbeta_score(y_train_balanced, preds_train, beta=2)

preds_test = lgb.predict(X_test)
accuracy_test = accuracy_score(y_test, preds_test)
recall_test = recall_score(y_test, preds_test)
precision_test = precision_score(y_test, preds_test)
f1_test = f1_score(y_test, preds_test)
auc_test = roc_auc_score(y_test, preds_test)
fbeta_test = fbeta_score(y_test, preds_test, beta=2)

pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1', 'F2', 'AUC'],
    'Train Score': [accuracy_train, recall_train, precision_train, f1_train, fbeta_train, auc_train],
    'Test Score': [accuracy_test, recall_test, precision_test, f1_test, fbeta_test, auc_test],
})


# %% [markdown]
# ## Final Feature Importance Chart
# - L1 norms (N2) surprisingly was a good predictor for fraud

# %%
# Get feature importances (by split count) and combine with column names
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb.booster_.feature_importance(importance_type='split')
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(
    data=feature_importance,
    y='Feature',
    x='Importance',
    palette='viridis'
)
plt.title('LightGBM Feature Importance (by Split Count)', fontsize=14)
plt.xlabel('Importance (Number of Splits)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print(*feature_importance['Feature'].tolist(), sep=', ')

# %%
lgbm.plot_metric(lgb)

# %%
# !jupytext --set-formats ipynb,py creditcardfraud.ipynb --sync
