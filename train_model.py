import pandas as pd, numpy as np, json, joblib, os, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE,'models')

print("Loading dataset...")
df = pd.read_csv(os.path.join(BASE,'data/dataset.csv'))
df.fillna(df.select_dtypes(include=np.number).median(), inplace=True)

le_item   = LabelEncoder()
le_season = LabelEncoder()
df['Item_enc']   = le_item.fit_transform(df['Item'].astype(str).str.strip())
df['Season_enc'] = le_season.fit_transform(df['Season'].astype(str).str.strip())

crop_map   = dict(zip(le_item.classes_, le_item.transform(le_item.classes_).tolist()))
season_map = dict(zip(le_season.classes_, le_season.transform(le_season.classes_).tolist()))

feature_cols = ['avg_rainfall','pesticides_tonnes','avg_temp','Radiation','Season_enc','Item_enc']
X = df[feature_cols]
y = df['hg/ha_yield']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

models = {
    "Random Forest":  RandomForestRegressor(n_estimators=200,random_state=42),
    "SVR":            SVR(kernel='rbf',C=100,gamma=0.1,epsilon=0.1),
    "KNN":            KNeighborsRegressor(n_neighbors=7,weights='distance'),
    "Decision Tree":  DecisionTreeRegressor(max_depth=12,random_state=42),
    "XGBoost":        XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=7,random_state=42,verbosity=0),
    "LightGBM":       LGBMRegressor(n_estimators=300,learning_rate=0.05,max_depth=7,random_state=42,verbose=-1),
    "CatBoost":       CatBoostRegressor(iterations=300,learning_rate=0.05,depth=7,random_state=42,verbose=0),
}

results = {}
trained = {}
print("\nTraining 7 models...")
for name, m in models.items():
    m.fit(X_train_s, y_train)
    yp = m.predict(X_test_s)
    mse = mean_squared_error(y_test, yp)
    mae = mean_absolute_error(y_test, yp)
    r2  = r2_score(y_test, yp)
    results[name] = {'MSE':round(mse,2),'MAE':round(mae,2),'R2':round(r2,4)}
    trained[name] = m
    print(f"  {name:20s} | R²={r2:.4f} | MSE={mse:,.0f} | MAE={mae:,.0f}")

# SHAP on CatBoost (best model per document)
print("\nComputing SHAP values for CatBoost...")
cat_model = trained['CatBoost']
explainer = shap.Explainer(cat_model, X_train_s)
shap_vals = explainer(X_test_s[:200])
feat_labels = ['avg_rainfall','pesticides_tonnes','avg_temp','Radiation','Season','Crop']
mean_shap = np.abs(shap_vals.values).mean(axis=0).tolist()
shap_importance = dict(zip(feat_labels, mean_shap))
print("SHAP importance:", {k:round(v,4) for k,v in shap_importance.items()})

# Actual vs Predicted data (CatBoost)
y_pred_cat = cat_model.predict(X_test_s)
actual_vs_pred = [{'actual':round(float(a),2),'predicted':round(float(p),2)}
                  for a,p in zip(y_test.values[:100], y_pred_cat[:100])]

# Residuals data
residuals_data = [{'predicted':round(float(p),2),'residual':round(float(a-p),2)}
                  for a,p in zip(y_test.values[:100], y_pred_cat[:100])]

# Save all models
for name, m in trained.items():
    safe = name.replace(' ','_')
    joblib.dump(m, os.path.join(OUT,f'{safe}.pkl'))
joblib.dump(scaler,   os.path.join(OUT,'scaler.pkl'))
joblib.dump(le_item,  os.path.join(OUT,'le_item.pkl'))
joblib.dump(le_season,os.path.join(OUT,'le_season.pkl'))

# Crop / season yield stats
crop_avg   = df.groupby('Item')['hg/ha_yield'].mean().round(2).to_dict()
crop_min   = df.groupby('Item')['hg/ha_yield'].min().round(2).to_dict()
crop_max   = df.groupby('Item')['hg/ha_yield'].max().round(2).to_dict()
season_avg = df.groupby('Season')['hg/ha_yield'].mean().round(2).to_dict()

meta = {
    'crops':        le_item.classes_.tolist(),
    'seasons':      le_season.classes_.tolist(),
    'crop_map':     crop_map,
    'season_map':   season_map,
    'feature_cols': feature_cols,
    'model_results':results,
    'shap_importance': shap_importance,
    'actual_vs_pred':  actual_vs_pred,
    'residuals_data':  residuals_data,
    'crop_avg': crop_avg, 'crop_min': crop_min, 'crop_max': crop_max,
    'season_avg': season_avg,
    'dataset_stats':{
        'total': len(df),
        'avg_yield': round(float(df['hg/ha_yield'].mean()),2),
        'max_yield': round(float(df['hg/ha_yield'].max()),2),
        'min_yield': round(float(df['hg/ha_yield'].min()),2),
    }
}
with open(os.path.join(OUT,'meta.json'),'w') as f:
    json.dump(meta,f,indent=2)

print("\n✅ All 7 models + SHAP + meta saved!")
for n,r in results.items():
    print(f"  {n:20s} R²={r['R2']}  MSE={r['MSE']:,.0f}  MAE={r['MAE']:,.0f}")
