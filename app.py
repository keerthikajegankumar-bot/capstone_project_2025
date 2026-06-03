from flask import Flask, render_template, request, jsonify
import pandas as pd, numpy as np, joblib, json, os, warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_NAMES = ["Random_Forest","SVR","KNN","Decision_Tree","XGBoost","LightGBM","CatBoost"]
models = {}
for n in MODEL_NAMES:
    path = os.path.join(BASE, f'models/{n}.pkl')
    if os.path.exists(path):
        models[n.replace('_',' ')] = joblib.load(path)

scaler    = joblib.load(os.path.join(BASE,'models/scaler.pkl'))
le_item   = joblib.load(os.path.join(BASE,'models/le_item.pkl'))
le_season = joblib.load(os.path.join(BASE,'models/le_season.pkl'))
with open(os.path.join(BASE,'models/meta.json')) as f:
    meta = json.load(f)
df = pd.read_csv(os.path.join(BASE,'data/dataset.csv'))
FEAT_COLS = ['avg_rainfall','pesticides_tonnes','avg_temp','Radiation','Season_enc','Item_enc']

@app.route('/')
def home(): return render_template('home.html', meta=meta)

@app.route('/predict')
def predict_page(): return render_template('predict.html', meta=meta)

@app.route('/analytics')
def analytics_page(): return render_template('analytics.html', meta=meta)

@app.route('/dataset')
def dataset_page(): return render_template('dataset.html', meta=meta)

@app.route('/about')
def about_page(): return render_template('about.html', meta=meta)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        d = request.get_json()
        model_name = d.get('model','CatBoost')
        crop_enc   = int(le_item.transform([d['crop']])[0])
        season_enc = int(le_season.transform([d['season']])[0])
        feats = np.array([[float(d['rainfall']), float(d['pesticides']),
                           float(d['temp']),     float(d['radiation']),
                           season_enc, crop_enc]])
        feats_s = scaler.transform(feats)
        feat_df = pd.DataFrame(feats_s, columns=FEAT_COLS)
        m = models.get(model_name, models['CatBoost'])
        pred = float(m.predict(feat_df)[0])

        # SHAP on CatBoost
        import shap
        cat = models['CatBoost']
        # Use a small background sample for speed
        bg = pd.DataFrame(scaler.transform(
            df[['avg_rainfall','pesticides_tonnes','avg_temp','Radiation']].assign(
                Season_enc=le_season.transform(df['Season'].astype(str).str.strip()),
                Item_enc=le_item.transform(df['Item'].astype(str).str.strip())
            )[FEAT_COLS].values[:50]
        ), columns=FEAT_COLS)
        explainer = shap.TreeExplainer(cat)
        sv = explainer.shap_values(feat_df)[0]
        feat_labels = ['Rainfall','Pesticides','Temperature','Radiation','Season','Crop']
        shap_out = [{'feature': k, 'shap': round(float(v), 1)}
                    for k, v in zip(feat_labels, sv)]
        shap_out.sort(key=lambda x: abs(x['shap']), reverse=True)

        # Recommendations
        recs = []
        for s in shap_out:
            if s['feature'] == 'Temperature' and s['shap'] < -1000:
                recs.append("High temperature is reducing yield. Consider heat-resistant varieties or adjusted sowing dates.")
            if s['feature'] == 'Rainfall' and s['shap'] < -1000:
                recs.append("Low rainfall impact detected. Supplement with drip irrigation for better yield.")
            if s['feature'] == 'Pesticides' and s['shap'] < -500:
                recs.append("Excess pesticide usage may be harming soil health. Reduce dosage to recommended levels.")
            if s['feature'] == 'Radiation' and s['shap'] < -500:
                recs.append("Low solar radiation is limiting photosynthesis. Ensure crop rows face maximum sun exposure.")
        if not recs:
            recs.append("All climate parameters are within optimal range for this crop. Maintain current farming practices.")
        recs.append(f"Model used: {model_name}. For highest accuracy, CatBoost (R²=0.9849) is recommended.")

        crop_avg = float(df[df['Item'] == d['crop']]['hg/ha_yield'].mean())
        comp     = round((pred - crop_avg) / crop_avg * 100, 1)

        return jsonify(success=True,
            pred=round(pred,2), low=round(pred*0.9,2), high=round(pred*1.1,2),
            tonnes=round(pred/10000,3), comparison=comp,
            model=model_name, crop=d['crop'], season=d['season'],
            shap=shap_out, recommendations=recs)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e))

@app.route('/api/analytics')
def api_analytics():
    corr = df[['avg_rainfall','pesticides_tonnes','avg_temp','Radiation','hg/ha_yield']]\
             .corr()['hg/ha_yield'].drop('hg/ha_yield').round(4).to_dict()
    pivot = df.groupby(['Item','Season'])['hg/ha_yield'].mean().round(2).reset_index()
    heatmap = {}
    for crop in df['Item'].unique():
        heatmap[str(crop)] = {}
        for s in df['Season'].unique():
            v = pivot[(pivot['Item']==crop)&(pivot['Season']==s)]['hg/ha_yield']
            heatmap[str(crop)][str(s)] = float(v.values[0]) if len(v) else 0
    return jsonify(
        crop_avg=meta['crop_avg'], crop_min=meta['crop_min'], crop_max=meta['crop_max'],
        season_avg=meta['season_avg'], model_results=meta['model_results'],
        shap_importance=meta['shap_importance'],
        actual_vs_pred=meta['actual_vs_pred'],
        residuals_data=meta['residuals_data'],
        dataset_stats=meta['dataset_stats'],
        corr=corr, heatmap=heatmap
    )

@app.route('/api/dataset')
def api_dataset():
    page = int(request.args.get('page', 1))
    per  = 20
    chunk = df.iloc[(page-1)*per : page*per]
    return jsonify(data=chunk.to_dict(orient='records'),
                   total=len(df), page=page,
                   pages=int(np.ceil(len(df)/per)))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
