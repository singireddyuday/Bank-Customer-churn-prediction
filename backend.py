from flask import Flask, request, render_template
import joblib
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly
import json

app = Flask(__name__)

# Load model and SHAP explainer
model = joblib.load("rf_churn_model.joblib")
explainer = joblib.load("shap_explainer.joblib")

# Encoding maps
geography_map = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_map = {'Female': 0, 'Male': 1}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    data = {
        'CreditScore': float(form['CreditScore']),
        'Geography': geography_map[form['Geography']],
        'Gender': gender_map[form['Gender']],
        'Age': float(form['Age']),
        'Tenure': float(form.get('Tenure', 0)),
        'Balance': float(form['Balance']),
        'NumOfProducts': float(form['NumOfProducts']),
        'HasCrCard': int('HasCrCard' in form),
        'IsActiveMember': int('IsActiveMember' in form),
        'EstimatedSalary': float(form.get('EstimatedSalary', 0))
    }

    df = pd.DataFrame([data])
    df = df[model.feature_names_in_]

    prob = model.predict_proba(df)[0][1]
    confidence = max(model.predict_proba(df)[0])
    risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'

    shap_vals_all = explainer.shap_values(df)
    shap_vals = shap_vals_all[1][0] if isinstance(shap_vals_all, list) else shap_vals_all[0]
    shap_vals = shap_vals.flatten()

    expected_features = model.feature_names_in_
    importance = {f: abs(v) for f, v in zip(expected_features, shap_vals)}
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    top_features = [
        {'feature': f, 'contribution': float(shap_vals[list(expected_features).index(f)])}
        for f, _ in sorted_feats[:3]
    ]

    # Charts
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=list(expected_features),
        y=df.iloc[0].values,
        mode='lines+markers',
        line=dict(color='lightgreen')
    ))
    line_fig.update_layout(template='plotly_dark', height=500)

    pie_fig = go.Figure(data=[go.Pie(
        labels=[f for f, _ in sorted_feats[:5]],
        values=[v for _, v in sorted_feats[:5]],
        hole=0.3
    )])
    pie_fig.update_layout(template='plotly_dark', height=500)

    bar_data = {
        'Feature': [f for f, _ in sorted_feats[:5]],
        'Value': [v for _, v in sorted_feats[:5]]
    }
    bar_fig = px.bar(
        bar_data, x='Feature', y='Value',
        title='Feature Importance Bar Chart',
        color_discrete_sequence=['#a889f4']
    )
    bar_fig.update_layout(template='plotly_dark', height=500)

    return render_template("result.html",
        churn_prob=f"{prob * 100:.1f}",
        confidence=f"{confidence * 100:.1f}",
        risk_level=risk_level,
        insight="Limited product engagement increases churn likelihood.",
        top_features=top_features,
        line_chart_json=json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder),
        pie_chart_json=json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder),
        bar_chart_json=json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder),
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, reloader_type='stat')
