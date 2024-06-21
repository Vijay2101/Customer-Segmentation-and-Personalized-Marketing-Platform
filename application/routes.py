from application import app, db
from flask import render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import xgboost as xgb
import joblib
from application import mail
from application import profiles
from datetime import datetime
import json
from application.models import survey






# Load XGBoost model and scaler
model = xgb.XGBClassifier()
model.load_model('application/model/model.xgb')
scaler = joblib.load('application/scaler/scaler.pkl')

def preprocess_input(user_input):
    columns = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 'Work_Experience', 'Spending_Score', 'Family_Size']
    
    gender_map = {'Female': 0, 'Male': 1}
    ever_married_map = {'No': 0, 'Prefer not to disclose': 1, 'Yes': 2}
    graduated_map = {'No': 0, 'Prefer not to disclose': 1, 'Yes': 2}
    profession_map = {'Artist': 0, 'Doctor': 1, 'Engineer': 2, 'Entertainment': 3, 'Executive': 4, 'Healthcare': 5, 'Homemaker': 6, 'Lawyer': 7, 'Marketing': 8, 'Others': 9}
    spending_score_map = {'Average': 0, 'High': 1, 'Low': 2}

    df = pd.DataFrame([user_input], columns=columns)
    df['Gender'] = df['Gender'].map(gender_map)
    df['Ever_Married'] = df['Ever_Married'].map(ever_married_map)
    df['Graduated'] = df['Graduated'].map(graduated_map)
    df['Profession'] = df['Profession'].map(profession_map)
    df['Spending_Score'] = df['Spending_Score'].map(spending_score_map)
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df

def add_person_to_db(data, prediction):

    entry = survey(
        date=datetime.utcnow(),
        name=data['name'],
        email=data['user_email'],
        gender=data['gender'],
        ever_married=data['ever_married'],
        age=data['age'],
        graduated=data['graduated'],
        profession=data['profession'],
        work_experience=data['work_experience'],
        spending_score=data['spending_score'],
        family_size=data['family_size'],
        cluster=prediction
    )
    db.session.add(entry)
    db.session.commit()

def survey_mail(email,name,pred):
    mail_status = mail.send_email(email, name, pred)
    return mail_status
    

@app.route('/')
def show():
    entries = survey.query.order_by(survey.date.desc()).all()
    return render_template('show.html',title="Home", entries=entries)

@app.route("/delete/<int:entry_id>")
def delete(entry_id):
    entry = survey.query.get_or_404(int(entry_id))
    db.session.delete(entry)
    db.session.commit()
    flash("Survey Data Deleted!!",'primary')
    return redirect(url_for("show"))

@app.route("/individual_mail/<string:email>/<string:name>/<string:pred>")
def individual_mail(email,name,pred):
    survey_mail(email, name, pred)
    flash(f"Mail sent to {name}, {email}!!",'primary')
    return redirect(url_for("show"))

@app.route("/dashboard")
def dashboard():
    cluster_data = db.session.query(db.func.count(survey.cluster),survey.cluster).group_by(survey.cluster).order_by(survey.cluster).all()

    count = []
    cluster_label = []
    for cluster_count, cluster in cluster_data:
        count.append(cluster_count)
        cluster_label.append(cluster)

    prof_data = db.session.query(db.func.count(survey.profession),survey.profession).group_by(survey.profession).order_by(survey.profession).all()

    prof_counts = []
    prof_label = []
    for prof_count, prof in prof_data:
        prof_counts.append(prof_count)
        prof_label.append(prof)

    print(prof_counts,"dfd",prof_label)
        
    return render_template("dashboard.html",title="Dashboard",
                           count=json.dumps(count),
                           cluster_label = json.dumps(cluster_label),
                           prof_count=json.dumps(prof_counts),
                           prof_label = json.dumps(prof_label))


@app.route('/survey')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = [
        data['gender'], data['ever_married'], data['age'], data['graduated'],
        data['profession'], data['work_experience'], data['spending_score'], data['family_size']
    ]
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)[0]
    
    
    if prediction == 0:
        pred = 'cluster A'
        profile = profiles.cluster_A
    elif prediction == 1:
        pred = 'cluster B'
        profile = profiles.cluster_B
    elif prediction == 2:
        pred = 'cluster C'
        profile = profiles.cluster_C
    else:
        pred = 'cluster D'
        profile = profiles.cluster_D
    
    response = {
        'prediction': pred,
        'profile': profile
    }
    add_person_to_db(data, pred)
    
    mail_status = survey_mail(data['user_email'],data['name'],pred)

    response['mail_status'] = 'success' if mail_status else 'failure'
    
    return jsonify(response)





if __name__ == '__main__':
    app.run(debug=True)

