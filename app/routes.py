from app import app
from app import db
from flask import render_template, flash, make_response

from app.forms import DiabetesForm, MalariaForm, BrainTumorForm

from app.diabetesDetection.prediction import DiabetesModel
from app.malariaDetection.prediction import MalariaDetection
from app.brainTumorDetection.prediction import TumorDetection

from app.models import History

diabetesModel = DiabetesModel()
malariaModel = MalariaDetection()
brainTumorModel = TumorDetection()
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/diabetes-detection', methods=['GET', 'POST'])
def diabetesDetection():
    form = DiabetesForm()
    if form.validate_on_submit():
        res = diabetesModel.predict(form.glucose.data, form.insulin.data, form.bmi.data, form.age.data)
        outcome = "diabetic" if int(res[0])==1 else "not diabetic"
        prob = round(float(res[1])*100, 2)
        msg = 'There is a {} % chance that you are {}'.format(prob, outcome)
        h = History(model = "Diabetes Detection", result = msg)
        db.session.add(h)
        db.session.commit()
        return render_template('diabetes-detection.html', title= ("Result - Diabetes Detection"), form=form, result = msg)
    return render_template('diabetes-detection.html', title="Diabetes Detection", form=form)    

@app.route('/malaria-detection', methods=['GET', 'POST'])
def malariaDetection():
    form = MalariaForm()
    if form.validate_on_submit():
        result, original, image = malariaModel.predict(form.upload.data)
        h = History(model = "Malaria Detection", result = result)
        db.session.add(h)
        db.session.commit()
        return render_template('malaria-detection.html', title="Result - Malaria Detection", form=form, result=result, image=image, original=original)
    return render_template('malaria-detection.html', title="Malaria Detection", form=form)    

@app.route('/brain-tumor-detection', methods=['GET', 'POST'])
def brainTumorDetection():
    form = BrainTumorForm()
    if form.validate_on_submit():
        result, original, image = brainTumorModel.predict(form.upload.data)
        h = History(model = "Brain Tumor Detection", result = result)
        db.session.add(h)
        db.session.commit()
        return render_template('brain-tumor-detection.html', title="Result - Brain Tumor Detection", form=form, result=result, image=image, original=original)
    return render_template('brain-tumor-detection.html', title="Brain Tumor Detection", form=form)    
	
@app.route('/view-history')
def history():
    return render_template('view-history.html', title="View History", history=History.query.order_by(History.id.desc()).all())