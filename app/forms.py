from flask_wtf import FlaskForm
from wtforms import IntegerField, DecimalField, SubmitField
from wtforms.validators import InputRequired
from flask_wtf.file import FileField, FileAllowed, FileRequired

class DiabetesForm(FlaskForm):
    pregnancies = IntegerField('Number of Pregnancies', validators=[InputRequired()]) #int
    glucose = IntegerField('Glucose', validators=[InputRequired()]) #int
    bloodPressure = IntegerField('Blood Pressure', validators=[InputRequired()]) #int
    skinThickness = IntegerField('Skin Thickness', validators=[InputRequired()]) #int
    insulin = IntegerField('Insulin', validators=[InputRequired()]) #int
    bmi = DecimalField('BMI', validators=[InputRequired()]) #float
    diabetes = DecimalField('Diabetes', validators=[InputRequired()]) #float
    age = IntegerField('Age', validators=[InputRequired()]) #int
    submit = SubmitField('Submit')
class MalariaForm(FlaskForm):
    upload = FileField('Attach Image Sample For Processing : ', validators=[FileRequired(),FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    submit = SubmitField('Submit')
class BrainTumorForm(FlaskForm):
    upload = FileField('Attach Image Sample For Processing : ', validators=[FileRequired(),FileAllowed(['jpg', 'jpeg', 'png'], 'Images only!')])
    submit = SubmitField('Submit')