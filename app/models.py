from datetime import datetime
from app import db

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime,index=False,unique=False,nullable=False,default=datetime.utcnow)
    model = db.Column(db.Text,index=False,unique=False,nullable=False)
    result = db.Column(db.Text,index=False,unique=False,nullable=False)
    def __repr__(self):
        return '<History {}>'.format(self.id)    