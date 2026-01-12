from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Clothing(db.Model):
    id = db.Column(db.Integer, primary_key=True)      # SQL primary key
    image_path = db.Column(db.String(300))
    category = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) 

# called in app.py to init databse
def init_db_app(app):
    db.init_app(app)
    # Create tabs if don't exist
    with app.app_context():
        db.create_all()

# User table to store user infos
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  
    clothes = db.relationship('Clothing', backref='owner', lazy=True)