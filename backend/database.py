from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Clothing(db.Model):
    id = db.Column(db.Integer, primary_key=True)      # SQL primary key
    image_path = db.Column(db.String(300))
    category = db.Column(db.String(100))
    color = db.Column(db.String(50))

# called in app.py to init databse
def init_db_app(app):
    db.init_app(app)
    # Create tabs if don't exist
    with app.app_context():
        db.create_all()