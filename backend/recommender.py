#recommand cloth of similar colors.
#first: find all cloth of same colors.

#recreate space of the app.py
from flask import Flask
from backend.database import db, Clothing, init_db_app
from backend.prediction_model_resnet18 import detect_clothing
import random
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

init_db_app(app)


def get_same_colors(color):
    with app.app_context():
        clothes_to_recommend = Clothing.query.filter(Clothing.color==color).all()
        if clothes_to_recommend:
            return(random.choice(clothes_to_recommend))
        return(None)

if __name__ == '__main__':
    image_path = "static/uploads/cloth_20251110_171257_713120.jpg"
    category = detect_clothing(image_path)
    cloth = Clothing(image_path=image_path)
    recommendation = get_same_colors(cloth)
    if recommendation:
         print(f"Recommandation trouvée : {recommendation.category}, couleur: {recommendation.color}, chemin: {recommendation.image_path}")
    else:
        print("No same colors")