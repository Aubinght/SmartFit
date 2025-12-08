from flask import Flask, render_template, request
import os


from backend.database import db, Clothing, init_db_app 
from backend.prediction_model_resnet18 import detect_clothing, save_image, detect_color

from backend.recommender import get_same_colors

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#init database
init_db_app(app) 

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#homepage
@app.route('/') 
def homepage():
    return render_template('homepage.html')

#Upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    category = None
    filepath = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file :
            filepath = save_image(file)
            category = detect_clothing(filepath)
            item_color = detect_color(filepath)
            new_item = Clothing(image_path=filepath, category=category, color=item_color)
            db.session.add(new_item)
            db.session.commit()
        
    return render_template('upload.html',predicted_category=category, filepath=filepath)

@app.route('/wardrobe')
def wardrobe():
    all_clothes = Clothing.query.all()
    
    return render_template('wardrobe.html', clothes=all_clothes)

@app.route('/recommendation/<color>')
def recommendation(color):
    recommended_clothes = get_same_colors(color)
    return render_template('recommendation.html', recommendations=recommended_clothes)

if __name__ == '__main__':
    app.run(debug=True)
