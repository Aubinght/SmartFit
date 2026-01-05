from flask import Flask, render_template, request, redirect, url_for
import os


from backend.database import db, Clothing, init_db_app 
from backend.prediction_model_resnet18 import detect_clothing, save_image, detect_color

#from backend.recommender import get_same_colors

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#init database
init_db_app(app) 

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/delete/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    item = Clothing.query.get_or_404(item_id)
    try:
        if os.path.exists(item.image_path):
            os.remove(item.image_path)
        db.session.delete(item)
        db.session.commit()
    except Exception as e:
        print(f"Error during deletion : {e}")
        db.session.rollback()

    return redirect(url_for('wardrobe'))

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
    selected_category = request.args.get('category')
    if selected_category:
        all_clothes = Clothing.query.filter_by(category=selected_category).all()
    else:
        all_clothes = Clothing.query.all()
    #show only available categories
    categories = db.session.query(Clothing.category).distinct().all()
    categories = [c[0] for c in categories]
    return render_template('wardrobe.html', clothes=all_clothes, categories=categories)

@app.route('/recommendation')
@app.route('/recommendation/<int:item_id>')
def recommendation(item_id=None):
    if item_id:
        reference_item = Clothing.query.get_or_404(item_id)
        recommended_clothes = Clothing.query.filter(
            Clothing.color == reference_item.color, 
            Clothing.id != item_id
        ).all()
        return render_template('recommendation.html', 
                               reference_item=reference_item, 
                               suggestions=recommended_clothes)
    else:
        all_clothes = Clothing.query.all()
        return render_template('recommendation.html', 
                               reference_item=None, 
                               all_clothes=all_clothes)
if __name__ == '__main__':
    app.run(debug=True)
