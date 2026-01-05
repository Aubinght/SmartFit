from flask import Flask, render_template, request, redirect, url_for, flash
import os

#importation
from backend.model import detect_clothing, save_image
from backend.database import db, Clothing, init_db_app
from backend.prediction_model_resnet18 import CATEGORIES

app = Flask(__name__)

#configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#init database
init_db_app(app) 

#sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#homepage
@app.route('/') 
def homepage():
    return render_template('homepage.html')

#Upload page (step 1: upload, confirmation)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    #if POST (=upload of an image)
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            #save the image
            filepath = save_image(file)
            
            #detect the catégorie with imported function detect_clothing
            category = detect_clothing(filepath)
            
            #systematic confirmation
            return render_template(
                'upload.html',
                predicted_category=category,
                filepath=filepath,
                show_confirmation=True, #display the confirmation form
                categories=CATEGORIES #categories defined in backend
            )
        else:
            #error messages if no file uploaded
            flash("Please select a file", "error")
            return redirect(url_for('upload'))
            
    #For a GET (first access) ou after an error, display the form
    return render_template('upload.html', show_confirmation=False)

#Step 2: validation, save
@app.route('/confirm_add', methods=['POST'])
def confirm_add():
    #get the confirmed data
    final_category = request.form.get('final_category') #category (possibly modified by user with confirmation form)
    filepath = request.form.get('filepath')

    if not final_category or not filepath:
        flash("Error: Confirmation data incomplete. Please try again", "error")
        return redirect(url_for('upload'))
    
    #save (sql)
    try:
        new_item = Clothing(image_path=filepath, category=final_category)
        db.session.add(new_item)
        db.session.commit()
        
        #success message
        flash(f"Your '{final_category}' was added to the wardrobe!", "success")
        return redirect(url_for('wardrobe'))

    except Exception as e:
        #general error message
        flash(f"Database error: {e}", "error")
        return redirect(url_for('upload'))


@app.route('/wardrobe')
def wardrobe():
    all_clothes = Clothing.query.all()
    return render_template('wardrobe.html', clothes=all_clothes)

if __name__ == '__main__':
    app.run(debug=True)