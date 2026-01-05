from flask import Flask, render_template, request, redirect, url_for, flash
import os

#importation
from backend.database import db, Clothing, init_db_app
from backend.prediction_model_resnet18 import detect_clothing, save_image, detect_color, CATEGORIES

app = Flask(__name__)
app.secret_key = 'secret_key'
#configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#init database
init_db_app(app) 

#sauvegarde des images uploadées
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

#Upload page (step 1: upload, confirmation)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    item_color = None
    #if POST (=upload of an image)
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            #save the image
            filepath = save_image(file)
            
            #detect the catégorie with imported function detect_clothing
            category = detect_clothing(filepath)
            item_color = detect_color(filepath)

            #systematic confirmation
            return render_template(
                'upload.html',
                predicted_category=category,
                filepath=filepath,
                item_color=item_color,
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
    item_color = request.form.get('item_color')

    if not final_category or not filepath:
        flash("Error: Confirmation data incomplete. Please try again", "error")
        return redirect(url_for('upload'))
    
    #save (sql)
    try:
        new_item = Clothing(image_path=filepath, category=final_category, color=item_color)
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