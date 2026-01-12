from flask import Flask, render_template, request, flash, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash, generate_password_hash

import os

from backend.prediction_model_resnet18 import detect_clothing, add_to_wardrobe_json, save_image
from backend.database import db, Clothing, init_db_app, User

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///macollection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#init database
init_db_app(app) 

#The login manager permits to keep track of who is logged in 
# as they click through different pages
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


#signup
# Takes an username and password, hashes the password
# and save the new user to "mycollection.db"
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')


#login
# Checks if the username exists and if the hashed password matches what was typed
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('wardrobe'))
        flash('Invalid username or password')
    return render_template('login.html')

#logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('homepage'))

# dossier de sauvegarde des images uploadées
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#homepage
@app.route('/') 
def homepage():
    return render_template('homepage.html')

#Upload page
# Receive the image file
# Uses our model "detect_clothing" to guess what it is
# Saves the image and the guess into the database, linked to the current user
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    category = None
    filepath = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file :
            filepath = save_image(file)
            category = detect_clothing(filepath)
            new_item = Clothing(image_path=filepath, category=category, user_id=current_user.id)
            db.session.add(new_item)
            db.session.commit()
        
    return render_template('upload.html',predicted_category=category, filepath=filepath)

# Closet 
# Displays all the clothes where "user_id" matches the user who is logged in
@app.route('/wardrobe')
@login_required
def wardrobe():
    user_clothes = Clothing.query.filter_by(user_id=current_user.id).all()
    return render_template('wardrobe.html', clothes=user_clothes)
    
if __name__ == '__main__':
    app.run(debug=True)
