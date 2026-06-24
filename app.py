from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from datetime import datetime, timedelta

from flask_sqlalchemy import SQLAlchemy
#from transformers import pipeline


# load once globally
#classifier = pipeline("text-classification")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Database Models ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    # Relationship to link history to users
    history = db.relationship('SearchHistory', backref='author', lazy=True)

def get_local_time():
    # Adjusts UTC to IST (UTC + 5 hours 30 minutes)
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    # Change default to your new local time function
    timestamp = db.Column(db.DateTime, default=get_local_time) 
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- ML Model Setup ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_detection_model")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Fake Job 🚨" if predicted_class == 1 else "Real Job ✅"


# --- Routes ---

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        hashed_pw = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=request.form['username'], password=hashed_pw)
        try:
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username already exists.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Check username and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    prediction = ""
    if request.method == 'POST':
        text = request.form['job_text']
        prediction = predict(text)
        
        # SAVE TO HISTORY
        new_entry = SearchHistory(job_text=text, prediction=prediction, user_id=current_user.id)
        db.session.add(new_entry)
        db.session.commit()

    # FETCH HISTORY (Latest 10 searches)
    history = SearchHistory.query.filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).limit(10).all()
    
    return render_template('index.html', prediction=prediction, history=history)

@app.route('/delete-history/<int:id>')
@login_required
def delete_history(id):
    history_item = SearchHistory.query.get_or_404(id)
    # Security check: Ensure the current user owns this history item
    if history_item.user_id == current_user.id:
        db.session.delete(history_item)
        db.session.commit()
        flash('History record deleted.', 'info')
    return redirect(url_for('home'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)