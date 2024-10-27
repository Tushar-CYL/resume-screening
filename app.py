# app.py
from flask import Flask, render_template, request, jsonify
import docx2txt
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
ALLOWED_EXTENSIONS = {'docx'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained models
clf = pickle.load(open('uploads/clf.pkl', 'rb'))
tfidf = pickle.load(open('uploads/tfidf.pkl', 'rb'))

# Category mapping
CATEGORY_MAPPING = {
    0: "Advocate",
    1: "Arts",
    2: "Automation Testing",
    3: "Blockchain",
    4: "Business Analyst",
    5: "Civil Engineer",
    6: "Data Science",
    7: "Database",
    8: "DevOps Engineer",
    9: "DotNet Developer",
    10: "ETL Developer",
    11: "Electrical Engineering",
    12: "HR",
    13: "Hadoop",
    14: "Health and fitness",
    15: "Android Developer",
    16: "Mechanical Engineer",
    17: "Network Security Engineer",
    18: "Operations Manager",
    19: "PMO",
    20: "Python Developer",
    21: "SAP Developer",
    22: "Sales",
    23: "Testing",
    24: "Web Designing"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def predict_category(text):
    input_features = tfidf.transform([text])
    prediction_id = clf.predict(input_features)[0]
    return CATEGORY_MAPPING.get(prediction_id, "Unknown")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only DOCX files are allowed'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        resume_text = extract_text_from_docx(filepath)
        category = predict_category(resume_text)
        
        # Remove the temporary file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'resume_text': resume_text,
            'category': category
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
