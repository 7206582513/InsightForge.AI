# === Unified Flask App: InsightForge AI Assistant ===

from flask import Flask, render_template, request, send_file, session
import os
import pandas as pd
import pytesseract
import cv2
import numpy as np
import requests
from werkzeug.utils import secure_filename
from pdf2image import convert_from_bytes
from langdetect import detect
from modules.eda_pipeline import auto_eda_pipeline
from modules.model_pipeline import train_best_model
from modules.insight_refiner import generate_questions, clean_and_structure

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
GROQ_API_KEY = "Your_GROQ_API_KEY"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
SECRET_KEY = 'your_secret_key_here'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# === OCR Utilities ===
def extract_chart_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 150:
            chart_img = image[y:y + h, x:x + w]
            cropped.append(chart_img)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("static/charts/ocr_overlay.png", image)
    return cropped

def ocr_chart(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        images = convert_from_bytes(f.read())
    full_text = ''
    for page in images:
        np_img = np.array(page)
        cropped_charts = extract_chart_regions(np_img)
        for chart_img in cropped_charts:
            full_text += ocr_chart(chart_img) + "\n"
    return full_text

def load_dataset(csv_path):
    try:
        return pd.read_csv(csv_path)
    except:
        return None

def generate_insight_with_llm(chart_text, df):
    df_preview = df.head(10).to_string() if df is not None else "No dataset available."
    prompt = f"""
You are a data analyst AI. Here is some text extracted from chart regions in a dashboard:

--- Chart Text ---
{chart_text}

--- CSV Dataset Preview ---
{df_preview}

Please generate 3-5 meaningful business insights based on trends shown in the charts.
"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else f"Error: {response.text}"

def ask_groq_about_chart(question, context):
    lang = detect(question)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Language: {lang.upper()}\nContext: {context}\n\nUser: {question}"
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'] if response.status_code == 200 else f"Error: {response.text}"

# === ROUTES ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        task_type = request.form.get('task_type')
        target_col = request.form.get('target_col')
        pdf_file = request.files.get('pdf_file')

        if not (file and task_type and target_col):
            return "❌ Missing required fields."

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath, encoding='utf-8', engine='python')
        df.columns = df.columns.str.strip()
        target_col = target_col.strip()

        if target_col not in df.columns:
            return f"❌ Error: Target column '{target_col}' not found."

        if target_col.lower() == 'price':
            df[target_col] = df[target_col].astype(str).str.replace(',', '').replace({'Ask For Price': None})
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        clean_df, eda_summary = auto_eda_pipeline(df, task_type=task_type, target_col=target_col)
        clean_path = os.path.join(OUTPUT_FOLDER, "cleaned_data.csv")
        clean_df.to_csv(clean_path, index=False)
        best_model, report = train_best_model(clean_df, task_type=task_type)

        # === Insight from EDA report PDF ===
        eda_pdf_path = "outputs/eda_report.pdf"
        if os.path.exists(eda_pdf_path):
            eda_chart_text = extract_text_from_pdf(eda_pdf_path)
            eda_insight = generate_insight_with_llm(eda_chart_text, clean_df)
            report["EDA Chart Insight"] = clean_and_structure(eda_insight)
            report["EDA Suggested Questions"] = generate_questions(eda_insight)

        # === Insight from additional Power BI PDF ===
        if pdf_file:
            pdf_path = os.path.join(UPLOAD_FOLDER, secure_filename(pdf_file.filename))
            pdf_file.save(pdf_path)
            powerbi_text = extract_text_from_pdf(pdf_path)
            powerbi_insight = generate_insight_with_llm(powerbi_text, clean_df)
            report["Power BI Chart Insight"] = clean_and_structure(powerbi_insight)
            report["PowerBI Suggested Questions"] = generate_questions(powerbi_insight)

        return render_template("result.html", report=report, clean_path=clean_path)

    except Exception as e:
        return f"❌ Error: {str(e)}"

@app.route('/chart-talk', methods=['GET', 'POST'])
def chart_talk():
    if 'chat_history' not in session:
        session['chat_history'] = []

    insight = session.get("insight", "")
    if request.method == 'POST':
        if 'pdf_file' in request.files and 'csv_file' in request.files:
            pdf_file = request.files['pdf_file']
            csv_file = request.files['csv_file']

            pdf_path = os.path.join(UPLOAD_FOLDER, secure_filename(pdf_file.filename))
            csv_path = os.path.join(UPLOAD_FOLDER, secure_filename(csv_file.filename))
            pdf_file.save(pdf_path)
            csv_file.save(csv_path)

            chart_text = extract_text_from_pdf(pdf_path)
            df = load_dataset(csv_path)
            insight = generate_insight_with_llm(chart_text, df)

            session['insight'] = insight
            session['chat_history'] = []
            session.modified = True

        elif request.form.get("question"):
            question = request.form.get("question")
            context = session.get("insight", "")
            reply = ask_groq_about_chart(question, context)

            if 'chat_history' not in session:
                session['chat_history'] = []

            session['chat_history'].append((question, reply))
            session.modified = True

    return render_template('chart_talk.html',
                           insight=session.get('insight', ''),
                           chat_history=session.get('chat_history', []))

@app.route('/ask-question', methods=['POST'])
def ask_question():
    try:
        context = request.form.get('context', '')
        question = request.form.get('question', '')
        reply = ask_groq_about_chart(question, context)

        # ✅ Store into session history for both /chart-talk and /upload views
        if 'chat_history' not in session:
            session['chat_history'] = []

        session['chat_history'].append((question, reply))
        session.modified = True  # ✅ Required to persist in Flask

        return {'answer': reply}
    except Exception as e:
        return {'answer': f"Error: {str(e)}"}


@app.route('/download_chat')
def download_chat():
    chat = session.get('chat_history', [])
    if not chat:
        return "❌ No chat history found yet. Please ask at least one question first."

    file_path = os.path.join(OUTPUT_FOLDER, "chat_history.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("🧠 InsightForge.AI - Chat Q&A History\n\n")
        for q, a in chat:
            f.write(f"Q: {q}\nA: {a}\n\n")
    return send_file(file_path, as_attachment=True)

@app.route('/download')
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, 'cleaned_data.csv'), as_attachment=True)

@app.route('/download_pdf')
def download_pdf():
    return send_file("outputs/eda_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
