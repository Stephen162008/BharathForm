import os
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI
from langdetect import detect, DetectorFactory
from flask_dance.contrib.google import make_google_blueprint, google

# ---------------------------------------------------------
# INIT
# ---------------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = "bharatform-secret-key"

# ---------------------------------------------------------
# GOOGLE LOGIN
# ---------------------------------------------------------
google_bp = make_google_blueprint(
    client_id="2166503653-aif0f0v40of1si6g4fj9ugsupmkb09m6.apps.googleusercontent.com",
    client_secret="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",
    scope=["profile", "email"]
)
app.register_blueprint(google_bp, url_prefix="/login")

# ---------------------------------------------------------
# UPLOAD CONFIG
# ---------------------------------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DetectorFactory.seed = 0
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# OCR CONFIG
# ---------------------------------------------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

POPPLER_PATH_ARG = POPPLER_PATH if os.path.exists(POPPLER_PATH) else None

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    user = None
    if google.authorized:
        resp = google.get("/oauth2/v2/userinfo")
        user = resp.json()
    return render_template("index.html", user=user)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("document")
    if not file or file.filename == "":
        return "Please upload a document."

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    extracted_text = ""

    if filename.lower().endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
        except Exception:
            pass

        if not extracted_text.strip():
            images = convert_from_path(file_path, dpi=400, poppler_path=POPPLER_PATH_ARG)
            for img in images:
                extracted_text += pytesseract.image_to_string(img)

    elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(img)

    else:
        return "Unsupported file type."

    if not extracted_text.strip():
        return "Could not extract text."

    user_lang = request.form.get("language", "English")
    explanation = get_ai_explanation(extracted_text, user_lang)

    return render_template("result.html", explanation=explanation)

# ---------------------------------------------------------
# AI LOGIC
# ---------------------------------------------------------
# ---------------------------------------------------------
# AI LOGIC (HYBRID: GPT-3.5 + GPT-4o)
# ---------------------------------------------------------
def get_ai_explanation(text, target_lang):
    try:
        # Rough token estimate (1 token ≈ 4 characters)
        estimated_tokens = len(text) // 4

        # Hybrid model selection
        if estimated_tokens < 1500:
            model_name = "gpt-3.5-turbo"
        else:
            model_name = "gpt-4o"

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"""
Explain the following document in {target_lang}.
Use very simple language.

DOCUMENT:
{text}
"""
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
