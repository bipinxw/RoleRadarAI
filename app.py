import os
import re
import random
import concurrent.futures
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from serpapi import GoogleSearch
import pdfplumber
from dateutil.parser import parse
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# -------------------------------
# Configuration
# -------------------------------
JOB_TIMEOUT_SECONDS = 12          # per‑job timeout
time_24_hours_ago = datetime.now() - timedelta(days=1)
formatted_time = time_24_hours_ago.strftime('%Y-%m-%d')

# Temporary storage for fetched jobs
job_listings_temp = []

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load spaCy (download model if not present)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Flask app
app = Flask(__name__)
CORS(app)  # not strictly needed when frontend is served from same origin, but harmless

# -------------------------------
# Helper functions (unchanged from your original)
# -------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/17.17134",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0"
]

def fetch_job_description(job_url, req_timeout=8):
    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        response = requests.get(job_url, headers=headers, timeout=req_timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching job description ({job_url}): {e}")
        return ""

def format_posting_date(posting_time):
    try:
        posting_date = parse(posting_time)
        now = datetime.now()
        diff = now - posting_date
        if diff < timedelta(hours=1):
            return f"{int(diff.total_seconds() // 60)} minutes ago"
        elif diff < timedelta(days=1):
            return f"{int(diff.total_seconds() // 3600)} hours ago"
        elif diff < timedelta(days=7):
            return f"{int(diff.total_seconds() // 86400)} days ago"
        else:
            return posting_date.strftime("%Y-%m-%d")
    except Exception:
        return "Unknown date"

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def extract_skills_from_text_with_ai(text):
    doc = nlp(text)
    extracted_skills = []
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'NORP', 'GPE']:
            extracted_skills.append(ent.text)
    return list(set(extracted_skills))

def generate_ai_explanation(resume_text, job_description, resume_embeddings=None):
    if not resume_text or not job_description:
        return "No relevant data to match"
    if resume_embeddings is None:
        resume_embeddings = get_bert_embeddings(resume_text)
    job_embeddings = get_bert_embeddings(job_description)
    similarity_score = cosine_similarity(resume_embeddings.numpy(), job_embeddings.numpy())[0][0]
    job_skills = extract_skills_from_text_with_ai(job_description)
    resume_skills = extract_skills_from_text_with_ai(resume_text)
    explanation = f"The job description matches your resume with a similarity score of {similarity_score:.2f}. "
    matched_skills = set(job_skills).intersection(set(resume_skills))
    if matched_skills:
        explanation += f"The job requires the following key skills: {', '.join(matched_skills)}. These skills are reflected in your resume, showcasing a solid match in relevant areas."
    return explanation

def _process_single_job(job, resume_text, resume_embeddings=None, req_timeout=8):
    name = job.get("name")
    link = job.get("link")
    posted_date = job.get("posted_date")
    job_description = fetch_job_description(link, req_timeout=req_timeout)
    if not job_description:
        return {"name": name, "link": link, "explanation": f"SKIPPED: failed to fetch job description.", "posted_date": posted_date, "score": 0.0}
    if "too many jobs to parse" in job_description:
        return {"name": name, "link": link, "explanation": f"SKIPPED: multiple jobs on page.", "posted_date": posted_date, "score": 0.0}
    job_description_trimmed = job_description[:3000]
    explanation = generate_ai_explanation(resume_text, job_description_trimmed, resume_embeddings=resume_embeddings)
    score = 0.0
    match = re.search(r"similarity score of (\d+\.\d+)", explanation)
    if match:
        try:
            score = float(match.group(1))
        except:
            score = 0.0
    return {"name": name, "link": link, "explanation": explanation, "posted_date": posted_date, "score": float(score)}

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_jobs', methods=['POST'])
def fetch_jobs():
    try:
        if 'designation' not in request.form:
            return jsonify({"message": "Missing 'designation' in the request."}), 400
        designation = request.form['designation']
        api_key = os.environ.get('SERPAPI_KEY')
        if not api_key:
            return jsonify({"message": "Server configuration error: missing SerpAPI key."}), 500

        query = f'site:/* OR site:/careers/* "{designation}" AND ("Remote" OR "Anywhere" OR "WFH" OR "Work from home") AND "India" after:{formatted_time}'
        search_params = {"q": query, "api_key": api_key, "engine": "google"}
        search = GoogleSearch(search_params)
        results = search.get_dict()
        jobs = [
            {"name": r.get("title"), "link": r.get("link"), "posted_date": format_posting_date(r.get("date")), "explanation": ""}
            for r in results.get("organic_results", [])
            if r.get("title") and r.get("link")
        ]
        if not jobs:
            return jsonify({"message": "No jobs found."}), 404
        global job_listings_temp
        job_listings_temp = jobs
        return jsonify(jobs)
    except Exception as e:
        print(f"Error in /fetch_jobs: {e}")
        return jsonify({"message": "Internal server error."}), 500

@app.route('/process_resume', methods=['POST'])
def process_resume():
    if "resume" not in request.files:
        return jsonify({"message": "Missing 'resume' in the request."}), 400
    resume_file = request.files["resume"]
    if resume_file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_file)
    else:
        resume_text = resume_file.read().decode("utf-8", errors="ignore")
    if not job_listings_temp:
        return jsonify({"message": "No job listings available to score."}), 400
    top_jobs = []
    try:
        resume_embeddings = get_bert_embeddings(resume_text)
    except Exception as e:
        print(f"Error computing resume embeddings: {e}")
        resume_embeddings = None
    max_workers = min(4, max(1, len(job_listings_temp)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(_process_single_job, job, resume_text, resume_embeddings, 8): job
            for job in job_listings_temp
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result(timeout=JOB_TIMEOUT_SECONDS)
                if result:
                    top_jobs.append(result)
            except concurrent.futures.TimeoutError:
                top_jobs.append({
                    "name": job.get("name"),
                    "link": job.get("link"),
                    "explanation": f"SKIPPED: processing exceeded {JOB_TIMEOUT_SECONDS} seconds.",
                    "posted_date": job.get("posted_date"),
                    "score": 0.0
                })
            except Exception as e:
                print(f"Error processing job {job.get('name')}: {e}")
                top_jobs.append({
                    "name": job.get("name"),
                    "link": job.get("link"),
                    "explanation": f"SKIPPED: error during processing.",
                    "posted_date": job.get("posted_date"),
                    "score": 0.0
                })
    if not top_jobs:
        return jsonify({"message": "No top jobs found for your resume."}), 404
    sorted_top_jobs = sorted(top_jobs, key=lambda x: x['score'], reverse=True)
    return jsonify(sorted_top_jobs)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))