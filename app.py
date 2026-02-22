"""
app.py — Fake Job Posting Detection System
==========================================
Flask web application that serves the ML model for inference.

Run:
    python app.py

Endpoints:
    GET  /           → Home (input form)
    POST /predict    → Run analysis, render results
    GET  /about      → About the system
    GET  /health     → JSON health check
"""

import os
import re
import joblib
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import shap

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# ─── Model Paths ───────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

# ─── Load Model at Startup (not per request) ───────────────────────────────────
model      = None
vectorizer = None

def load_model():
    global model, vectorizer
    try:
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("✅ Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        logger.warning(
            "⚠️  Model files not found. Run `python train.py` first. "
            "Running in DEMO mode."
        )

# Create explainer once at startup (not per request)
explainer = None
def load_explainer():
    global explainer
    if model is not None and vectorizer is not None:
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(
            vectorizer.transform([""]), max_samples=100
        ))
        logger.info("✅ SHAP explainer loaded.")

load_model()
load_explainer()

# ─── Domain Rule Engine ────────────────────────────────────────────────────────
FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "mail.com", "icloud.com", "protonmail.com",
    "yandex.com", "zoho.com", "gmx.com", "inbox.com",
    "live.com", "msn.com", "rediffmail.com", "mailinator.com",
    "guerrillamail.com", "tempmail.com", "throwam.com",
}

SUSPICIOUS_DOMAIN_PATTERNS = [
    r"\d{4,}",           # lots of numbers: jobs12345.com
    r"^hr\d+\.",         # hr123.com
    r"recruit\d+\.",     # recruit99.com
    r"\.xyz$",           # suspicious TLDs
    r"\.top$",
    r"\.click$",
    r"\.tk$",
]


def analyze_email_domain(email: str) -> dict:
    """
    Rule-based recruiter email domain analysis.
    Returns a dict with verdict, reason, and severity.
    """
    if not email or "@" not in email:
        return {
            "provided":  False,
            "domain":    None,
            "verdict":   "not_provided",
            "label":     "Not Provided",
            "reason":    "No recruiter email was given.",
            "severity":  "neutral",
        }

    email  = email.strip().lower()
    domain = email.split("@")[-1]

    # Check free providers
    if domain in FREE_EMAIL_PROVIDERS:
        return {
            "provided":  True,
            "domain":    domain,
            "verdict":   "suspicious",
            "label":     "⚠ Suspicious",
            "reason":    f"Recruiter uses a free email provider ({domain}). "
                         "Legitimate companies typically use corporate email addresses.",
            "severity":  "high",
        }

    # Check suspicious patterns
    for pattern in SUSPICIOUS_DOMAIN_PATTERNS:
        if re.search(pattern, domain):
            return {
                "provided":  True,
                "domain":    domain,
                "verdict":   "suspicious",
                "label":     "⚠ Suspicious",
                "reason":    f"Domain '{domain}' matches known suspicious patterns.",
                "severity":  "high",
            }

    # Very short domain (less than 4 chars before TLD)
    domain_name = domain.split(".")[0]
    if len(domain_name) < 4:
        return {
            "provided":  True,
            "domain":    domain,
            "verdict":   "uncertain",
            "label":     "? Uncertain",
            "reason":    f"Domain '{domain}' is unusually short. Could not verify.",
            "severity":  "medium",
        }

    # Looks legitimate
    return {
        "provided":  True,
        "domain":    domain,
        "verdict":   "legitimate",
        "label":     "✓ Looks Legitimate",
        "reason":    f"'{domain}' appears to be a corporate domain. "
                     "Always verify independently.",
        "severity":  "low",
    }


# ─── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Risk Tier ─────────────────────────────────────────────────────────────────
def get_risk_tier(score: float) -> dict:
    if score >= 0.70:
        return {
            "tier":    "HIGH",
            "label":   "High Risk — Likely Scam",
            "color":   "red",
            "icon":    "🚨",
            "advice":  "This posting shows strong indicators of fraud. "
                       "Do NOT share personal information or pay any fees.",
        }
    elif score >= 0.35:
        return {
            "tier":    "MEDIUM",
            "label":   "Medium Risk — Proceed with Caution",
            "color":   "amber",
            "icon":    "⚠️",
            "advice":  "Some suspicious signals detected. Research the company "
                       "independently before applying or sharing any personal details.",
        }
    else:
        return {
            "tier":    "LOW",
            "label":   "Low Risk — Appears Genuine",
            "color":   "green",
            "icon":    "✅",
            "advice":  "No strong fraud signals detected. Still apply standard "
                       "caution — always verify company identity before interviews.",
        }


# ─── SHAP-style Top Signals ────────────────────────────────────────────────────

def get_top_signals(text: str, top_n: int = 8) -> dict:
    if explainer is None or vectorizer is None:
        return {"fraud": [], "genuine": []}

    X = vectorizer.transform([text])
    shap_values = explainer.shap_values(X)

    # shap_values shape: (1, n_features)
    values = shap_values[0]
    tokens = vectorizer.get_feature_names_out()

    # Only look at features present in this document
    cx = X.tocsr()
    nz_indices = cx.indices

    scored = [(tokens[i], float(values[i])) for i in nz_indices]

    fraud_signals = sorted(
        [(w, s) for w, s in scored if s > 0],
        key=lambda x: x[1], reverse=True
    )[:top_n]

    genuine_signals = sorted(
        [(w, s) for w, s in scored if s < 0],
        key=lambda x: x[1]
    )[:3]

    return {
        "fraud":   [{"word": w, "score": round(s, 4)} for w, s in fraud_signals],
        "genuine": [{"word": w, "score": round(abs(s), 4)} for w, s in genuine_signals],
    }


# ─── Demo Mode (no model loaded) ───────────────────────────────────────────────
DEMO_SCAM_KEYWORDS = [
    "work from home", "no experience", "unlimited earning",
    "easy money", "guaranteed income", "wire transfer",
    "western union", "click here", "make money fast",
    "no interview", "immediate start"
]

def demo_predict(text: str) -> float:
    """Heuristic score when model isn't loaded yet (for testing UI)."""
    text_lower = text.lower()
    hits = sum(1 for kw in DEMO_SCAM_KEYWORDS if kw in text_lower)
    return min(0.15 + hits * 0.12, 0.97)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    job_text = request.form.get("job_text", "").strip()
    email    = request.form.get("recruiter_email", "").strip()

    # Validation
    errors = []
    if not job_text:
        errors.append("Please paste a job description.")
    elif len(job_text) < 50:
        errors.append("Job description is too short (minimum 50 characters).")
    elif len(job_text) > 15000:
        errors.append("Job description is too long (max 15,000 characters).")

    if errors:
        return render_template("index.html", errors=errors,
                               job_text=job_text, recruiter_email=email)

    # Preprocess
    cleaned = clean_text(job_text)
    demo_mode = (model is None or vectorizer is None)

    # Predict
    if demo_mode:
        fraud_score = demo_predict(cleaned)
        signals     = {"fraud": [], "genuine": []}
        logger.warning("Running in DEMO mode — model not loaded.")
    else:
        X           = vectorizer.transform([cleaned])
        fraud_score = float(model.predict_proba(X)[0][1])
        signals     = get_top_signals(cleaned)

    # Risk tier
    risk    = get_risk_tier(fraud_score)
    domain  = analyze_email_domain(email)

    # Compose result
    result = {
        "fraud_score":     round(fraud_score * 100, 1),
        "fraud_score_raw": fraud_score,
        "risk":            risk,
        "domain":          domain,
        "signals":         signals,
        "char_count":      len(job_text),
        "word_count":      len(job_text.split()),
        "demo_mode":       demo_mode,
        "timestamp":       datetime.now().strftime("%B %d, %Y at %H:%M"),
    }

    logger.info(
        f"Prediction: score={fraud_score:.3f} tier={risk['tier']} "
        f"domain_verdict={domain['verdict']} demo={demo_mode}"
    )

    return render_template("result.html", result=result,
                           job_text=job_text, recruiter_email=email)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    return jsonify({
        "status":          "ok",
        "model_loaded":    model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp":       datetime.now().isoformat(),
    })


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    logger.info(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
