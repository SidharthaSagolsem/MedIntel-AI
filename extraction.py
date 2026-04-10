# extraction.py
# Medical Named Entity Recognition (NER) using scispacy.
# Falls back to a regex-based extractor if scispacy models aren't installed.
#
# Setup:
#   pip install scispacy
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
#
# If the above URL fails, use:
#   python -m spacy download en_core_web_sm
# and the code will automatically use the fallback model.

import re
import logging
from typing import Optional
from utils import clean_text, deduplicate_list, normalize_entity

logger = logging.getLogger(__name__)

# ── Model Loading ─────────────────────────────────────────────────────────────

_nlp_model = None  # Cached model to avoid reloading on each call

def _load_nlp():
    """
    Lazy-load the NLP model. Tries scispacy first, then spacy en_core_web_sm.
    Returns the loaded model or None if nothing is available.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    
    # Try scispacy medical model first
    try:
        import spacy
        _nlp_model = spacy.load("en_core_sci_sm")
        logger.info("Loaded scispacy en_core_sci_sm model")
        return _nlp_model
    except Exception:
        pass
    
    # Fallback to standard spacy model
    try:
        import spacy
        _nlp_model = spacy.load("en_core_web_sm")
        logger.info("Loaded spacy en_core_web_sm model (fallback)")
        return _nlp_model
    except Exception:
        pass
    
    logger.warning("No spacy model found — using regex-only extraction")
    return None


# ── Keyword Lists ─────────────────────────────────────────────────────────────

# Common medical conditions for regex-based detection
DISEASE_KEYWORDS = [
    "diabetes", "hypertension", "asthma", "copd", "pneumonia", "cancer",
    "tuberculosis", "tb", "anemia", "arthritis", "osteoporosis", "hypothyroidism",
    "hyperthyroidism", "depression", "anxiety", "schizophrenia", "heart disease",
    "coronary artery disease", "heart failure", "atrial fibrillation", "stroke",
    "ckd", "chronic kidney disease", "liver disease", "hepatitis", "hiv", "aids",
    "sepsis", "urinary tract infection", "uti", "appendicitis", "gastritis",
    "ibs", "crohn", "ulcerative colitis", "gerd", "obesity", "malnutrition",
    "covid", "influenza", "hypertension", "dyslipidemia", "hyperlipidemia",
    "hypercholesterolemia", "hyperglycemia", "neuropathy", "retinopathy",
    "nephropathy", "pancreatitis", "cholecystitis", "pulmonary embolism",
    "deep vein thrombosis", "dvt", "myocardial infarction", "angina",
]

# Common medications
MEDICATION_KEYWORDS = [
    "metformin", "insulin", "glipizide", "glibenclamide", "sitagliptin",
    "atorvastatin", "rosuvastatin", "simvastatin", "aspirin", "clopidogrel",
    "warfarin", "heparin", "lisinopril", "enalapril", "amlodipine", "metoprolol",
    "atenolol", "bisoprolol", "furosemide", "hydrochlorothiazide", "spironolactone",
    "omeprazole", "pantoprazole", "ranitidine", "amoxicillin", "azithromycin",
    "ciprofloxacin", "doxycycline", "metronidazole", "fluconazole", "paracetamol",
    "acetaminophen", "ibuprofen", "naproxen", "diclofenac", "morphine", "tramadol",
    "codeine", "prednisone", "prednisolone", "dexamethasone", "salbutamol",
    "budesonide", "fluticasone", "levothyroxine", "propylthiouracil", "carbimazole",
    "sertraline", "fluoxetine", "escitalopram", "amitriptyline", "diazepam",
    "lorazepam", "alprazolam", "haloperidol", "risperidone", "olanzapine",
    "amlodipine", "nifedipine", "verapamil", "digoxin", "amiodarone",
]

# Lab test patterns — match "value unit" pairs like "HbA1c 8.5%", "BP 140/90"
LAB_PATTERNS = [
    r"(?:hba1c|hemoglobin\s*a1c)\s*[:\-]?\s*([\d.]+\s*%?)",
    r"(?:blood\s*glucose|fasting\s*glucose|rbs|fbs|ppbs)\s*[:\-]?\s*([\d.]+\s*(?:mg/dl|mmol/l)?)",
    r"(?:bp|blood\s*pressure)\s*[:\-]?\s*([\d]+\s*/\s*[\d]+\s*(?:mmhg)?)",
    r"(?:creatinine)\s*[:\-]?\s*([\d.]+\s*(?:mg/dl|μmol/l)?)",
    r"(?:hemoglobin|hb|haemoglobin)\s*[:\-]?\s*([\d.]+\s*(?:g/dl)?)",
    r"(?:cholesterol|ldl|hdl|triglycerides?)\s*[:\-]?\s*([\d.]+\s*(?:mg/dl)?)",
    r"(?:tsh|t3|t4)\s*[:\-]?\s*([\d.]+\s*(?:miu/l|ng/dl|pmol/l)?)",
    r"(?:wbc|rbc|platelet|plt)\s*(?:count)?\s*[:\-]?\s*([\d.]+\s*(?:×10|x10|k/ul|/ul)?[\d]*)",
    r"(?:sodium|potassium|chloride|bicarbonate)\s*[:\-]?\s*([\d.]+\s*(?:meq/l|mmol/l)?)",
    r"(?:alt|ast|alp|bilirubin)\s*[:\-]?\s*([\d.]+\s*(?:u/l|mg/dl)?)",
    r"(?:urea|bun)\s*[:\-]?\s*([\d.]+\s*(?:mg/dl)?)",
    r"(?:albumin)\s*[:\-]?\s*([\d.]+\s*(?:g/dl)?)",
    r"(?:o2\s*sat|spo2|oxygen\s*saturation)\s*[:\-]?\s*([\d.]+\s*%?)",
    r"(?:pulse|heart\s*rate|hr)\s*[:\-]?\s*([\d]+\s*(?:bpm)?)",
    r"(?:temperature|temp)\s*[:\-]?\s*([\d.]+\s*(?:°[cf]|f|c)?)",
    r"(?:weight)\s*[:\-]?\s*([\d.]+\s*(?:kg|lbs)?)",
    r"(?:height)\s*[:\-]?\s*([\d.]+\s*(?:cm|ft|m)?)",
    r"(?:bmi)\s*[:\-]?\s*([\d.]+)",
    r"(?:egfr|gfr)\s*[:\-]?\s*([\d.]+\s*(?:ml/min)?)",
]

# Labels matching each LAB_PATTERNS entry above
LAB_LABELS = [
    "HbA1c", "Blood Glucose", "Blood Pressure", "Creatinine", "Hemoglobin",
    "Cholesterol/Lipids", "Thyroid (TSH/T3/T4)", "CBC (WBC/RBC/PLT)", "Electrolytes",
    "Liver Enzymes", "Urea/BUN", "Albumin", "O2 Saturation", "Heart Rate",
    "Temperature", "Weight", "Height", "BMI", "eGFR",
]


# ── Main Extraction Function ──────────────────────────────────────────────────

def extract_medical_entities(text: str) -> dict:
    """
    Extract medical entities from clinical text.
    
    Returns a dict:
    {
        "diseases": ["Diabetes", "Hypertension", ...],
        "medications": ["Metformin 500mg", ...],
        "lab_values": [{"name": "HbA1c", "value": "8.5%"}, ...],
        "dates": ["January 15, 2022", ...]
    }
    """
    text_clean = clean_text(text)
    text_lower = text_clean.lower()
    
    diseases    = _extract_diseases(text_clean, text_lower)
    medications = _extract_medications(text_clean, text_lower)
    lab_values  = _extract_lab_values(text_clean)
    dates       = _extract_dates(text_clean)
    
    return {
        "diseases":    deduplicate_list(diseases),
        "medications": deduplicate_list(medications),
        "lab_values":  deduplicate_list(lab_values),
        "dates":       deduplicate_list(dates),
    }


# ── Sub-extractors ────────────────────────────────────────────────────────────

def _extract_diseases(text: str, text_lower: str) -> list[str]:
    """
    Combine NLP-based NER (if available) + keyword matching.
    """
    found = []
    
    # 1. Try NLP model
    nlp = _load_nlp()
    if nlp:
        try:
            doc = nlp(text[:100000])  # Limit to 100k chars to avoid memory issues
            for ent in doc.ents:
                # scispacy labels diseases as DISEASE; spacy uses various labels
                if ent.label_ in ("DISEASE", "ORG", "GPE", "PERSON") and ent.label_ == "DISEASE":
                    found.append(ent.text.title())
        except Exception as e:
            logger.debug(f"NLP entity extraction error: {e}")
    
    # 2. Keyword-based fallback / supplement
    for keyword in DISEASE_KEYWORDS:
        # Use word boundary matching
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            # Try to grab surrounding context for a richer label
            match = re.search(
                r'(?:diagnosed with|history of|known case of|presents with|complains of)?\s*' +
                r'(' + re.escape(keyword) + r'(?:\s+(?:type\s*[12]|mellitus|stage\s*\w+))?)',
                text_lower
            )
            label = match.group(1).title() if match else keyword.title()
            found.append(label)
    
    return found


def _extract_medications(text: str, text_lower: str) -> list[str]:
    """
    Extract medication names with optional dosage from the text.
    """
    found = []
    
    for med in MEDICATION_KEYWORDS:
        pattern = r'\b' + re.escape(med) + r'(?:\s+[\d.]+\s*(?:mg|mcg|ml|g|iu|units?))?\b'
        match = re.search(pattern, text_lower)
        if match:
            # Find original-case version
            start = match.start()
            end = match.end()
            original = text[start:end]
            found.append(original.strip().title())
    
    # Also look for generic "prescribed X" / "started on X" patterns
    rx_patterns = [
        r'(?:prescribed|started on|taking|on)\s+([A-Z][a-z]+(?:\s+[\d.]+\s*mg)?)',
        r'(?:tab|tablet|cap|capsule|inj|injection)\s+([A-Z][a-z]+)',
    ]
    for pat in rx_patterns:
        for m in re.finditer(pat, text):
            candidate = m.group(1).strip()
            if 2 < len(candidate) < 40:
                found.append(candidate)
    
    return found


def _extract_lab_values(text: str) -> list[dict]:
    """
    Use regex patterns to find lab values with their numeric readings.
    Returns list of {"name": "...", "value": "..."} dicts.
    """
    found = []
    text_lower = text.lower()
    
    for label, pattern in zip(LAB_LABELS, LAB_PATTERNS):
        match = re.search(pattern, text_lower)
        if match:
            value = match.group(1).strip()
            found.append({"name": label, "value": value})
    
    return found


def _extract_dates(text: str) -> list[str]:
    """
    Extract and normalize date mentions using dateparser.
    Falls back to regex if dateparser is unavailable.
    """
    found = []
    
    # Comprehensive date patterns
    date_patterns = [
        # "January 15, 2022" / "15 January 2022"
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
        # "15/01/2022" or "01-15-2022"
        r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
        # "2022-01-15" (ISO)
        r'\b\d{4}-\d{2}-\d{2}\b',
        # "2021" standalone year
        r'\b(20[0-2]\d|19[5-9]\d)\b',
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            date_str = match.group(0).strip()
            # Normalize with dateparser
            try:
                import dateparser
                parsed = dateparser.parse(date_str)
                if parsed:
                    normalized = parsed.strftime("%B %d, %Y")
                    found.append(normalized)
                else:
                    found.append(date_str)
            except Exception:
                found.append(date_str)
    
    return found
