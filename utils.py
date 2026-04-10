# utils.py
import re
import logging

logger = logging.getLogger(__name__)

# Try to load a general English model for detecting human names
_anon_nlp = None
try:
    import spacy
    # en_core_web_sm is better at finding regular names than the medical model
    _anon_nlp = spacy.load("en_core_web_sm")
except Exception:
    pass

def clean_text(text: str) -> str:
    """Removes excessive whitespace and weird PDF artifact characters."""
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()

def deduplicate_list(items: list) -> list:
    """Removes duplicates from a list while preserving the original order."""
    seen = set()
    result = []
    for item in items:
        identifier = str(item) if isinstance(item, dict) else item
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result

def normalize_entity(entity: str) -> str:
    """Capitalizes the first letter of each word cleanly."""
    return entity.strip().title()

def anonymize_text(text: str) -> str:
    """
    Masks PII (Personally Identifiable Information) like names, phone numbers, 
    emails, and IDs to ensure GDPR compliance.
    """
    # 1. Regex Masking (Fast & Deterministic)
    # Phone numbers (e.g., 555-123-4567)
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', ' [REDACTED PHONE] ', text)
    
    # Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' [REDACTED EMAIL] ', text)
    
    # SSN / Medical ID (e.g., XXX-XX-XXXX)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', ' [REDACTED ID] ', text)
    
    # Dates of Birth (DOB: XX/XX/XXXX)
    text = re.sub(r'(?i)(dob|date of birth)[\s:]*[\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4}\b', r'\1: [REDACTED DOB]', text)

    # 2. NLP Masking for Human Names
    if _anon_nlp:
        try:
            doc = _anon_nlp(text[:100000]) # Cap at 100k chars for memory safety
            spans_to_replace = []
            
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "PER"]:
                    spans_to_replace.append((ent.start_char, ent.end_char))
            
            # Replace from back to front so index positions don't shift
            spans_to_replace.sort(key=lambda x: x[0], reverse=True)
            
            for start, end in spans_to_replace:
                text = text[:start] + "[PATIENT NAME]" + text[end:]
        except Exception as e:
            logger.debug(f"Anonymization NLP error: {e}")

    return text