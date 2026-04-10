# timeline.py
# Builds a chronological patient timeline by mapping extracted medical
# entities to the nearest date found in the surrounding text.

import re
import logging
from utils import clean_text, deduplicate_list

logger = logging.getLogger(__name__)


# Keywords that suggest a clinical event is tied to a date
EVENT_TRIGGERS = {
    "diagnosis": [
        "diagnosed with", "diagnosis of", "known case of", "found to have",
        "presented with", "admitted with", "confirmed", "newly diagnosed",
    ],
    "medication_start": [
        "started on", "prescribed", "initiated", "commenced", "began",
        "switched to", "added", "changed to",
    ],
    "medication_stop": [
        "stopped", "discontinued", "withdrawn", "ceased", "held",
    ],
    "procedure": [
        "underwent", "procedure", "surgery", "operation", "biopsy",
        "angioplasty", "stenting", "catheterization", "transplant",
    ],
    "hospitalization": [
        "admitted", "hospitalized", "discharge", "discharged", "icu",
        "emergency", "inpatient",
    ],
    "lab_event": [
        "hba1c", "blood glucose", "hemoglobin", "creatinine", "cholesterol",
        "blood pressure", "bp:", "sugar level",
    ],
    "follow_up": [
        "follow up", "follow-up", "review", "reviewed", "returned",
        "outpatient", "clinic visit",
    ],
}


def build_timeline(text: str, entities: dict) -> list[dict]:
    """
    Build a chronological timeline from document text and extracted entities.
    
    Strategy:
      1. Split text into sentences
      2. For each sentence, check if it contains a date and a medical event
      3. Map event → nearest date (within ±3 sentences)
      4. Sort everything chronologically
    
    Returns:
        List of {"date": str, "event": str, "category": str} dicts
    """
    text_clean = clean_text(text)
    sentences = _split_into_sentences(text_clean)
    
    events = []
    
    # Pass 1: Extract sentence-level events with nearby dates
    for i, sentence in enumerate(sentences):
        date = _find_date_in_window(sentences, i, window=3)
        if not date:
            continue
        
        category, event_text = _classify_event(sentence)
        if event_text:
            events.append({
                "date": date,
                "event": event_text,
                "category": category,
                "_sort_key": _date_sort_key(date),
            })
    
    # Pass 2: Add entity-based events that might have been missed
    events += _events_from_entities(entities, text_clean)
    
    # Sort chronologically, deduplicate, and clean up
    events.sort(key=lambda e: e.get("_sort_key", "9999"))
    
    # Remove internal sort key before returning
    for e in events:
        e.pop("_sort_key", None)
    
    return deduplicate_list(events)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    # Split on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Also split on newlines (clinical notes often use line-per-event)
    result = []
    for s in sentences:
        for line in s.split('\n'):
            line = line.strip()
            if len(line) > 10:  # Skip very short fragments
                result.append(line)
    return result


def _find_date_in_window(sentences: list[str], center: int, window: int) -> str:
    """
    Look for a date string in sentences[center-window : center+window].
    Returns the first date found, or empty string.
    """
    start = max(0, center - window)
    end   = min(len(sentences), center + window + 1)
    
    for i in range(start, end):
        date = _extract_date_from_text(sentences[i])
        if date:
            return date
    return ""


def _extract_date_from_text(text: str) -> str:
    """
    Extract a single date from a text fragment using dateparser.
    Returns normalized date string or empty string.
    """
    date_patterns = [
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
        r'Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b(20[0-2]\d|19[5-9]\d)\b',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(0)
            try:
                import dateparser
                parsed = dateparser.parse(raw)
                if parsed:
                    return parsed.strftime("%B %Y")
            except Exception:
                pass
            return raw
    return ""


def _classify_event(sentence: str) -> tuple[str, str]:
    """
    Classify a sentence into an event category and generate a summary label.
    Returns ("category", "event text") or ("", "") if not an event sentence.
    """
    sentence_lower = sentence.lower()
    
    for category, triggers in EVENT_TRIGGERS.items():
        for trigger in triggers:
            if trigger in sentence_lower:
                # Build a clean event summary (truncated to 80 chars)
                summary = _summarize_event(sentence, trigger, category)
                if summary:
                    return category, summary
    
    return "", ""


def _summarize_event(sentence: str, trigger: str, category: str) -> str:
    """
    Generate a human-readable event summary from a sentence.
    """
    # Capitalize and truncate
    sentence = sentence.strip()
    
    # Try to build a specific summary
    if category == "diagnosis":
        # "Diagnosed with Type 2 Diabetes"
        match = re.search(
            r'(?:diagnosed with|diagnosis of|known case of|found to have|presented with)\s+([^.;,]{3,60})',
            sentence, re.IGNORECASE
        )
        if match:
            return "Diagnosis: " + match.group(1).strip().title()
    
    elif category == "medication_start":
        match = re.search(
            r'(?:started on|prescribed|initiated|commenced|began|switched to|added)\s+([^.;,]{3,50})',
            sentence, re.IGNORECASE
        )
        if match:
            return "Medication: Started " + match.group(1).strip().title()
    
    elif category == "medication_stop":
        match = re.search(
            r'(?:stopped|discontinued|withdrawn|ceased|held)\s+([^.;,]{3,50})',
            sentence, re.IGNORECASE
        )
        if match:
            return "Medication: Stopped " + match.group(1).strip().title()
    
    elif category == "hospitalization":
        match = re.search(
            r'(?:admitted|hospitalized)\s+(?:for|with)?\s*([^.;,]{3,60})',
            sentence, re.IGNORECASE
        )
        if match:
            return "Hospitalization: " + match.group(1).strip().title()
        return "Hospitalization event"
    
    elif category == "procedure":
        match = re.search(
            r'(?:underwent|procedure|surgery)\s+(?:for|of)?\s*([^.;,]{3,60})',
            sentence, re.IGNORECASE
        )
        if match:
            return "Procedure: " + match.group(1).strip().title()
    
    elif category == "lab_event":
        # Return truncated sentence as-is
        return sentence[:80] + ("..." if len(sentence) > 80 else "")
    
    # Generic fallback: first 80 chars
    return sentence[:80] + ("..." if len(sentence) > 80 else "")


def _events_from_entities(entities: dict, text: str) -> list[dict]:
    """
    Create rough timeline entries from extracted entities even without
    nearby date context (assign to the first date found in document).
    """
    events = []
    
    # Find the most common/first date in the document as a default anchor
    default_date = _extract_date_from_text(text[:2000]) or "Date Unknown"
    
    for disease in entities.get("diseases", [])[:5]:  # Limit to 5
        events.append({
            "date": default_date,
            "event": f"Condition noted: {disease}",
            "category": "diagnosis",
            "_sort_key": _date_sort_key(default_date) + "_z",  # Sort last within same date
        })
    
    return events


def _date_sort_key(date_str: str) -> str:
    """
    Convert a date string to a zero-padded sort key (YYYY-MM).
    Returns "9999-99" if parsing fails (sorts to end).
    """
    try:
        import dateparser
        parsed = dateparser.parse(date_str)
        if parsed:
            return parsed.strftime("%Y-%m")
    except Exception:
        pass
    
    # Try to extract 4-digit year
    match = re.search(r'\b(20[0-2]\d|19[5-9]\d)\b', date_str)
    if match:
        return match.group(1) + "-00"
    
    return "9999-99"
