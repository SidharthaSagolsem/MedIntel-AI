# risk.py
# Rule-based clinical risk detection.
# Analyses extracted entities and lab values to flag health risks.
# All rules are transparent and explainable — no black-box models needed.

import re
import logging
from utils import normalize_entity

logger = logging.getLogger(__name__)


# Risk level constants
HIGH   = "high"
MEDIUM = "medium"
LOW    = "low"
INFO   = "info"


def detect_risks(entities: dict, text: str = "") -> list[dict]:
    """
    Run all risk rules against extracted entities and return a list of alerts.
    
    Each alert:
    {
        "level":   "high" | "medium" | "low" | "info",
        "title":   "Short title",
        "message": "Explanation of the risk",
        "icon":    "🔴 / 🟡 / 🟢 / ℹ️"
    }
    
    Args:
        entities: Output from extraction.extract_medical_entities()
        text:     Raw document text (for additional pattern matching)
    
    Returns:
        List of risk alert dicts, sorted by severity (high → medium → low → info)
    """
    alerts = []
    
    diseases    = [normalize_entity(d) for d in entities.get("diseases", [])]
    medications = [normalize_entity(m) for m in entities.get("medications", [])]
    lab_values  = {lv["name"].lower(): lv["value"] for lv in entities.get("lab_values", [])}
    text_lower  = text.lower()
    
    # Run each rule group
    alerts += _check_diabetes_risk(diseases, lab_values, medications)
    alerts += _check_hypertension_risk(diseases, lab_values, text_lower)
    alerts += _check_cardiac_risk(diseases, medications, lab_values)
    alerts += _check_renal_risk(diseases, lab_values)
    alerts += _check_anemia_risk(diseases, lab_values)
    alerts += _check_infection_risk(diseases, text_lower)
    alerts += _check_respiratory_risk(diseases, medications)
    alerts += _check_liver_risk(diseases, lab_values)
    alerts += _check_thyroid_risk(diseases, lab_values, medications)
    alerts += _check_polypharmacy_risk(medications)
    alerts += _check_mental_health_risk(diseases, medications)
    
    # If no risks found, add a positive note
    if not alerts:
        alerts.append({
            "level":       INFO,
            "title":       "No Major Risks Detected",
            "message":     "Based on the extracted entities, no high-risk conditions were identified. "
                           "This assessment is based on available document data only.",
            "explanation": "All checked clinical rules passed. No lab thresholds were breached and "
                           "no high-risk diagnosis patterns were matched in the document.",
            "icon":        "✅",
        })

    # Sort: high → medium → low → info
    severity_order = {HIGH: 0, MEDIUM: 1, LOW: 2, INFO: 3}
    alerts.sort(key=lambda a: severity_order.get(a["level"], 4))

    return alerts


# ── Individual Rule Groups ────────────────────────────────────────────────────

def _check_diabetes_risk(diseases, lab_values, medications) -> list[dict]:
    alerts = []
    has_diabetes = any("diabet" in d for d in diseases)

    # Check HbA1c
    hba1c_raw = lab_values.get("hba1c", "")
    hba1c_val = _parse_numeric(hba1c_raw)

    if hba1c_val is not None:
        if hba1c_val >= 9.0:
            alerts.append({
                "level":       HIGH,
                "title":       "Poor Glycaemic Control",
                "message":     f"HbA1c of {hba1c_raw} indicates poor diabetes management. "
                               "Target is typically <7% for most adults. Urgent review recommended.",
                "explanation": f"HbA1c value ({hba1c_raw}) meets or exceeds the critical threshold of ≥9.0%. "
                               "Values this high are associated with significantly elevated risk of "
                               "microvascular complications (retinopathy, nephropathy, neuropathy). "
                               "Clinical target is <7% (ADA/NICE guidelines).",
                "icon":        "🔴",
            })
        elif hba1c_val >= 7.0:
            alerts.append({
                "level":       MEDIUM,
                "title":       "Suboptimal Glycaemic Control",
                "message":     f"HbA1c of {hba1c_raw} is above the recommended target of <7%. "
                               "Medication adjustment and lifestyle modification may be needed.",
                "explanation": f"HbA1c value ({hba1c_raw}) is in the range 7.0–8.9% (above target, below critical). "
                               "This indicates suboptimal glycaemic control. Guideline target is <7% for most "
                               "adults; values ≥7% increase the risk of long-term diabetes complications.",
                "icon":        "🟡",
            })

    # Check fasting glucose
    glucose_raw = lab_values.get("blood glucose", "")
    glucose_val = _parse_numeric(glucose_raw)

    if glucose_val is not None:
        if glucose_val >= 200:
            alerts.append({
                "level":       HIGH,
                "title":       "Hyperglycaemia Detected",
                "message":     f"Blood glucose of {glucose_raw} mg/dL is critically elevated. "
                               "Risk of diabetic ketoacidosis or hyperosmolar state.",
                "explanation": f"Blood glucose ({glucose_raw} mg/dL) exceeds the critical threshold of ≥200 mg/dL. "
                               "WHO diagnostic criterion for diabetes is ≥200 mg/dL with symptoms. "
                               "Values this high risk DKA and HONK/HHS — medical emergency.",
                "icon":        "🔴",
            })
        elif glucose_val >= 126:
            alerts.append({
                "level":       MEDIUM,
                "title":       "Elevated Fasting Glucose",
                "message":     f"Fasting glucose of {glucose_raw} mg/dL meets diagnostic criteria for diabetes "
                               "or indicates uncontrolled existing diabetes.",
                "explanation": f"Fasting blood glucose ({glucose_raw} mg/dL) is ≥126 mg/dL. "
                               "ADA/WHO diagnostic threshold for diabetes mellitus is ≥126 mg/dL on fasting sample. "
                               "Normal fasting glucose is <100 mg/dL.",
                "icon":        "🟡",
            })
        elif glucose_val >= 100:
            alerts.append({
                "level":       LOW,
                "title":       "Prediabetes Range Glucose",
                "message":     f"Fasting glucose of {glucose_raw} mg/dL is in the prediabetes range (100–125 mg/dL). "
                               "Lifestyle modification recommended.",
                "explanation": f"Fasting glucose ({glucose_raw} mg/dL) falls in the ADA prediabetes range of "
                               "100–125 mg/dL (impaired fasting glucose). Without intervention, approximately "
                               "15–30% of patients progress to type 2 diabetes within 5 years.",
                "icon":        "🟢",
            })

    # Diabetes present but no glucose-lowering medication found
    if has_diabetes and not any(m in medications for m in [
        "metformin", "insulin", "glipizide", "sitagliptin", "glibenclamide"
    ]):
        alerts.append({
            "level":       MEDIUM,
            "title":       "Diabetes Without Documented Treatment",
            "message":     "Diabetes is noted but no glucose-lowering medication was detected in the document. "
                           "Verify if treatment details are missing from the record.",
            "explanation": "Rule triggered: diabetes keyword found in diagnoses, but none of the expected "
                           "antidiabetic agents (metformin, insulin, glipizide, sitagliptin, glibenclamide) "
                           "were matched in the medication list. This may indicate a documentation gap.",
            "icon":        "🟡",
        })

    return alerts


def _check_hypertension_risk(diseases, lab_values, text_lower) -> list[dict]:
    alerts = []
    has_htn = any("hypertens" in d for d in diseases)

    bp_raw = lab_values.get("blood pressure", "")
    systolic, diastolic = _parse_bp(bp_raw)

    if systolic is not None:
        if systolic >= 180 or (diastolic is not None and diastolic >= 120):
            alerts.append({
                "level":       HIGH,
                "title":       "Hypertensive Crisis",
                "message":     f"Blood pressure of {bp_raw} mmHg indicates a hypertensive crisis. "
                               "Immediate medical attention required.",
                "explanation": f"Recorded BP ({bp_raw} mmHg) meets the hypertensive crisis criterion: "
                               "systolic ≥180 mmHg or diastolic ≥120 mmHg (JNC 8 / ESC guidelines). "
                               "Risk of hypertensive encephalopathy, stroke, or acute MI is significantly elevated.",
                "icon":        "🔴",
            })
        elif systolic >= 140 or (diastolic is not None and diastolic >= 90):
            alerts.append({
                "level":       MEDIUM,
                "title":       "Stage 2 Hypertension",
                "message":     f"Blood pressure of {bp_raw} mmHg indicates stage 2 hypertension. "
                               "Medication review and lifestyle modification advised.",
                "explanation": f"Recorded BP ({bp_raw} mmHg) falls in the Stage 2 hypertension range: "
                               "systolic 140–179 mmHg or diastolic 90–119 mmHg (JNC 8 classification). "
                               "Pharmacotherapy is usually indicated at this level.",
                "icon":        "🟡",
            })
        elif systolic >= 130 or (diastolic is not None and diastolic >= 80):
            alerts.append({
                "level":       LOW,
                "title":       "Stage 1 Hypertension",
                "message":     f"Blood pressure of {bp_raw} mmHg is in the stage 1 hypertension range. "
                               "Lifestyle changes and monitoring recommended.",
                "explanation": f"Recorded BP ({bp_raw} mmHg) meets the ACC/AHA 2017 Stage 1 hypertension "
                               "definition: systolic 130–139 mmHg or diastolic 80–89 mmHg. "
                               "Lifestyle interventions are first-line at this stage.",
                "icon":        "🟢",
            })
    elif has_htn:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Hypertension Noted",
            "message":     "Hypertension is documented. Blood pressure values were not clearly extracted — "
                           "please review the full record.",
            "explanation": "The word 'hypertension' was matched in the diagnoses list but no parseable "
                           "blood pressure reading (format: systolic/diastolic) was found in the document. "
                           "Manual review of the source PDF is needed.",
            "icon":        "🟡",
        })

    return alerts


def _check_cardiac_risk(diseases, medications, lab_values) -> list[dict]:
    alerts = []
    cardiac_conditions = ["coronary artery disease", "heart failure", "myocardial infarction",
                           "angina", "atrial fibrillation", "heart disease", "cad", "chf", "mi"]

    has_cardiac = any(any(cc in d for cc in cardiac_conditions) for d in diseases)

    cholesterol_raw = lab_values.get("cholesterol/lipids", "")
    chol_val = _parse_numeric(cholesterol_raw)

    if has_cardiac:
        alerts.append({
            "level":       HIGH,
            "title":       "Cardiovascular Disease Present",
            "message":     "Patient has documented cardiovascular condition. "
                           "Ensure antiplatelet, statin, and ACE inhibitor/beta-blocker therapy is reviewed.",
            "explanation": "One or more of the following high-risk cardiac diagnoses was matched: "
                           "coronary artery disease, heart failure, myocardial infarction, angina, "
                           "or atrial fibrillation. These conditions require active secondary prevention "
                           "therapy per ACC/AHA/ESC guidelines.",
            "icon":        "🔴",
        })

    if chol_val is not None and chol_val >= 240:
        alerts.append({
            "level":       MEDIUM,
            "title":       "High Cholesterol",
            "message":     f"Total cholesterol/lipid value of {cholesterol_raw} mg/dL is elevated. "
                           "Increased cardiovascular risk. Statin therapy should be considered.",
            "explanation": f"Cholesterol/lipid value ({cholesterol_raw} mg/dL) exceeds 240 mg/dL, "
                           "classified as high by NCEP ATP III. Values ≥240 mg/dL are associated with "
                           "approximately double the cardiovascular risk vs values <200 mg/dL.",
            "icon":        "🟡",
        })

    # Cardiac patient without aspirin or statin
    if has_cardiac:
        if not any(m in medications for m in ["aspirin", "clopidogrel", "warfarin", "heparin"]):
            alerts.append({
                "level":       MEDIUM,
                "title":       "No Antiplatelet Therapy Documented",
                "message":     "Cardiovascular disease noted but no antiplatelet agent found in records. "
                               "Verify if therapy is documented elsewhere.",
                "explanation": "Rule triggered: a cardiac diagnosis was matched but none of the expected "
                               "antiplatelet/anticoagulant agents (aspirin, clopidogrel, warfarin, heparin) "
                               "were found in the medication list. Could indicate a documentation gap or "
                               "contraindication to therapy.",
                "icon":        "🟡",
            })

    return alerts


def _check_renal_risk(diseases, lab_values) -> list[dict]:
    alerts = []
    has_ckd = any(("kidney" in d or "renal" in d or "ckd" in d or "nephrop" in d) for d in diseases)

    creatinine_raw = lab_values.get("creatinine", "")
    creat_val = _parse_numeric(creatinine_raw)

    egfr_raw = lab_values.get("egfr", "")
    egfr_val = _parse_numeric(egfr_raw)

    if creat_val is not None and creat_val >= 2.0:
        alerts.append({
            "level":       HIGH,
            "title":       "Severely Elevated Creatinine",
            "message":     f"Creatinine of {creatinine_raw} mg/dL suggests significant renal impairment. "
                           "Nephrology review and medication dose adjustment required.",
            "explanation": f"Serum creatinine ({creatinine_raw} mg/dL) is ≥2.0 mg/dL. "
                           "Normal range is approximately 0.6–1.2 mg/dL. Values ≥2.0 mg/dL suggest "
                           "significant GFR reduction (typically <30 ml/min/1.73m² at this creatinine level), "
                           "consistent with CKD Stage 4 or worse.",
            "icon":        "🔴",
        })
    elif creat_val is not None and creat_val >= 1.3:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Elevated Creatinine",
            "message":     f"Creatinine of {creatinine_raw} mg/dL is mildly elevated. "
                           "Monitor renal function and review nephrotoxic medications.",
            "explanation": f"Serum creatinine ({creatinine_raw} mg/dL) is in the range 1.3–1.9 mg/dL. "
                           "This is above the upper limit of normal (~1.2 mg/dL for males, ~1.0 mg/dL for females) "
                           "and may indicate early-to-moderate CKD. Nephrotoxic drugs (NSAIDs, aminoglycosides) "
                           "should be reviewed.",
            "icon":        "🟡",
        })

    if egfr_val is not None and egfr_val < 30:
        alerts.append({
            "level":       HIGH,
            "title":       "Severely Reduced eGFR",
            "message":     f"eGFR of {egfr_raw} ml/min indicates severe CKD (Stage 4+). "
                           "Specialist referral and close monitoring essential.",
            "explanation": f"eGFR ({egfr_raw} ml/min/1.73m²) is below 30, corresponding to CKD Stage 4 "
                           "(eGFR 15–29) or Stage 5 (<15). KDIGO guidelines recommend nephrology referral "
                           "and preparation for renal replacement therapy at eGFR <30.",
            "icon":        "🔴",
        })
    elif egfr_val is not None and egfr_val < 60:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Reduced eGFR",
            "message":     f"eGFR of {egfr_raw} ml/min suggests moderate CKD. "
                           "Medication dosing review and nephrology consultation advised.",
            "explanation": f"eGFR ({egfr_raw} ml/min/1.73m²) falls in the range 30–59, consistent with "
                           "CKD Stage 3a (45–59) or 3b (30–44) by KDIGO 2012 classification. "
                           "Drug dosing adjustments are commonly required for renally-cleared medications.",
            "icon":        "🟡",
        })

    return alerts


def _check_anemia_risk(diseases, lab_values) -> list[dict]:
    alerts = []
    hb_raw = lab_values.get("hemoglobin", "")
    hb_val = _parse_numeric(hb_raw)
    
    if hb_val is not None:
        if hb_val < 7.0:
            alerts.append({
                "level":       HIGH,
                "title":       "Severe Anæmia",
                "message":     f"Hemoglobin of {hb_raw} g/dL indicates severe anæmia. "
                               "Transfusion may be required. Urgent evaluation needed.",
                "explanation": f"Haemoglobin ({hb_raw} g/dL) is below the severe anæmia threshold of 7.0 g/dL "
                               "(WHO classification). At this level cardiac output compensation may be insufficient; "
                               "transfusion triggers are generally ≤7 g/dL for most adults (BSH/AABB guidelines).",
                "icon":        "🔴",
            })
        elif hb_val < 10.0:
            alerts.append({
                "level":       MEDIUM,
                "title":       "Moderate Anæmia",
                "message":     f"Hemoglobin of {hb_raw} g/dL indicates moderate anæmia. "
                               "Iron studies, B12, and folate should be checked.",
                "explanation": f"Haemoglobin ({hb_raw} g/dL) is in the 7.0–9.9 g/dL range (WHO moderate anæmia). "
                               "Common causes include iron deficiency, chronic disease, B12/folate deficiency, "
                               "or haemolysis. Full blood count and iron panel are recommended.",
                "icon":        "🟡",
            })
        elif hb_val < 12.0:
            alerts.append({
                "level":       LOW,
                "title":       "Mild Anæmia",
                "message":     f"Hemoglobin of {hb_raw} g/dL is mildly low. "
                               "Monitor and investigate the underlying cause.",
                "explanation": f"Haemoglobin ({hb_raw} g/dL) is below 12.0 g/dL (mild anæmia threshold by WHO "
                               "for females; ≤13.0 g/dL for males). Monitoring and investigation of "
                               "nutritional or chronic-disease causes is advised.",
                "icon":        "🟢",
            })
    
    return alerts


def _check_infection_risk(diseases, text_lower) -> list[dict]:
    alerts = []
    serious_infections = ["sepsis", "septicemia", "bacteremia", "meningitis", "endocarditis"]

    if any(inf in text_lower for inf in serious_infections):
        alerts.append({
            "level":       HIGH,
            "title":       "Serious Infection Documented",
            "message":     "Sepsis or serious systemic infection noted. Requires immediate medical attention "
                           "and appropriate antimicrobial therapy.",
            "explanation": "One or more of the following high-acuity infection keywords was matched in the "
                           "document text: sepsis, septicemia, bacteremia, meningitis, endocarditis. "
                           "These conditions carry significant mortality risk and require urgent treatment.",
            "icon":        "🔴",
        })

    # Active TB
    if any("tuberculosis" in d or " tb" in d for d in diseases):
        alerts.append({
            "level":       HIGH,
            "title":       "Tuberculosis Noted",
            "message":     "Tuberculosis is documented. Ensure appropriate anti-TB therapy and contact tracing.",
            "explanation": "'Tuberculosis' or 'TB' was matched in the extracted diagnoses. "
                           "TB is a notifiable disease with specific treatment (RHEZ regimen) and "
                           "infection control implications. Contact tracing is a public health requirement.",
            "icon":        "🔴",
        })

    return alerts


def _check_respiratory_risk(diseases, medications) -> list[dict]:
    alerts = []
    resp_conditions = ["asthma", "copd", "chronic obstructive", "respiratory failure", "pulmonary"]

    has_resp = any(any(rc in d for rc in resp_conditions) for d in diseases)

    if has_resp:
        if not any(m in medications for m in ["salbutamol", "budesonide", "fluticasone", "ipratropium"]):
            alerts.append({
                "level":       MEDIUM,
                "title":       "Respiratory Condition Without Inhaler",
                "message":     "Asthma/COPD is documented but no inhaler medication was found. "
                               "Verify treatment compliance and documentation.",
                "explanation": "An asthma or COPD diagnosis was matched but none of the expected "
                               "bronchodilator/inhaled corticosteroid agents (salbutamol, budesonide, "
                               "fluticasone, ipratropium) were found in the medication list. "
                               "GINA/GOLD guidelines require inhaler therapy for both conditions.",
                "icon":        "🟡",
            })

    return alerts


def _check_liver_risk(diseases, lab_values) -> list[dict]:
    alerts = []
    liver_conditions = ["hepatitis", "cirrhosis", "liver disease", "liver failure", "jaundice"]
    has_liver = any(any(lc in d for lc in liver_conditions) for d in diseases)

    alt_raw = lab_values.get("liver enzymes", "")
    alt_val = _parse_numeric(alt_raw)

    if alt_val is not None and alt_val >= 3 * 40:  # 3x upper limit of normal (~40 U/L)
        alerts.append({
            "level":       HIGH,
            "title":       "Significantly Elevated Liver Enzymes",
            "message":     f"Liver enzymes (ALT/AST) at {alt_raw} U/L are more than 3x the normal range. "
                           "Hepatotoxic medications should be reviewed.",
            "explanation": f"ALT/AST value ({alt_raw} U/L) exceeds 3× the upper limit of normal (ULN ~40 U/L). "
                           "Per DILI/RUCAM criteria, values >3× ULN trigger further investigation. "
                           "Common causes include drug-induced liver injury, viral hepatitis, or alcoholic liver disease.",
            "icon":        "🔴",
        })

    if has_liver:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Liver Disease Documented",
            "message":     "Hepatic condition noted. Avoid hepatotoxic drugs and monitor LFTs regularly.",
            "explanation": "One or more of the following hepatic diagnoses was matched: hepatitis, cirrhosis, "
                           "liver disease, liver failure, or jaundice. Hepatotoxic agents (statins, "
                           "paracetamol at high doses, NSAIDs) should be used with caution or avoided.",
            "icon":        "🟡",
        })

    return alerts


def _check_thyroid_risk(diseases, lab_values, medications) -> list[dict]:
    alerts = []
    tsh_raw = lab_values.get("thyroid (tsh/t3/t4)", "")
    tsh_val = _parse_numeric(tsh_raw)

    if tsh_val is not None:
        if tsh_val > 10:
            alerts.append({
                "level":       MEDIUM,
                "title":       "Elevated TSH — Hypothyroidism",
                "message":     f"TSH of {tsh_raw} mIU/L is elevated, suggesting hypothyroidism. "
                               "Levothyroxine therapy should be reviewed or initiated.",
                "explanation": f"TSH ({tsh_raw} mIU/L) exceeds 10 mIU/L. Normal TSH range is approximately "
                               "0.4–4.0 mIU/L. Values >10 mIU/L indicate overt hypothyroidism per ATA/ETA "
                               "guidelines, warranting levothyroxine replacement therapy.",
                "icon":        "🟡",
            })
        elif tsh_val < 0.4:
            alerts.append({
                "level":       MEDIUM,
                "title":       "Suppressed TSH — Hyperthyroidism",
                "message":     f"TSH of {tsh_raw} mIU/L is suppressed, suggesting hyperthyroidism. "
                               "Antithyroid therapy review and endocrinology referral advised.",
                "explanation": f"TSH ({tsh_raw} mIU/L) is below 0.4 mIU/L (lower limit of normal). "
                               "Suppressed TSH indicates autonomous thyroid hormone production consistent with "
                               "hyperthyroidism or thyrotoxicosis. FT4/FT3 and thyroid antibodies should be checked.",
                "icon":        "🟡",
            })

    return alerts


def _check_polypharmacy_risk(medications) -> list[dict]:
    alerts = []

    if len(medications) >= 10:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Polypharmacy Detected",
            "message":     f"{len(medications)} medications identified. Polypharmacy increases risk of "
                           "drug-drug interactions and adverse effects. Medication reconciliation recommended.",
            "explanation": f"{len(medications)} distinct medications were matched (threshold: ≥10). "
                           "Polypharmacy is defined as concurrent use of ≥5 medications (minor) or ≥10 (major). "
                           "It increases risk of adverse drug reactions, falls, and non-adherence "
                           "(Maher et al., Age & Ageing 2014).",
            "icon":        "🟡",
        })
    elif len(medications) >= 6:
        alerts.append({
            "level":       LOW,
            "title":       "Multiple Medications",
            "message":     f"{len(medications)} medications noted. Periodic review of all medications "
                           "is advisable to check for interactions.",
            "explanation": f"{len(medications)} distinct medications were matched (threshold: 6–9). "
                           "This is in the minor polypharmacy range. Routine annual medication review "
                           "is recommended to check for unnecessary, duplicate, or interacting drugs.",
            "icon":        "🟢",
        })

    return alerts


def _check_mental_health_risk(diseases, medications) -> list[dict]:
    alerts = []
    mh_conditions = ["depression", "anxiety", "schizophrenia", "bipolar", "psychosis", "suicidal"]

    has_mh = any(any(mhc in d for mhc in mh_conditions) for d in diseases)

    if has_mh:
        alerts.append({
            "level":       MEDIUM,
            "title":       "Mental Health Condition Documented",
            "message":     "Mental health diagnosis noted. Ensure psychosocial support, "
                           "medication adherence monitoring, and regular follow-up.",
            "explanation": "One or more of the following mental health keywords was matched in the "
                           "diagnoses: depression, anxiety, schizophrenia, bipolar, psychosis, suicidal. "
                           "These conditions benefit from structured follow-up, adherence monitoring, "
                           "and multidisciplinary care (NICE CG90/CG178).",
            "icon":        "🟡",
        })

    return alerts


# ── Numeric Parsing Helpers ───────────────────────────────────────────────────

def _parse_numeric(value_str: str) -> float | None:
    """Extract the first number from a value string like '8.5%' or '140 mg/dL'."""
    if not value_str:
        return None
    match = re.search(r'[\d.]+', str(value_str))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    return None


def _parse_bp(bp_str: str) -> tuple:
    """
    Parse blood pressure string like '140/90' or '140/90 mmHg'.
    Returns (systolic, diastolic) as floats, or (None, None) on failure.
    """
    if not bp_str:
        return None, None
    match = re.search(r'(\d+)\s*/\s*(\d+)', str(bp_str))
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            pass
    return None, None
