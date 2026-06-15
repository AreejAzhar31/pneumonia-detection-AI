# ============================================================
#  recommendations.py — Module 4: Output & Recommendation System
#  Pneumonia Detection AI Project
#
#  Called by app.py after Module 3 (inference.py) runs:
#      report, meta = get_recommendation(label, confidence)
#      save_report(report)
# ============================================================
 
import os
import json
from datetime import datetime
 
# ── Output Directory ───────────────────────────────────────
REPORT_DIR = 'reports'
os.makedirs(REPORT_DIR, exist_ok=True)
 
# ============================================================
#  KNOWLEDGE BASE  (JSON-style, hardcoded)
#  Each class has: display_label, description, symptoms,
#  medications, cautions, follow_up
# ============================================================
 
KNOWLEDGE_BASE = {
    "normal": {
        "display_label": "No Pneumonia Detected",
        "severity_level": 0,
        "description": (
            "The chest X-ray shows no significant signs of pneumonia. "
            "Lung fields appear clear with no visible consolidation or infiltrates."
        ),
        "symptoms": [
            "No respiratory distress observed",
            "Lung fields appear clear",
            "No abnormal opacities detected"
        ],
        "medications": [
            "No medication required at this time",
            "Continue any existing prescribed treatments",
            "Maintain regular hydration and rest"
        ],
        "cautions": [
            "This result is AI-generated and must be confirmed by a licensed physician",
            "Seek medical attention if symptoms such as fever, cough, or breathlessness develop",
            "Schedule a follow-up if symptoms persist beyond 3–5 days"
        ],
        "follow_up": "Routine check-up recommended. No immediate action required."
    },
 
    "mild": {
        "display_label": "Mild Pneumonia (Viral)",
        "severity_level": 1,
        "description": (
            "The chest X-ray shows signs consistent with mild viral pneumonia. "
            "Mild interstitial infiltrates may be present. "
            "Early medical evaluation is strongly recommended."
        ),
        "symptoms": [
            "Mild fever (typically 38–39°C)",
            "Dry or productive cough",
            "Fatigue and general weakness",
            "Mild shortness of breath on exertion",
            "Possible chest discomfort"
        ],
        "medications": [
            "Antipyretics (e.g., Paracetamol 500mg) for fever management",
            "Rest and increased fluid intake",
            "Antiviral therapy only if prescribed by a physician",
            "Cough suppressants or expectorants as advised by a doctor",
            "Do NOT self-prescribe antibiotics — viral pneumonia does not respond to antibiotics"
        ],
        "cautions": [
            "This result is AI-generated and must be confirmed by a licensed physician",
            "Monitor oxygen saturation (SpO2) — seek emergency care if below 94%",
            "Avoid contact with vulnerable individuals (elderly, immunocompromised)",
            "Do not ignore worsening symptoms — escalation to severe is possible",
            "Antibiotics are NOT recommended for viral pneumonia without medical guidance"
        ],
        "follow_up": (
            "Consult a physician within 24–48 hours. "
            "A repeat X-ray may be advised after 7–10 days to monitor resolution."
        )
    },
 
    # ✅ FIX: 'pneumonia' is the label returned by the fixed inference.py
    # for the PNEUMONIA class. Maps to the same content as 'severe'
    # since the Kaggle dataset only has NORMAL/PNEUMONIA (no severity split).
    "pneumonia": {
        "display_label": "Pneumonia Detected",
        "severity_level": 2,
        "description": (
            "The chest X-ray shows signs consistent with pneumonia. "
            "Consolidation or infiltrates may be visible in the lung fields. "
            "Medical evaluation is strongly recommended."
        ),
        "symptoms": [
            "Fever and chills",
            "Productive cough",
            "Shortness of breath or difficulty breathing",
            "Chest pain worsening with deep breathing",
            "Fatigue and body aches",
            "Rapid breathing or rapid heart rate"
        ],
        "medications": [
            "Consult a physician immediately for appropriate antibiotic or antiviral therapy",
            "Antipyretics (e.g., Paracetamol) for fever under physician guidance",
            "Supplemental oxygen if SpO2 is below 94%",
            "Do NOT self-medicate — treatment depends on type (bacterial vs viral)"
        ],
        "cautions": [
            "⚠️  Seek medical attention promptly",
            "This result is AI-generated and must be confirmed by a licensed physician",
            "Monitor for worsening symptoms — go to ER if breathing becomes severely difficult",
            "Do NOT delay evaluation — pneumonia can progress rapidly if untreated"
        ],
        "follow_up": (
            "Visit a physician or hospital as soon as possible for clinical evaluation and treatment."
        )
    },
 
    "severe": {
        "display_label": "Severe Pneumonia (Bacterial)",
        "severity_level": 2,
        "description": (
            "The chest X-ray shows signs consistent with severe bacterial pneumonia. "
            "Significant consolidation or lobar involvement may be present. "
            "IMMEDIATE medical attention is required."
        ),
        "symptoms": [
            "High fever (above 39°C) with chills and rigors",
            "Severe productive cough, possibly with rust-coloured sputum",
            "Significant shortness of breath even at rest",
            "Chest pain worsening with deep breathing (pleuritic pain)",
            "Rapid breathing (tachypnoea) and rapid heart rate",
            "Confusion or altered mental status in severe cases",
            "Cyanosis (bluish discolouration of lips or fingertips)"
        ],
        "medications": [
            "URGENT: Intravenous or oral antibiotics as prescribed by a physician",
            "Common first-line agents include Amoxicillin-Clavulanate or Azithromycin (physician decision)",
            "Supplemental oxygen therapy if SpO2 is below 94%",
            "Antipyretics (e.g., Paracetamol) for fever under physician guidance",
            "Hospitalisation may be required for IV therapy and monitoring",
            "Do NOT delay treatment — bacterial pneumonia can be life-threatening"
        ],
        "cautions": [
            "⚠️  SEEK EMERGENCY MEDICAL CARE IMMEDIATELY",
            "This result is AI-generated and must be confirmed by a licensed physician",
            "Do NOT self-medicate — incorrect antibiotic use worsens outcomes",
            "Monitor SpO2 continuously — go to ER if below 92%",
            "Risk of sepsis if untreated — do not delay hospital evaluation",
            "Keep patient upright and calm; avoid strenuous activity"
        ],
        "follow_up": (
            "IMMEDIATE hospital evaluation required. "
            "Do not wait for scheduled appointments — visit an emergency department now."
        )
    }
}
 
# ============================================================
#  STEP 1 — Build Formatted Report String
# ============================================================
 
def _format_list(items, indent=4):
    """Format a list of strings with bullet points."""
    pad = ' ' * indent
    return '\n'.join(f"{pad}• {item}" for item in items)
 
def build_report(label, confidence, knowledge):
    """
    Build a human-readable text report from the knowledge base entry.
    Returns the full report string.
    """
    now       = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    display   = knowledge['display_label']
    conf_pct  = f"{confidence * 100:.1f}%"
    severity  = knowledge['severity_level']
    stars     = ('●' * severity) + ('○' * (2 - severity))   # ●○ / ●●○ / ●●●
 
    border = '═' * 58
 
    report = f"""
{border}
  PNEUMONIA DETECTION — RECOMMENDATION REPORT
{border}
  Date / Time  : {now}
  Diagnosis    : {display}
  Confidence   : {conf_pct}
  Severity     : [{stars}]  ({['Low', 'Moderate', 'High'][severity]})
{border}
 
  DESCRIPTION
  {knowledge['description']}
 
  OBSERVED / EXPECTED SYMPTOMS
{_format_list(knowledge['symptoms'])}
 
  RECOMMENDED MEDICATIONS & ACTIONS
{_format_list(knowledge['medications'])}
 
  ⚠  CAUTIONS & IMPORTANT NOTICES
{_format_list(knowledge['cautions'])}
 
  FOLLOW-UP ADVICE
    {knowledge['follow_up']}
 
{border}
  DISCLAIMER
    This report is generated by an AI model for informational
    purposes only. It does NOT replace professional medical
    diagnosis. Always consult a qualified physician before
    taking any medical action.
{border}
"""
    return report
 
# ============================================================
#  STEP 2 — Public API: get_recommendation()
# ============================================================
 
def get_recommendation(label: str, confidence: float):
    """
    Main entry point called by app.py / inference pipeline.
 
    Parameters
    ----------
    label      : str   — 'normal', 'mild', or 'severe'
    confidence : float — prediction confidence 0.0–1.0
 
    Returns
    -------
    report : str  — full formatted text report
    meta   : dict — structured data (label, confidence, knowledge entry)
    """
    label = label.lower().strip()
 
    if label not in KNOWLEDGE_BASE:
        raise ValueError(
            f"Unknown label '{label}'. Expected one of: {list(KNOWLEDGE_BASE.keys())}"
        )
 
    knowledge = KNOWLEDGE_BASE[label]
    report    = build_report(label, confidence, knowledge)
 
    meta = {
        'label':      label,
        'confidence': confidence,
        'knowledge':  knowledge
    }
 
    return report, meta
 
# ============================================================
#  STEP 3 — Public API: save_report()
# ============================================================
 
def save_report(report: str, filename: str = None) -> str:
    """
    Save the text report to the reports/ directory.
 
    Parameters
    ----------
    report   : str  — the full report string from get_recommendation()
    filename : str  — optional custom filename (auto-generated if None)
 
    Returns
    -------
    filepath : str — path where the report was saved
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f"report_{timestamp}.txt"
 
    filepath = os.path.join(REPORT_DIR, filename)
 
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
 
    print(f"\n[Module 4] Report saved → {filepath}")
    return filepath
 
# ============================================================
#  STEP 4 — Optional: save_report_json()
#  Saves structured meta as JSON alongside the text report
# ============================================================
 
def save_report_json(meta: dict, filename: str = None) -> str:
    """
    Save the structured recommendation metadata as JSON.
    Useful for downstream systems or dashboards.
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename  = f"report_{timestamp}.json"
 
    filepath = os.path.join(REPORT_DIR, filename)
 
    # Make a clean copy (remove non-serialisable items if any)
    export = {
        'label':         meta['label'],
        'confidence':    round(meta['confidence'], 4),
        'display_label': meta['knowledge']['display_label'],
        'severity_level': meta['knowledge']['severity_level'],
        'symptoms':      meta['knowledge']['symptoms'],
        'medications':   meta['knowledge']['medications'],
        'cautions':      meta['knowledge']['cautions'],
        'follow_up':     meta['knowledge']['follow_up'],
        'generated_at':  datetime.now().isoformat()
    }
 
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export, f, indent=2)
 
    print(f"[Module 4] JSON report saved → {filepath}")
    return filepath
 
# ============================================================
#  Standalone test — python modules/recommendations.py
# ============================================================
 
if __name__ == '__main__':
    print("\n=== Module 4 Standalone Test ===\n")
 
    for test_label, test_conf in [('normal', 0.94), ('mild', 0.81), ('severe', 0.97)]:
        print(f"\nTesting label='{test_label}', confidence={test_conf}")
        report, meta = get_recommendation(test_label, test_conf)
        print(report)
        save_report(report)
        save_report_json(meta)
 
    print("\n✓ All test cases passed. Reports saved in reports/")