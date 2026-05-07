"""
Text utility functions for MedVis-X pipeline.
"""
import re
from typing import List


def clean_clinical_text(text: str) -> str:
    """Clean and normalize clinical text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize common abbreviations
    abbreviations = {
        r'\bpt\b': 'patient',
        r'\bhx\b': 'history',
        r'\bdx\b': 'diagnosis',
        r'\btx\b': 'treatment',
        r'\brx\b': 'prescription',
        r'\bsx\b': 'symptoms',
        r'\bbp\b': 'blood pressure',
        r'\bhr\b': 'heart rate',
        r'\brr\b': 'respiratory rate',
        r'\bspo2\b': 'oxygen saturation',
        r'\bwbc\b': 'white blood cell count',
        r'\brbc\b': 'red blood cell count',
        r'\bhgb\b': 'hemoglobin',
        r'\bhba1c\b': 'glycated hemoglobin',
    }
    for abbr, full in abbreviations.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    return text


def extract_dosage_info(text: str) -> List[dict]:
    """Extract dosage information from prescription text."""
    dosage_pattern = re.compile(
        r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)\s*'
        r'(?:(\d+)\s*(?:times?|x)\s*(?:daily|a day|per day|/day))?',
        re.IGNORECASE
    )
    results = []
    for match in dosage_pattern.finditer(text):
        results.append({
            "medication": match.group(1),
            "dose": float(match.group(2)),
            "unit": match.group(3).lower(),
            "frequency": int(match.group(4)) if match.group(4) else None,
        })
    return results


def format_entity_summary(entities: dict) -> str:
    """Format extracted entities into a readable summary."""
    lines = []
    for category, items in entities.items():
        if category == "raw_entities":
            continue
        if not isinstance(items, list) or not items:
            continue
        entity_texts = [
            f"{e['text']} ({e['score']:.0%})"
            for e in items if isinstance(e, dict) and "text" in e
        ]
        if entity_texts:
            lines.append(f"**{category.replace('_', ' ').title()}:** "
                         f"{', '.join(entity_texts)}")
    return "\n".join(lines) if lines else "No entities extracted."
