# modules/insight_refiner.py

import re
import spacy
import random

nlp = spacy.load("en_core_web_sm")

TEMPLATES = [
    "How does {X} impact {Y}?",
    "What policies could improve {Y} considering {X}?",
    "Why might {X} be affecting {Y}?",
    "What can be inferred from the relationship between {X} and {Y}?",
    "How can we reduce or increase {Y} by changing {X}?"
]

def extract_entities(text):
    doc = nlp(text)
    chunks = list(set([chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]))
    return chunks

def generate_questions(text):
    entities = extract_entities(text)
    if len(entities) < 2:
        return ["What are the implications of this insight?"]

    questions = []
    for _ in range(min(3, len(entities))):
        X, Y = random.sample(entities, 2)
        template = random.choice(TEMPLATES)
        questions.append(template.format(X=X, Y=Y))
    return questions

def clean_and_structure(insight_block):
    # Clean Markdown or bold markers
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", insight_block).strip()
    return cleaned
