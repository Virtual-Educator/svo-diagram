import streamlit as st
import spacy
import matplotlib.pyplot as plt

# Load spaCy model with fallback using spacy.cli
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Function to draw a Reed-Kellogg-style diagram with modifiers

def draw_diagram(subject, verb, obj, subject_mod=None, object_mod=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    # Baseline for subject and predicate
    ax.plot([0.1, 0.9], [1, 1], color="black")
    # Divider between subject and predicate
    ax.plot([0.5, 0.5], [1, 0.85], color="black")

    # Plot the main words on the baseline
    ax.text(0.3, 1.05, subject, ha='center', va='bottom', fontsize=12)
    ax.text(0.7, 1.05, verb, ha='center', va='bottom', fontsize=12)
    if obj:
        ax.plot([0.7, 0.9], [1, 1], color="black")
        ax.text(0.9, 1.05, obj, ha='center', va='bottom', fontsize=12)

    # Plot subject modifiers on a slanted line
    if subject_mod:
        ax.plot([0.27, 0.3], [0.85, 1], color="black")
        ax.text(0.27, 0.80, subject_mod, ha='center', va='top', fontsize=10)

    # Plot object modifiers on a slanted line
    if object_mod:
        ax.plot([0.87, 0.9], [0.85, 1], color="black")
        ax.text(0.87, 0.80, object_mod, ha='center', va='top', fontsize=10)

    ax.axis("off")
    return fig

# Streamlit app interface
st.title("Reed-Kellogg Sentence Diagrammer")
sentence = st.text_input("Enter a sentence to diagram:")

if sentence:
    doc = nlp(sentence)
    subject = ""
    verb_tokens = []
    obj = ""
    subject_mod = ""
    object_mod = ""

    # Extract subject, predicate (including auxiliaries), object, and their modifiers
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
            mods = [child.text for child in token.children if child.dep_ in ["det", "amod", "compound"]]
            subject_mod = " ".join(mods)
        elif token.dep_ == "aux":
            verb_tokens.append(token.text)
        elif token.dep_ == "ROOT":
            verb_tokens.append(token.text)
        elif token.dep_ == "dobj":
            obj = token.text
            mods = [child.text for child in token.children if child.dep_ in ["det", "amod", "compound"]]
            object_mod = " ".join(mods)

    verb = " ".join(verb_tokens)

    if subject and verb:
        fig = draw_diagram(subject, verb, obj, subject_mod, object_mod)
        st.pyplot(fig)
    else:
        st.warning("This prototype only handles simple subject-verb-object sentences.")
