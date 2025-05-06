import streamlit as st
import spacy
import matplotlib.pyplot as plt

# Load spaCy model with fallback using spacy.cli (works better in hosted environments)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Helper to get noun phrase with modifiers
def get_full_noun_phrase(token):
    words = [(child.i, child.text) for child in token.children if child.dep_ in ["det", "amod", "compound"]]
    words.append((token.i, token.text))
    words.sort()
    return " ".join([w[1] for w in words])

# Define a function to draw a Reed-Kellogg-style diagram with modifiers

def draw_basic_diagram(subject, verb, obj, subject_mod=None, object_mod=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot([0.1, 0.9], [1, 1], color="black")  # subject-predicate baseline
    ax.plot([0.5, 0.5], [1, 0.85], color="black")  # divider

    ax.text(0.3, 1.05, subject, ha='center', va='bottom', fontsize=12)
    ax.text(0.7, 1.05, verb, ha='center', va='bottom', fontsize=12)

    if obj:
        ax.plot([0.7, 0.9], [1, 1], color="black")
        ax.text(0.9, 1.05, obj, ha='center', va='bottom', fontsize=12)

    if subject_mod:
        ax.plot([0.27, 0.3], [0.85, 1], color="black")  # slanted line
        ax.text(0.27, 0.8, subject_mod, ha='center', va='top', fontsize=10)

    if object_mod:
        ax.plot([0.87, 0.9], [0.85, 1], color="black")
        ax.text(0.87, 0.8, object_mod, ha='center', va='top', fontsize=10)

    ax.axis("off")
    return fig

# Streamlit app interface
st.title("Reed-Kellogg Sentence Diagrammer")
sentence = st.text_input("Enter a sentence to diagram:")

if sentence:
    doc = nlp(sentence)
    subject = obj = subject_mod = object_mod = ""
    verb_tokens = []

    for token in doc:
        if token.dep_ == "nsubj":
            subject = get_full_noun_phrase(token)
            for child in token.children:
                if child.dep_ == "det":
                    subject_mod = child.text
        elif token.dep_ == "aux":
            verb_tokens.append(token.text)
        elif token.dep_ == "ROOT":
            verb_tokens.append(token.text)
        elif token.dep_ == "dobj":
            obj = get_full_noun_phrase(token)
            for child in token.children:
                if child.dep_ == "det":
                    object_mod = child.text

    verb = " ".join(verb_tokens)

    if subject and verb:
        fig = draw_basic_diagram(subject, verb, obj, subject_mod, object_mod)
        st.pyplot(fig)
    else:
        st.warning("This prototype only handles simple subject-verb-object sentences.")
