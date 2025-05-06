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

# Define a function to draw a basic Reed-Kellogg-style diagram
def draw_basic_diagram(subject, verb, obj):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot([0, 1], [1, 1], color="black")  # horizontal base line
    ax.text(0, 1.05, subject, ha='center', va='bottom', fontsize=12)
    ax.text(0.5, 1.05, verb, ha='center', va='bottom', fontsize=12)
    ax.text(1, 1.05, obj, ha='center', va='bottom', fontsize=12)
    ax.axis("off")
    return fig

# Streamlit app interface
st.title("Reed-Kellogg Sentence Diagrammer")
sentence = st.text_input("Enter a sentence to diagram:")

if sentence:
    doc = nlp(sentence)
    subject = verb = obj = ""

    # Simple SVO extraction
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
        elif token.dep_ == "ROOT":
            verb = token.text
        elif token.dep_ == "dobj":
            obj = token.text

    if subject and verb:
        fig = draw_basic_diagram(subject, verb, obj)
        st.pyplot(fig)
    else:
        st.warning("This prototype only handles simple subject-verb-object sentences.")
