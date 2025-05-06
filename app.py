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

# Helper to get noun phrase text and modifier (det, amod, compound)
def get_full_noun_phrase(token):
    parts = [(token.i, token.text)]
    for child in token.children:
        if child.dep_ in ["det", "amod", "compound"]:
            parts.append((child.i, child.text))
    parts.sort()
    return " ".join([text for _, text in parts])

# Draw Reed-Kellogg diagram with optional object phrase (direct or prepositional)
def draw_diagram(subject, verb, obj_phrase=None, subject_mod=None, object_mod=None):
    end_x = 1.1
    fig, ax = plt.subplots(figsize=(10, 3))
    # Baseline
    ax.plot([0.1, end_x], [1, 1], color="black")
    # Divider between subject and predicate
    ax.plot([0.5, 0.5], [1, 0.85], color="black")

    # Labels on baseline
    ax.text(0.3, 1.05, subject, ha='center', va='bottom', fontsize=12)
    ax.text(0.7, 1.05, verb, ha='center', va='bottom', fontsize=12)
    if obj_phrase:
        ax.plot([0.7, end_x], [1, 1], color="black")
        ax.text(end_x, 1.05, obj_phrase, ha='center', va='bottom', fontsize=12)

    # Subject modifier slanted line
    if subject_mod:
        ax.plot([0.27, 0.3], [0.85, 1], color="black")
        ax.text(0.27, 0.8, subject_mod, ha='center', va='top', fontsize=10)

    # Object modifier slanted line
    if object_mod and obj_phrase:
        ax.plot([end_x - 0.03, end_x], [0.85, 1], color="black")
        ax.text(end_x - 0.03, 0.8, object_mod, ha='center', va='top', fontsize=10)

    ax.axis("off")
    return fig

# Streamlit interface
st.title("Reed-Kellogg Sentence Diagrammer")
sentence = st.text_input("Enter a sentence to diagram:")

if sentence:
    doc = nlp(sentence)
    subject = ""
    subject_mod = ""
    verb_tokens = []
    obj_phrase = ""
    object_mod = ""
    prep = None

    # Parse tokens
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
            # collect first determiner or adjective as modifier
            for child in token.children:
                if child.dep_ in ["det", "amod", "compound"]:
                    subject_mod = child.text
                    break
        elif token.dep_ == "aux":
            verb_tokens.append(token.text)
        elif token.dep_ == "ROOT":
            verb_tokens.append(token.text)
        elif token.dep_ == "dobj":
            # direct object phrase
            obj_phrase = get_full_noun_phrase(token)
            for child in token.children:
                if child.dep_ in ["det", "amod", "compound"]:
                    object_mod = child.text
                    break
        elif token.dep_ == "prep":
            # handle prepositional phrase
            prep = token.text
            for child in token.children:
                if child.dep_ == "pobj":
                    # object of preposition
                    np = get_full_noun_phrase(child)
                    obj_phrase = f"{prep} {np}"
                    for g in child.children:
                        if g.dep_ in ["det", "amod", "compound"]:
                            object_mod = g.text
                            break
                    break

    verb = " ".join(verb_tokens)

    if subject and verb:
        fig = draw_diagram(subject, verb, obj_phrase or None, subject_mod or None, object_mod or None)
        st.pyplot(fig)
    else:
        st.warning("This prototype only handles simple sentences with subjects and verbs.")
