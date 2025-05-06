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

# Draw Reed-Kellogg diagram with multiple modifiers and compounds
def draw_diagram(subject, verb, obj_phrase=None, subject_mods=None, object_mods=None):
    end_x = 1.1
    fig, ax = plt.subplots(figsize=(10, 3))
    # Baseline
    ax.plot([0.1, end_x], [1, 1], color="black")
    # Divider between subject and predicate
    ax.plot([0.5, 0.5], [1, 0.85], color="black")

    # Main labels on baseline
    subj_x = 0.3
    verb_x = 0.7
    ax.text(subj_x, 1.05, subject, ha='center', va='bottom', fontsize=12)
    ax.text(verb_x, 1.05, verb, ha='center', va='bottom', fontsize=12)
    if obj_phrase:
        ax.plot([verb_x, end_x], [1, 1], color="black")
        ax.text(end_x, 1.05, obj_phrase, ha='center', va='bottom', fontsize=12)

    # Subject modifiers (det, amod, compound)
    if subject_mods:
        for i, mod in enumerate(subject_mods):
            x_start = subj_x - 0.05 * (i+1)
            y_start = 0.85 - 0.05 * i
            ax.plot([x_start, subj_x], [y_start, 1], color="black")
            ax.text(x_start, y_start - 0.03, mod, ha='center', va='top', fontsize=10)

    # Object modifiers
    if object_mods and obj_phrase:
        for i, mod in enumerate(object_mods):
            x_end = end_x
            x_start = x_end - 0.05 * (i+1)
            y_start = 0.85 - 0.05 * i
            ax.plot([x_start, x_end], [y_start, 1], color="black")
            ax.text(x_start, y_start - 0.03, mod, ha='center', va='top', fontsize=10)

    ax.axis("off")
    return fig

# Streamlit interface
st.title("Reed-Kellogg Sentence Diagrammer")
sentence = st.text_input("Enter a sentence to diagram:")

if sentence:
    doc = nlp(sentence)
    subject = ""
    subject_mods = []
    verb_tokens = []
    obj_phrase = ""
    object_mods = []
    prep = None

    # Parse tokens for S, V, O/PP
    for token in doc:
        if token.dep_ == "nsubj":
            subject = token.text
            # collect all modifiers (det, amod, compound)
            subject_mods = [child.text for child in sorted(token.children, key=lambda c: c.i)
                            if child.dep_ in ["det", "amod", "compound"]]
        elif token.dep_ == "aux":
            verb_tokens.append(token.text)
        elif token.dep_ == "ROOT":
            verb_tokens.append(token.text)
        elif token.dep_ == "dobj":
            # direct object
            obj_phrase = token.text
            object_mods = [child.text for child in sorted(token.children, key=lambda c: c.i)
                           if child.dep_ in ["det", "amod", "compound"]]
        elif token.dep_ == "prep":
            # prepositional phrase
            prep = token.text
            for child in token.children:
                if child.dep_ == "pobj":
                    # object of preposition
                    obj_phrase = f"{prep} {child.text}"
                    object_mods = [g.text for g in sorted(child.children, key=lambda c: c.i)
                                   if g.dep_ in ["det", "amod", "compound"]]
                    break

    verb = " ".join(verb_tokens)

    if subject and verb:
        fig = draw_diagram(subject, verb, obj_phrase or None,
                           subject_mods or None, object_mods or None)
        st.pyplot(fig)
    else:
        st.warning("This prototype handles simple sentences with subject, verb, object, and a single PP.")
