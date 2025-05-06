import streamlit as st
import spacy
import matplotlib.pyplot as plt

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
st.title("Reed-Kellogg Sentence Diagrammer (Refined Layout)")

# Helper: gather modifiers for a token
def get_modifiers(token):
    return [child.text for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in ["det", "amod", "compound"]]

# Helper: build full noun phrase text
def get_full_np(token):
    parts = []
    # include det, compound, amod
    for child in sorted(token.children, key=lambda c: c.i):
        if child.dep_ in ["det", "compound", "amod"]:
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)

# Streamlit interface
text = st.text_input("Enter a sentence to diagram:")
if text:
    doc = nlp(text)
    # Identify main elements
    subj = next((t for t in doc if t.dep_ == "nsubj"), None)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    # Determine object phrase (direct or prepositional)
    obj_phrase = None
    object_mods = []

    # Direct object
    dobj = next((t for t in doc if t.dep_ == "dobj"), None)
    if dobj:
        obj_phrase = get_full_np(dobj)
        object_mods = get_modifiers(dobj)
    elif root:
        # Prepositional object
        prep = next((c for c in root.children if c.dep_ == "prep"), None)
        if prep:
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj:
                # include prep plus full noun phrase
                np = get_full_np(pobj)
                obj_phrase = f"{prep.text} {np}"
                object_mods = get_modifiers(pobj)
    
    if subj and root:
        # Coordinates for baseline
        x_subj, x_root, x_obj = 0.1, 0.5, (0.9 if obj_phrase else None)
        y_base = 0.5
        # Build baseline points
        xs = [x_subj, x_root]
        ys = [y_base, y_base]
        if x_obj is not None:
            xs.append(x_obj)
            ys.append(y_base)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        # Draw baseline horizontally
        ax.plot(xs, ys, color='black')
        # Divider at root
        ax.plot([x_root, x_root], [y_base, y_base - 0.1], color='black')
        
        # Assemble full verb phrase (aux + ROOT)
        verb_tokens = [c.text for c in root.children if c.dep_ == 'aux'] + [root.text]
        verb = " ".join(verb_tokens)
        
        # Labels on baseline
        ax.text(x_subj, y_base + 0.02, subj.text, ha='center', va='bottom', fontsize=12)
        ax.text(x_root, y_base + 0.02, verb, ha='center', va='bottom', fontsize=12)
        if obj_phrase and x_obj is not None:
            ax.text(x_obj, y_base + 0.02, obj_phrase, ha='center', va='bottom', fontsize=12)
        
        # Draw subject modifiers slanted
        for i, mod in enumerate(get_modifiers(subj)):
            sl_y = y_base + 0.1 + i * 0.05
            sl_x = x_subj - 0.05 - i * 0.02
            ax.plot([sl_x, x_subj], [sl_y, y_base], color='black')
            ax.text(sl_x, sl_y + 0.02, mod, ha='center', va='bottom', fontsize=10)
        
        # Draw object modifiers slanted
        if obj_phrase and x_obj is not None:
            for i, mod in enumerate(object_mods):
                sl_y = y_base + 0.1 + i * 0.05
                sl_x = x_obj + 0.05 + i * 0.02
                ax.plot([x_obj, sl_x], [y_base, sl_y], color='black')
                ax.text(sl_x, sl_y + 0.02, mod, ha='center', va='bottom', fontsize=10)
        
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Could not find subject and root to diagram.")
else:
    st.info("Enter a sentence to see its Reed-Kellogg diagram.")
