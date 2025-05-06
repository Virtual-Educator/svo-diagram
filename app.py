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

# Streamlit interface
text = st.text_input("Enter a sentence to diagram:")
if text:
    doc = nlp(text)
    # Identify main elements
    subj = next((t for t in doc if t.dep_ == "nsubj"), None)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    obj = next((t for t in doc if t.dep_ == "dobj"), None)
    # fallback to prepositional object of first prep
    if not obj and root:
        prep = next((c for c in root.children if c.dep_ == "prep"), None)
        if prep:
            obj = next((c for c in prep.children if c.dep_ == "pobj"), None)
    
    if subj and root:
        # Coordinates for baseline
        x_subj, x_root, x_obj = 0.1, 0.5, (0.9 if obj else None)
        y_base = 0.5
        # Build baseline points
        xs = [x_subj, x_root]
        ys = [y_base, y_base]
        if x_obj:
            xs.append(x_obj)
            ys.append(y_base)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        # Draw baseline
        ax.plot(xs, ys, color='black')
        # Divider at root
        ax.plot([x_root, x_root], [y_base, y_base - 0.1], color='black')
        
        # Assemble full verb phrase (aux + ROOT)
        verb_tokens = [c.text for c in root.children if c.dep_ == 'aux'] + [root.text]
        verb = " ".join(verb_tokens)
        
        # Labels on baseline
        ax.text(x_subj, y_base + 0.02, subj.text, ha='center', va='bottom', fontsize=12)
        ax.text(x_root, y_base + 0.02, verb, ha='center', va='bottom', fontsize=12)
        if obj and x_obj:
            # show only head for obj phrase
            ax.text(x_obj, y_base + 0.02, obj.text, ha='center', va='bottom', fontsize=12)
        
        # Draw subject modifiers slanted
        for i, mod in enumerate(get_modifiers(subj)):
            sl_y = y_base + 0.1 + i * 0.05
            sl_x = x_subj - 0.05 - i * 0.02
            ax.plot([sl_x, x_subj], [sl_y, y_base], color='black')
            ax.text(sl_x, sl_y + 0.02, mod, ha='center', va='bottom', fontsize=10)
        
        # Draw object modifiers slanted
        if obj and x_obj:
            for i, mod in enumerate(get_modifiers(obj)):
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
