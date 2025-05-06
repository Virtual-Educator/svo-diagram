import streamlit as st
import spacy
from graphviz import Digraph

# Load spaCy model (requires en_core_web_sm installed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")
st.title("Reed-Kellogg Sentence Diagrammer (Advanced)")

sentence = st.text_input("Enter a sentence to diagram:")
if sentence:
    doc = nlp(sentence)
    
    # Build a Graphviz directed graph for dependency structure
    dot = Digraph(format='svg')
    dot.attr(rankdir='LR', splines='polyline')
    dot.attr('node', shape='plaintext')

    # Create nodes: use label with token and dep tag
    for token in doc:
        label = f"{token.text}\n<sub>{token.dep_}</sub>"
        dot.node(str(token.i), label)

    # Create edges
    for token in doc:
        if token.dep_ != 'ROOT':
            dot.edge(str(token.head.i), str(token.i))

    # Render diagram
    st.graphviz_chart(dot)
    
    st.markdown("---")
    st.markdown("**Notes:** this view shows the full dependency tree, including subjects, verbs, objects, modifiers, and clauses. You can hover over tokens to see their part-of-speech and dependency labels.")
else:
    st.info("Enter a sentence to see its full dependency diagram.")
