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
st.title("Reed-Kellogg Sentence Diagrammer (Custom Layout)")

class Node:
    def __init__(self, token):
        self.token = token
        self.children = []
        self.x = 0
        self.y = 0

    def add_child(self, node):
        self.children.append(node)

# Build full dependency tree excluding punctuation
def build_tree(token):
    node = Node(token)
    for child in token.children:
        if child.dep_ != 'punct':
            node.add_child(build_tree(child))
    return node

# Compute x positions by in-order traversal
def layout_tree(node, depth=0):
    node.y = -depth
    if not node.children:
        node.x = layout_tree.counter
        layout_tree.counter += 1
    else:
        for child in node.children:
            layout_tree(child, depth+1)
        # position at midpoint of children
        xs = [child.x for child in node.children]
        node.x = sum(xs) / len(xs)

layout_tree.counter = 0

# Draw the diagram with lines and labels
def draw_tree(node, ax):
    ax.text(node.x, node.y, node.token.text, ha='center', va='bottom', fontsize=10)
    for child in node.children:
        # Draw horizontal-vertical elbow
        ax.plot([node.x, child.x], [node.y, child.y], color='black')
        draw_tree(child, ax)

# Streamlit interface
text = st.text_input("Enter a sentence to diagram:")
if text:
    doc = nlp(text)
    root = next((tok for tok in doc if tok.dep_ == 'ROOT'), None)
    if root:
        layout_tree.counter = 0
        tree = build_tree(root)
        layout_tree(tree)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_tree(tree, ax)
        ax.invert_yaxis()
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("Could not find the root of the sentence.")
else:
    st.info("Enter a sentence to see its diagram.")
