import streamlit as st
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load spaCy model
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

# Define helper functions
def get_modifiers(token, dep_types=None):
    if dep_types is None:
        dep_types = ["det", "amod", "compound", "nummod", "poss"]
    return [child for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in dep_types]

def get_full_np(token):
    parts = []
    # Include det, compound, amod, nummod, poss
    for child in sorted(token.children, key=lambda c: c.i):
        if child.dep_ in ["det", "compound", "amod", "nummod", "poss"]:
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)

def get_verb_phrase(root):
    aux_tokens = [c.text for c in root.children if c.dep_ in ['aux', 'auxpass']]
    neg_tokens = [c.text for c in root.children if c.dep_ == 'neg']
    all_parts = aux_tokens + neg_tokens + [root.text]
    # Sort by position in sentence
    positions = {}
    for t in aux_tokens + neg_tokens:
        for token in root.doc:
            if token.text == t:
                positions[t] = token.i
    positions[root.text] = root.i
    
    return " ".join([t for t in sorted(all_parts, key=lambda t: positions.get(t, 999))])

def get_adverbs(token):
    return [child for child in token.children if child.dep_ == "advmod"]

def get_prep_phrases(token):
    preps = []
    for child in token.children:
        if child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj:
                preps.append((child, pobj))
    return preps

def get_subject_complement(root):
    for child in root.children:
        if child.dep_ in ["attr", "acomp"]:
            return child
    return None

def draw_modifiers(ax, x_base, y_base, modifiers, direction="left"):
    offset = -0.05 if direction == "left" else 0.05
    for i, mod in enumerate(modifiers):
        sl_y = y_base + 0.1 + i * 0.08
        sl_x = x_base + offset * (i + 1) * 2
        
        # Draw slanted line
        ax.plot([x_base, sl_x], [y_base, sl_y], color='black', linewidth=1.5)
        
        # Add modifier text
        mod_text = mod.text
        ha = 'right' if direction == "left" else 'left'
        ax.text(sl_x, sl_y + 0.02, mod_text, ha=ha, va='bottom', fontsize=12)

def draw_adverbs(ax, x_root, y_base, adverbs):
    # Draw adverbs in a vertical stack below the verb
    for i, adv in enumerate(adverbs):
        y_pos = y_base - 0.15 - i * 0.1
        
        # Draw horizontal line
        line_length = 0.2
        x_start = x_root - line_length/2
        x_end = x_root + line_length/2
        ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=1.5)
        
        # Add diagonal connector to verb
        ax.plot([x_root, x_root], [y_base - 0.15, y_pos], color='black', linewidth=1.5, linestyle=':')
        
        # Add adverb text
        ax.text(x_root, y_pos + 0.02, adv.text, ha='center', va='bottom', fontsize=12)

def draw_prep_phrases(ax, x_anchor, y_base, prep_phrases, attachment_type):
    spacing = 0.3
    for i, (prep, obj) in enumerate(prep_phrases):
        # Position for prepositional phrase
        x_pp = x_anchor + (i - len(prep_phrases)/2 + 0.5) * spacing
        y_pp = y_base - 0.15
        
        # Draw diagonal line down from anchor
        ax.plot([x_anchor, x_pp], [y_base, y_pp], color='black', linewidth=1.5, linestyle=':')
        
        # Draw horizontal line for preposition
        pp_width = 0.2
        ax.plot([x_pp - pp_width/2, x_pp + pp_width/2], [y_pp, y_pp], color='black', linewidth=1.5)
        
        # Add preposition text
        ax.text(x_pp, y_pp + 0.02, prep.text, ha='center', va='bottom', fontsize=12)
        
        # Draw object of preposition
        y_obj = y_pp - 0.1
        ax.plot([x_pp, x_pp], [y_pp, y_obj], color='black', linewidth=1.5)
        
        # Draw horizontal line for object
        obj_width = 0.25
        ax.plot([x_pp - obj_width/2, x_pp + obj_width/2], [y_obj, y_obj], color='black', linewidth=1.5)
        
        # Add object text
        obj_text = get_full_np(obj)
        ax.text(x_pp, y_obj + 0.02, obj_text, ha='center', va='bottom', fontsize=12)
        
        # Draw object modifiers
        obj_mods = get_modifiers(obj)
        if obj_mods:
            for j, mod in enumerate(obj_mods):
                sl_y = y_obj + 0.08 + j * 0.08
                sl_x = x_pp - 0.05 * (j + 1)
                
                # Draw slanted line
                ax.plot([x_pp, sl_x], [y_obj, sl_y], color='black', linewidth=1.5)
                
                # Add modifier text
                ax.text(sl_x, sl_y + 0.02, mod.text, ha='right', va='bottom', fontsize=10)

def draw_reed_kellogg(doc, fig, ax):
    # Identify main elements
    subjects = [t for t in doc if t.dep_ in ["nsubj", "nsubjpass"]]
    if not subjects:
        st.warning("Could not identify a subject in this sentence.")
        return
    
    subj = subjects[0]  # Take the first subject if multiple exist
    
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root:
        st.warning("Could not identify the root verb in this sentence.")
        return
    
    # Determine object phrase (direct object)
    dobj = next((t for t in doc if t.dep_ == "dobj"), None)
    
    # Subject complement (predicate nominative/adjective)
    subj_complement = get_subject_complement(root)
    
    # Coordinates for baseline with better spacing
    fig_width = 10
    y_base = 0.5
    padding = 0.15
    
    x_subj = padding + 0.1
    x_root = fig_width/2
    x_obj = None
    
    if dobj:
        x_obj = fig_width - padding - 0.1
    elif subj_complement:
        x_obj = fig_width - padding - 0.1
    
    # Draw main baseline
    ax.plot([x_subj, x_root], [y_base, y_base], color='black', linewidth=2)
    
    # Draw vertical divider at root
    ax.plot([x_root, x_root], [y_base, y_base - 0.15], color='black', linewidth=2)
    
    # Add subject and verb
    subj_text = get_full_np(subj)
    verb_text = get_verb_phrase(root)
    
    ax.text(x_subj, y_base + 0.03, subj_text, ha='center', va='bottom', fontsize=14)
    ax.text(x_root, y_base + 0.03, verb_text, ha='center', va='bottom', fontsize=14)
    
    # Draw object or complement if it exists
    if dobj or subj_complement:
        ax.plot([x_root, x_obj], [y_base, y_base], color='black', linewidth=2)
        if dobj:
            obj_text = get_full_np(dobj)
            ax.text(x_obj, y_base + 0.03, obj_text, ha='center', va='bottom', fontsize=14)
            
            # Draw object modifiers slanted
            obj_mods = get_modifiers(dobj)
            draw_modifiers(ax, x_obj, y_base, obj_mods, direction="right")
        elif subj_complement:
            comp_text = get_full_np(subj_complement)
            ax.text(x_obj, y_base + 0.03, comp_text, ha='center', va='bottom', fontsize=14)
            
            # Draw complement modifiers
            comp_mods = get_modifiers(subj_complement)
            draw_modifiers(ax, x_obj, y_base, comp_mods, direction="right")
    
    # Draw subject modifiers
    subj_mods = get_modifiers(subj)
    draw_modifiers(ax, x_subj, y_base, subj_mods, direction="left")
    
    # Draw adverbs for the verb
    adverbs = get_adverbs(root)
    if adverbs:
        draw_adverbs(ax, x_root, y_base, adverbs)
    
    # Draw prepositional phrases attached to verb
    prep_phrases = get_prep_phrases(root)
    if prep_phrases:
        draw_prep_phrases(ax, x_root, y_base - 0.2, prep_phrases, "verb")
    
    # Draw prepositional phrases attached to object
    if dobj:
        obj_preps = get_prep_phrases(dobj)
        if obj_preps:
            draw_prep_phrases(ax, x_obj, y_base - 0.2, obj_preps, "object")
    
    # Draw prepositional phrases attached to subject
    subj_preps = get_prep_phrases(subj)
    if subj_preps:
        draw_prep_phrases(ax, x_subj, y_base - 0.2, subj_preps, "subject")

def main():
    # Load spaCy model first
    nlp = load_nlp_model()
    
    st.title("Reed-Kellogg Sentence Diagrammer")
    
    # Add UI elements for explanation
    with st.expander("About Reed-Kellogg Diagrams"):
        st.markdown("""
        **Reed-Kellogg sentence diagrams** are a method to graphically represent the grammatical structure of sentences.
        
        In these diagrams:
        - The main horizontal line contains the subject and predicate (verb)
        - A vertical line separates subject from predicate
        - Direct objects or complements appear on the horizontal line after the verb
        - Modifiers appear on slanted lines attached to the words they modify
        - Prepositional phrases appear on separate platforms below
        - Adverbs modifying verbs appear below the verb
        
        Enter a sentence below to see it diagrammed!
        """)
    
    # Main input
    text = st.text_area("Enter a sentence to diagram:", height=100)
    
    if st.button("Diagram Sentence") or text:
        if text.strip():
            try:
                doc = nlp(text)
                
                # Check if we have a complete sentence with subject and verb
                has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc)
                has_root = any(token.dep_ == "ROOT" for token in doc)
                
                if not has_subject or not has_root:
                    st.warning("This doesn't appear to be a complete sentence with both subject and verb.")
                else:
                    # Create diagram
                    fig, ax = plt.subplots(figsize=(12, 8))
                    draw_reed_kellogg(doc, fig, ax)
                    
                    # Set figure limits with padding
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 2)
                    ax.axis('off')
                    
                    # Display diagram
                    st.pyplot(fig)
                    
                    # Show parse information (can be hidden)
                    with st.expander("View Sentence Analysis Details", expanded=False):
                        st.subheader("Sentence Structure Analysis")
                        cols = st.columns(3)
                        cols[0].markdown("**Word**")
                        cols[1].markdown("**Part of Speech**")
                        cols[2].markdown("**Dependency**")
                        
                        for token in doc:
                            cols = st.columns(3)
                            cols[0].write(token.text)
                            cols[1].write(token.pos_)
                            cols[2].write(f"{token.dep_} (to {token.head.text})")
            except Exception as e:
                st.error(f"Error processing the sentence: {str(e)}")
                st.info("Try a simpler sentence or check for typos.")
        else:
            st.info("Please enter a sentence to diagram.")
    
    # Add examples section
    with st.expander("Example Sentences to Try"):
        examples = [
            "The happy dog chased the red ball.",
            "John quickly ran to the store.",
            "The students in the classroom studied diligently.",
            "My friend gave me a beautiful gift yesterday.",
            "The chef with the tall hat prepared delicious meals for the guests."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Try Example {i+1}"):
                st.session_state.text = example
                st.experimental_rerun()

if __name__ == "__main__":
    main()
