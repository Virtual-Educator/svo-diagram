import streamlit as st
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math # Needed for text width estimation approximation

# --- spaCy Model Loading ---
@st.cache_resource
def load_nlp_model():
    """Loads the spaCy model, downloading if necessary."""
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        st.info(f"Downloading spaCy model: {model_name}...")
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)

# --- Helper Functions for NLP Analysis ---
def get_modifiers(token, dep_types=None):
    """Gets specified types of modifiers for a token, sorted by sentence order."""
    if dep_types is None:
        # Common dependents treated as simple modifiers attached below
        dep_types = ["det", "amod", "compound", "nummod", "poss", "advmod"] # Added advmod here for modifiers of nouns/adjectives
    return [child for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in dep_types]

def get_full_text(token):
    """Gets the text of a token including its tightly bound compound parts and possessives."""
    parts = []
    # Include preceding compound words and possessives if they modify this token
    for child in sorted(token.children, key=lambda c: c.i):
        if child.dep_ in ["compound", "poss"] and child.i < token.i:
            parts.extend(get_full_text(child).split())
        elif child.dep_ == 'case' and child.head.dep_ == 'poss': # Handle possessive 's
             parts.append(child.text)

    parts.append(token.text)
    return " ".join(parts)


def get_verb_phrase(root):
    """Constructs the verb phrase including auxiliaries and negation, ordered correctly."""
    verb_parts = {}
    # Include auxiliaries (aux, auxpass), negation (neg)
    for child in root.children:
        if child.dep_ in ['aux', 'auxpass', 'neg']:
            verb_parts[child.i] = child.text
    verb_parts[root.i] = root.text # Add the main verb

    # Sort by index (position in sentence) and join
    return " ".join(verb_parts[i] for i in sorted(verb_parts.keys()))

def get_prep_phrases(token):
    """Finds prepositional phrases attached to a token."""
    preps = []
    for child in token.children:
        if child.dep_ == "prep":
            # Find the object of the preposition (pobj)
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj:
                preps.append((child, pobj))
    return preps

def get_subject_complement(root):
    """Finds subject complements (predicate nominative/adjective)."""
    for child in root.children:
        if child.dep_ in ["attr", "acomp"]:
            return child
    return None

def get_direct_object(root):
    """Finds direct objects."""
    for child in root.children:
        if child.dep_ == "dobj":
            return child
    return None

def get_adverbial_modifiers(token):
     """Gets adverbial modifiers (advmod) attached directly to the token."""
     return [child for child in token.children if child.dep_ == "advmod"]

# --- Drawing Functions using Matplotlib ---

# Constants for drawing adjustments
MODIFIER_SLANT_OFFSET_X = 0.03 # Reduced horizontal spread for modifiers
MODIFIER_SLANT_OFFSET_Y = 0.08 # Vertical spacing for stacked modifiers
LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 13
FONT_SIZE_MOD = 11
Y_BASE = 0.7 # Adjusted base height for more room below
PP_Y_OFFSET = 0.15 # How far below baseline PP starts
PP_X_SPACING = 0.4 # Horizontal spacing between multiple PPs
TEXT_Y_OFFSET = 0.02 # Offset text slightly above its line

def estimate_text_width(text, fontsize=FONT_SIZE_MAIN):
    """Very rough approximation of text width based on character count."""
    # This is a placeholder. Real text width depends on font, characters, etc.
    # For Matplotlib, precise width calculation is non-trivial without rendering.
    return len(text) * fontsize * 0.006 # Adjust the multiplier as needed

def draw_modifiers(ax, x_base, y_base, modifiers, text_width_base=0):
    """Draws modifiers below the word they modify on slanted lines."""
    # Calculate starting x based on text width to center modifiers roughly
    start_x = x_base #- text_width_base / 2 * 0.8 # Slightly bias to the left

    for i, mod in enumerate(modifiers):
        mod_text = get_full_text(mod) # Use full text for compounds etc.
        # Position modifier below the baseline
        sl_y = y_base - (PP_Y_OFFSET / 2) - i * MODIFIER_SLANT_OFFSET_Y
        # Position horizontally with slight slant based on index
        sl_x = start_x - MODIFIER_SLANT_OFFSET_X * (i + 1)

        # Draw slanted line from baseline down to modifier position
        ax.plot([x_base, sl_x], [y_base, sl_y], color='black', linewidth=LINE_WIDTH)

        # Add modifier text slightly above its end-point
        ax.text(sl_x, sl_y + TEXT_Y_OFFSET, mod_text, ha='right', va='bottom', fontsize=FONT_SIZE_MOD)

        # Recursively draw modifiers of this modifier (e.g., adverbs modifying adjectives)
        sub_mods = get_modifiers(mod, dep_types=["advmod"]) # Only adverbs here usually
        if sub_mods:
             draw_modifiers(ax, sl_x, sl_y, sub_mods) # Use modifier's end point as new base


def draw_adverbial_modifiers(ax, x_anchor, y_anchor, adverbs):
    """Draws adverbs modifying the verb, below the verb."""
    for i, adv in enumerate(adverbs):
        adv_text = get_full_text(adv)
        text_width = estimate_text_width(adv_text, FONT_SIZE_MOD)
        line_len = max(text_width * 1.1, 0.15) # Make line slightly wider than text

        # Position below the anchor point
        y_pos = y_anchor - PP_Y_OFFSET - i * MODIFIER_SLANT_OFFSET_Y * 1.5
        x_start = x_anchor - line_len / 2

        # Draw slanted connector line from anchor (verb) to the adverb line
        ax.plot([x_anchor, x_start], [y_anchor, y_pos], color='black', linewidth=LINE_WIDTH)

        # Draw horizontal line for the adverb
        ax.plot([x_start, x_start + line_len], [y_pos, y_pos], color='black', linewidth=LINE_WIDTH)

        # Add adverb text
        ax.text(x_start + line_len / 2, y_pos + TEXT_Y_OFFSET, adv_text,
                ha='center', va='bottom', fontsize=FONT_SIZE_MOD)

        # Draw modifiers of the adverb (e.g., "very quickly")
        sub_mods = get_modifiers(adv, dep_types=["advmod"])
        if sub_mods:
             draw_modifiers(ax, x_start + line_len / 2, y_pos, sub_mods, text_width)


def draw_prep_phrases(ax, x_anchor, y_anchor, prep_phrases):
    """Draws prepositional phrases below the word they modify."""
    num_phrases = len(prep_phrases)
    for i, (prep, obj) in enumerate(prep_phrases):
        prep_text = prep.text
        obj_text = get_full_text(obj) # Use full text
        obj_text_width = estimate_text_width(obj_text, FONT_SIZE_MOD)

        # Calculate position for the phrase, spreading them out horizontally
        # Center the group of phrases under the anchor
        x_pp_center = x_anchor + (i - (num_phrases - 1) / 2) * PP_X_SPACING

        # Position for preposition (on slanted line)
        y_prep = y_anchor - PP_Y_OFFSET
        x_prep_end = x_pp_center # Prep text goes here

        # Draw slanted line from anchor down to preposition level
        ax.plot([x_anchor, x_prep_end], [y_anchor, y_prep], color='black', linewidth=LINE_WIDTH)

        # Calculate preposition text width for line drawing
        prep_text_width = estimate_text_width(prep_text, FONT_SIZE_MOD)
        prep_line_len = max(prep_text_width * 1.1, 0.1)

        # Draw horizontal line for preposition text (adjust length based on text)
        ax.plot([x_prep_end, x_prep_end + prep_line_len], [y_prep, y_prep], color='black', linewidth=LINE_WIDTH)

        # Add preposition text
        ax.text(x_prep_end + prep_line_len / 2, y_prep + TEXT_Y_OFFSET, prep_text,
                ha='center', va='bottom', fontsize=FONT_SIZE_MOD)

        # Position for object (on horizontal line below preposition)
        x_obj_start = x_prep_end + prep_line_len # Object line starts where prep line ends
        y_obj = y_prep
        obj_line_len = max(obj_text_width * 1.1, 0.15)

        # Draw horizontal line for object
        ax.plot([x_obj_start, x_obj_start + obj_line_len], [y_obj, y_obj], color='black', linewidth=LINE_WIDTH)

        # Add object text
        ax.text(x_obj_start + obj_line_len / 2, y_obj + TEXT_Y_OFFSET, obj_text,
                ha='center', va='bottom', fontsize=FONT_SIZE_MOD)

        # Draw modifiers for the object of the preposition
        obj_mods = get_modifiers(obj)
        if obj_mods:
            # Anchor modifiers to the center of the object text
            draw_modifiers(ax, x_obj_start + obj_line_len / 2, y_obj, obj_mods, obj_text_width)

        # Draw prepositional phrases modifying the object
        obj_pps = get_prep_phrases(obj)
        if obj_pps:
            draw_prep_phrases(ax, x_obj_start + obj_line_len / 2, y_obj, obj_pps)


def draw_reed_kellogg(doc, fig, ax):
    """Main function to draw the diagram for a spaCy doc."""
    # --- Identify Core Components ---
    subjects = [t for t in doc if t.dep_ in ["nsubj", "nsubjpass"]]
    if not subjects:
        st.warning("Could not identify a subject.")
        return None, None # Return None if no subject
    subj = subjects[0] # Assume first subject for simplicity

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root or root.pos_ not in ["VERB", "AUX"]:
        st.warning("Could not identify a main verb.")
        return None, None # Return None if no root verb

    dobj = get_direct_object(root)
    subj_complement = get_subject_complement(root) if not dobj else None # Prefer dobj if both found

    # --- Calculate Positions ---
    # Basic dynamic width calculation (very approximate)
    subj_text = get_full_text(subj)
    verb_text = get_verb_phrase(root)
    subj_width = estimate_text_width(subj_text)
    verb_width = estimate_text_width(verb_text)

    # Padding and spacing
    h_padding = 0.5
    min_segment_len = 0.5
    subj_verb_sep = 0.1 # Gap for the vertical divider
    verb_obj_sep = 0.1 # Gap for vertical/slanted divider

    x_subj_start = h_padding
    x_subj_end = x_subj_start + max(subj_width, min_segment_len)
    x_subj_center = x_subj_start + (x_subj_end - x_subj_start) / 2

    x_verb_start = x_subj_end + subj_verb_sep
    x_verb_end = x_verb_start + max(verb_width, min_segment_len)
    x_verb_center = x_verb_start + (x_verb_end - x_verb_start) / 2

    # Determine position for object/complement if present
    x_obj_start, x_obj_end, x_obj_center = None, None, None
    obj_comp_text = ""
    obj_comp_width = 0
    obj_comp = dobj or subj_complement

    if obj_comp:
        obj_comp_text = get_full_text(obj_comp)
        obj_comp_width = estimate_text_width(obj_comp_text)
        if dobj:
             x_obj_start = x_verb_end + verb_obj_sep
        else: # Complement needs space for slanted line
             slant_h_proj = 0.1 # Horizontal projection of slanted line
             x_obj_start = x_verb_end + slant_h_proj + verb_obj_sep

        x_obj_end = x_obj_start + max(obj_comp_width, min_segment_len)
        x_obj_center = x_obj_start + (x_obj_end - x_obj_start) / 2

    # Total width needed
    total_width = (x_obj_end if obj_comp else x_verb_end) + h_padding

    # --- Draw Baselines and Dividers ---
    # Subject baseline
    ax.plot([x_subj_start, x_subj_end], [Y_BASE, Y_BASE], color='black', linewidth=LINE_WIDTH)
    # Subject-Verb divider (vertical)
    ax.plot([x_subj_end, x_verb_start], [Y_BASE, Y_BASE - 0.05], color='black', linewidth=LINE_WIDTH) # Short vertical line
    # Verb baseline
    ax.plot([x_verb_start, x_verb_end], [Y_BASE, Y_BASE], color='black', linewidth=LINE_WIDTH)

    # Object/Complement divider and baseline
    if dobj:
        # Direct Object: Vertical divider
        ax.plot([x_verb_end, x_obj_start], [Y_BASE, Y_BASE - 0.05], color='black', linewidth=LINE_WIDTH) # Short vertical
        ax.plot([x_obj_start, x_obj_end], [Y_BASE, Y_BASE], color='black', linewidth=LINE_WIDTH) # Object baseline
    elif subj_complement:
        # Subject Complement: Slanted divider '\'
        slant_start_x = x_verb_end
        slant_end_x = x_obj_start - verb_obj_sep # End before object baseline starts
        slant_end_y = Y_BASE - (slant_end_x - slant_start_x) # Make it roughly 45 degrees
        ax.plot([slant_start_x, slant_end_x], [Y_BASE, slant_end_y], color='black', linewidth=LINE_WIDTH)
        ax.plot([x_obj_start, x_obj_end], [Y_BASE, Y_BASE], color='black', linewidth=LINE_WIDTH) # Complement baseline

    # --- Place Text ---
    ax.text(x_subj_center, Y_BASE + TEXT_Y_OFFSET, subj_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    ax.text(x_verb_center, Y_BASE + TEXT_Y_OFFSET, verb_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    if obj_comp:
        ax.text(x_obj_center, Y_BASE + TEXT_Y_OFFSET, obj_comp_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)

    # --- Draw Modifiers and Phrases ---
    # Subject modifiers
    subj_mods = get_modifiers(subj)
    if subj_mods:
        draw_modifiers(ax, x_subj_center, Y_BASE, subj_mods, subj_width)
    subj_pps = get_prep_phrases(subj)
    if subj_pps:
        draw_prep_phrases(ax, x_subj_center, Y_BASE, subj_pps)

    # Verb modifiers (adverbs)
    verb_adverbs = get_adverbial_modifiers(root)
    if verb_adverbs:
        draw_adverbial_modifiers(ax, x_verb_center, Y_BASE, verb_adverbs)
    # Verb prepositional phrases
    verb_pps = get_prep_phrases(root)
    if verb_pps:
        draw_prep_phrases(ax, x_verb_center, Y_BASE, verb_pps)

    # Object/Complement modifiers
    if obj_comp:
        obj_comp_mods = get_modifiers(obj_comp)
        if obj_comp_mods:
            draw_modifiers(ax, x_obj_center, Y_BASE, obj_comp_mods, obj_comp_width)
        obj_comp_pps = get_prep_phrases(obj_comp)
        if obj_comp_pps:
            draw_prep_phrases(ax, x_obj_center, Y_BASE, obj_comp_pps)

    # Return calculated width and fixed height for setting limits
    return total_width, Y_BASE + 0.5 # Return width and estimated required height


# --- Streamlit App Main Function ---
def main():
    nlp = load_nlp_model()

    st.title("Reed-Kellogg Sentence Diagrammer 📊")

    with st.expander("ℹ️ About Reed-Kellogg Diagrams"):
        st.markdown("""
        **Reed-Kellogg sentence diagrams** graphically represent the grammatical structure of sentences.
        - **Baseline:** Holds the core subject, verb, and object/complement.
        - **Dividers:** Vertical line separates subject/verb. Vertical or slanted line separates verb/object or verb/complement.
        - **Modifiers:** Placed on slanted lines below the words they modify (adjectives, adverbs, determiners).
        - **Prepositional Phrases:** Appear on connected slanted/horizontal lines below the word they modify.

        *This app uses spaCy for parsing and attempts to follow basic R-K rules. Complex sentences may not render perfectly.*
        """)

    # Use session state to keep text across reruns triggered by examples
    if 'text' not in st.session_state:
        st.session_state.text = "The quick brown fox jumps over the lazy dog."

    text = st.text_area("Enter sentence:", value=st.session_state.text, height=100, key="sentence_input")
    st.session_state.text = text # Update state if user types

    if st.button("Diagram Sentence", key="diagram_button") or text:
        if text.strip():
            try:
                doc = nlp(text.strip())

                # Basic check for sentence structure
                has_subject = any(t.dep_ in ["nsubj", "nsubjpass"] for t in doc)
                has_root_verb = any(t.dep_ == "ROOT" and t.pos_ in ["VERB", "AUX"] for t in doc)

                if not has_subject or not has_root_verb:
                    st.warning("Please enter a more complete sentence with a clear subject and main verb.")
                else:
                    fig, ax = plt.subplots(figsize=(12, 6)) # Adjust figsize as needed

                    # Draw the diagram
                    diagram_width, diagram_height = draw_reed_kellogg(doc, fig, ax)

                    if diagram_width and diagram_height:
                        # Set dynamic limits with padding
                        ax.set_xlim(0, diagram_width)
                        ax.set_ylim(0, diagram_height) # Use calculated height
                        ax.axis('off')
                        plt.tight_layout(pad=0.5) # Reduce whitespace around figure

                        st.pyplot(fig)

                        # Show parse details
                        with st.expander("View Sentence Analysis Details"):
                            st.subheader("Dependency Parse")
                            cols = st.columns(4)
                            headers = ["Text", "POS", "Dependency", "Head"]
                            for col, header in zip(cols, headers):
                                col.markdown(f"**{header}**")

                            for token in doc:
                                cols = st.columns(4)
                                cols[0].write(token.text)
                                cols[1].write(token.tag_) # More specific POS tag
                                cols[2].write(token.dep_)
                                cols[3].write(f"{token.head.text} ({token.head.i})")
                    else:
                        # If draw_reed_kellogg returned None (due to parsing issues)
                        # Warnings should have already been displayed.
                        pass

            except Exception as e:
                st.error(f"An error occurred while processing the sentence: {str(e)}")
                st.exception(e) # Show full traceback for debugging
                st.info("Try a simpler sentence or check formatting.")
        else:
            st.info("Please enter a sentence to diagram.")

    # --- Example Sentences ---
    st.subheader("Examples")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "My friend gave me a beautiful gift yesterday.", # Might struggle with indirect obj 'me' and adv 'yesterday' placement
        "The students in the classroom studied diligently.",
        "She is very happy.", # Subject complement
        "Running quickly is good exercise.", # Gerund subject - will likely fail
        "The man who arrived late missed the train." # Relative clause - will likely fail
    ]

    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
         # Use unique keys for buttons
        if cols[i].button(f"Try Ex {i+1}", key=f"ex_button_{i}"):
            st.session_state.text = example
            # st.experimental_rerun() # Deprecated
            st.rerun() # Use the new standard rerun function

if __name__ == "__main__":
    main()
