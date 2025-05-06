import streamlit as st
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

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
def get_modifier_text(modifier_token):
    """Gets the display text for a modifier token, handling possessives."""
    if modifier_token.dep_ == 'poss':
        # Find the "'s" or "'" (case token)
        case_marker = next((child for child in modifier_token.children if child.dep_ == 'case'), None)
        if case_marker:
            return modifier_token.text + case_marker.text
        # Handle cases like "parents'" where ' is child of parents
        if modifier_token.text.endswith("'"): # Heuristic for already formed possessives like "parents'"
             return modifier_token.text
        # If no explicit case marker found as child, but token itself might be like "its"
        # For "John's", 'John' is poss, 's is case. get_modifier_text(John) should return "John's"
        # This part might need more robust handling based on spaCy's specific parsing of various possessives
    return modifier_token.text

def get_modifiers(token, dep_types=None):
    """Gets specified types of modifier tokens for a given token."""
    if dep_types is None:
        dep_types = ["det", "amod", "compound", "nummod", "poss", "advmod"]
    # Filter out 'case' children of 'poss' modifiers here, as get_modifier_text will handle it.
    return [child for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in dep_types and not (child.dep_ == 'case' and token.dep_ == 'poss')]


def get_full_text(token):
    """Gets the primary text of a token meant for the baseline (subject, object, complement head)."""
    # For compound words that are part of the main element (not drawn as separate modifiers below)
    # e.g., "high school" as a subject, if "high" is 'compound' of 'school'
    # This logic can be complex. For simplicity, often the headword is on the baseline,
    # and "compound" parts are drawn as modifiers.
    # Let's keep it simple: baseline text is the token's text. Compounds are modifiers.
    return token.text


def get_verb_phrase(verb_token):
    """Constructs the verb phrase including auxiliaries and negation, ordered correctly."""
    verb_parts = {}
    for child in verb_token.children:
        if child.dep_ in ['aux', 'auxpass', 'neg']:
            verb_parts[child.i] = child.text
    verb_parts[verb_token.i] = verb_token.text

    return " ".join(verb_parts[i] for i in sorted(verb_parts.keys()))

def get_prep_phrases(token):
    preps = []
    for child in token.children:
        if child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj:
                preps.append((child, pobj))
    return preps

def get_subject_complement(verb_token):
    return next((child for child in verb_token.children if child.dep_ in ["attr", "acomp"]), None)

def get_direct_object(verb_token):
    return next((child for child in verb_token.children if child.dep_ == "dobj"), None)

def get_adverbial_modifiers_of_verb(verb_token):
     return [child for child in verb_token.children if child.dep_ == "advmod"]

# --- Drawing Functions using Matplotlib ---
MODIFIER_SLANT_OFFSET_X = 0.03
MODIFIER_SLANT_OFFSET_Y = 0.08
LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 12
FONT_SIZE_MOD = 10
Y_BASE_START = 0.7 # Starting Y_BASE for the first (or only) clause
CLAUSE_Y_SPACING = 0.5 # Vertical space between compound clauses
PP_Y_OFFSET = 0.15
PP_X_SPACING = 0.4
TEXT_Y_OFFSET = 0.02
DIVIDER_EXTENSION = 0.04 # How much the vertical dividers extend above/below baseline

def estimate_text_width(text, fontsize=FONT_SIZE_MAIN):
    return len(text) * fontsize * 0.0065 # Adjusted factor

def draw_modifiers_recursive(ax, x_anchor, y_anchor, mod_token):
    """Helper to draw modifiers of a modifier."""
    sub_mods = get_modifiers(mod_token, dep_types=["advmod"]) # e.g., adverbs modifying adjectives
    if sub_mods:
        # Slightly adjust anchor for sub-modifiers to attach to the text of the current modifier
        # This is a simplification; precise placement would require knowing mod_text width
        sub_anchor_x = x_anchor - estimate_text_width(get_modifier_text(mod_token), FONT_SIZE_MOD) / 2
        draw_modifiers_on_baseline(ax, sub_anchor_x, y_anchor, sub_mods, is_sub_modifier=True)

def draw_modifiers_on_baseline(ax, x_baseline_center, y_baseline, modifiers, is_sub_modifier=False):
    """Draws modifiers for a word on the main baseline."""
    num_mods = len(modifiers)
    for i, mod_token in enumerate(modifiers):
        mod_text = get_modifier_text(mod_token)
        mod_text_width = estimate_text_width(mod_text, FONT_SIZE_MOD)

        # Adjust slant for sub-modifiers to make them less spread out
        slant_factor = 0.6 if is_sub_modifier else 1.0

        # Position modifier below the baseline
        # Stagger left and right from center for multiple modifiers
        # Horizontal positioning: try to spread them a bit under the word
        horizontal_offset_step = (mod_text_width + MODIFIER_SLANT_OFFSET_X) * slant_factor * 0.8
        base_x_offset = x_baseline_center + (i - (num_mods - 1) / 2) * horizontal_offset_step * 0.5
        
        sl_y = y_baseline - (PP_Y_OFFSET / 2) - (i % 2) * MODIFIER_SLANT_OFFSET_Y * 0.7 # slight y stagger for readability
        sl_x = base_x_offset - MODIFIER_SLANT_OFFSET_X * (i + 1) * slant_factor # Slant away

        ax.plot([base_x_offset, sl_x], [y_baseline, sl_y], color='black', linewidth=LINE_WIDTH)
        ax.text(sl_x, sl_y + TEXT_Y_OFFSET, mod_text, ha='right', va='bottom', fontsize=FONT_SIZE_MOD)

        draw_modifiers_recursive(ax, sl_x, sl_y, mod_token) # Draw modifiers of this modifier

def draw_adverbial_modifiers(ax, x_verb_center, y_baseline, adverbs):
    for i, adv_token in enumerate(adverbs):
        adv_text = get_modifier_text(adv_token) # Adverbs are modifiers
        text_width = estimate_text_width(adv_text, FONT_SIZE_MOD)
        line_len = max(text_width * 1.1, 0.2)

        y_pos = y_baseline - PP_Y_OFFSET - i * MODIFIER_SLANT_OFFSET_Y * 1.5
        # Center the adverb line under the verb for simplicity
        x_adv_center = x_verb_center
        x_adv_line_start = x_adv_center - line_len / 2
        
        # Slanted connector from verb baseline to adverb line
        ax.plot([x_verb_center, x_adv_center], [y_baseline, y_pos], color='black', linewidth=LINE_WIDTH)
        ax.plot([x_adv_line_start, x_adv_line_start + line_len], [y_pos, y_pos], color='black', linewidth=LINE_WIDTH)
        ax.text(x_adv_center, y_pos + TEXT_Y_OFFSET, adv_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x_adv_center, y_pos, adv_token)

def draw_prep_phrases(ax, x_anchor, y_baseline, prep_phrases):
    num_phrases = len(prep_phrases)
    for i, (prep_token, obj_token) in enumerate(prep_phrases):
        prep_text = prep_token.text
        obj_display_text = get_full_text(obj_token) # Object on its own line
        obj_text_width_est = estimate_text_width(obj_display_text, FONT_SIZE_MOD)

        x_pp_group_center = x_anchor + (i - (num_phrases - 1) / 2) * PP_X_SPACING
        y_prep = y_baseline - PP_Y_OFFSET
        
        # Slanted line to preposition
        ax.plot([x_anchor, x_pp_group_center], [y_baseline, y_prep], color='black', linewidth=LINE_WIDTH)

        prep_text_width_est = estimate_text_width(prep_text, FONT_SIZE_MOD)
        prep_line_len = max(prep_text_width_est * 1.1, 0.15)
        ax.plot([x_pp_group_center, x_pp_group_center + prep_line_len], [y_prep, y_prep], color='black', linewidth=LINE_WIDTH)
        ax.text(x_pp_group_center + prep_line_len / 2, y_prep + TEXT_Y_OFFSET, prep_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)

        x_obj_line_start = x_pp_group_center + prep_line_len
        obj_line_len = max(obj_text_width_est * 1.1, 0.2)
        ax.plot([x_obj_line_start, x_obj_line_start + obj_line_len], [y_prep, y_prep], color='black', linewidth=LINE_WIDTH)
        x_obj_center = x_obj_line_start + obj_line_len / 2
        ax.text(x_obj_center, y_prep + TEXT_Y_OFFSET, obj_display_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)

        obj_mods = get_modifiers(obj_token)
        if obj_mods:
            draw_modifiers_on_baseline(ax, x_obj_center, y_prep, obj_mods)
        
        obj_pps = get_prep_phrases(obj_token) # Nested PPs
        if obj_pps:
            draw_prep_phrases(ax, x_obj_center, y_prep, obj_pps)


def _draw_single_clause_diagram(ax, clause_verb, current_y_base, current_x_offset=0):
    """Draws the diagram for a single clause."""
    doc = clause_verb.doc # Get the full document from any token

    # Find subject of this specific clause verb
    subj = next((t for t in clause_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
    if not subj: # If verb is part of a conjoined verb phrase, subject might be attached to the first verb
        head_verb = clause_verb
        while head_verb.dep_ == 'conj' and head_verb.head.pos_ == 'VERB':
            head_verb = head_verb.head
            s = next((t for t in head_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
            if s:
                subj = s
                break
    if not subj: return None, current_y_base # Cannot draw clause without subject

    dobj = get_direct_object(clause_verb)
    subj_complement = get_subject_complement(clause_verb) if not dobj else None

    subj_text = get_full_text(subj)
    verb_text = get_verb_phrase(clause_verb)
    subj_width = estimate_text_width(subj_text)
    verb_width = estimate_text_width(verb_text)

    h_padding = 0.3 + current_x_offset # Start further right if offset
    segment_min_len = 0.4
    text_sep = 0.05 # Minimal space between text and divider line

    x_subj_start = h_padding
    x_subj_text_center = x_subj_start + subj_width / 2
    x_subj_baseline_end = x_subj_start + subj_width

    subj_verb_divider_x = x_subj_baseline_end + text_sep

    x_verb_baseline_start = subj_verb_divider_x + text_sep
    x_verb_text_center = x_verb_baseline_start + verb_width / 2
    x_verb_baseline_end = x_verb_baseline_start + verb_width

    obj_comp = dobj or subj_complement
    x_obj_comp_text_center = None
    final_x_pos = x_verb_baseline_end

    if obj_comp:
        obj_comp_text = get_full_text(obj_comp)
        obj_comp_width_est = estimate_text_width(obj_comp_text)
        
        verb_obj_divider_x = x_verb_baseline_end + text_sep
        x_obj_comp_baseline_start = verb_obj_divider_x + text_sep
        x_obj_comp_text_center = x_obj_comp_baseline_start + obj_comp_width_est / 2
        x_obj_comp_baseline_end = x_obj_comp_baseline_start + obj_comp_width_est
        final_x_pos = x_obj_comp_baseline_end

        # Subject-Verb Baseline
        ax.plot([x_subj_start, x_subj_baseline_end], [current_y_base, current_y_base], color='black', linewidth=LINE_WIDTH)
        # Verb Baseline
        ax.plot([x_verb_baseline_start, x_verb_baseline_end], [current_y_base, current_y_base], color='black', linewidth=LINE_WIDTH)
        # Object/Complement Baseline
        ax.plot([x_obj_comp_baseline_start, x_obj_comp_baseline_end], [current_y_base, current_y_base], color='black', linewidth=LINE_WIDTH)

        # Subject-Verb Divider (Full Vertical)
        ax.plot([subj_verb_divider_x, subj_verb_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], color='black', linewidth=LINE_WIDTH)

        if dobj: # Verb-Object Divider (Full Vertical)
            ax.plot([verb_obj_divider_x, verb_obj_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], color='black', linewidth=LINE_WIDTH)
        elif subj_complement: # Verb-Complement Divider (Slanted)
            ax.plot([verb_obj_divider_x, verb_obj_divider_x + 0.1], [current_y_base, current_y_base - 0.1], color='black', linewidth=LINE_WIDTH) # Simple slant

        ax.text(x_obj_comp_text_center, current_y_base + TEXT_Y_OFFSET, obj_comp_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
        obj_comp_mods = get_modifiers(obj_comp)
        if obj_comp_mods: draw_modifiers_on_baseline(ax, x_obj_comp_text_center, current_y_base, obj_comp_mods)
        obj_comp_pps = get_prep_phrases(obj_comp)
        if obj_comp_pps: draw_prep_phrases(ax, x_obj_comp_text_center, current_y_base, obj_comp_pps)

    else: # No object or complement
        # Subject-Verb Baseline
        ax.plot([x_subj_start, x_subj_baseline_end], [current_y_base, current_y_base], color='black', linewidth=LINE_WIDTH)
        # Verb Baseline (no direct object, so verb line extends to where verb text ends)
        ax.plot([x_verb_baseline_start, x_verb_baseline_end], [current_y_base, current_y_base], color='black', linewidth=LINE_WIDTH)
        # Subject-Verb Divider (Full Vertical)
        ax.plot([subj_verb_divider_x, subj_verb_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], color='black', linewidth=LINE_WIDTH)


    ax.text(x_subj_text_center, current_y_base + TEXT_Y_OFFSET, subj_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    ax.text(x_verb_text_center, current_y_base + TEXT_Y_OFFSET, verb_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)

    subj_mods = get_modifiers(subj)
    if subj_mods: draw_modifiers_on_baseline(ax, x_subj_text_center, current_y_base, subj_mods)
    subj_pps = get_prep_phrases(subj)
    if subj_pps: draw_prep_phrases(ax, x_subj_text_center, current_y_base, subj_pps)

    verb_adverbs = get_adverbial_modifiers_of_verb(clause_verb)
    if verb_adverbs: draw_adverbial_modifiers(ax, x_verb_text_center, current_y_base, verb_adverbs)
    verb_pps = get_prep_phrases(clause_verb)
    if verb_pps: draw_prep_phrases(ax, x_verb_text_center, current_y_base, verb_pps)
    
    # Return the x-coordinate of the verb for potential conjunction connection
    return final_x_pos + h_padding, x_verb_text_center, current_y_base


def draw_reed_kellogg(doc, fig, ax):
    """Main function to draw the diagram. Handles simple and attempts compound sentences."""
    
    # Identify potential main verbs of clauses
    clause_verbs = []
    conjunctions = [] # To store (cc_token, verb1_x, verb1_y, verb2_x, verb2_y)

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            clause_verbs.append(token)
        # Find coordinating conjunctions linking verbs (simplified detection)
        if token.dep_ == "cc" and token.head.pos_ in ("VERB", "AUX") and token.head.dep_ == "conj":
            # token is 'and', token.head is verb2, token.head.head is verb1
            verb1 = token.head.head
            verb2 = token.head
            if verb1 not in clause_verbs: clause_verbs.append(verb1) # Ensure verb1 is included if not ROOT
            if verb2 not in clause_verbs: clause_verbs.append(verb2)
            # For now, just note the conjunction. Drawing it requires knowing clause positions.
            conjunctions.append({'cc': token, 'v1': verb1, 'v2': verb2})


    if not clause_verbs: # If only one root and no conjunctions found above
         root_verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")), None)
         if root_verb:
            clause_verbs.append(root_verb)
         else:
            st.warning("Could not identify a main verb for any clause.")
            return None, None
    
    # Remove duplicates and sort by sentence order
    clause_verbs = sorted(list(set(clause_verbs)), key=lambda v: v.i)

    max_width = 0
    total_height_needed = Y_BASE_START + 0.5 # Initial height for one clause + padding
    
    drawn_clause_info = [] # To store {'verb_x': x, 'verb_y': y} for each clause verb

    # For now, this loop will effectively draw the first clause only,
    # as _draw_single_clause_diagram isn't fully equipped for y_offsets from loop yet
    # To handle multiple clauses, current_y_base would need to be updated in each iteration.
    
    current_y = Y_BASE_START
    if not clause_verbs:
        st.warning("No clauses identified to diagram.")
        return None, None

    # --- THIS IS WHERE MULTI-CLAUSE LOGIC WOULD EXPAND ---
    # For now, we'll process and draw the first identified clause verb's structure
    # In a full compound implementation, you'd loop through clause_verbs:
    # for i, cv in enumerate(clause_verbs):
    #     y_for_this_clause = Y_BASE_START + i * CLAUSE_Y_SPACING
    #     width, verb_x, verb_y = _draw_single_clause_diagram(ax, cv, y_for_this_clause)
    #     if width: max_width = max(max_width, width)
    #     drawn_clause_info.append({'verb_token': cv, 'x': verb_x, 'y': verb_y})
    #     total_height_needed = y_for_this_clause + 0.5
    # Then draw conjunctions using info in drawn_clause_info and conjunctions list.
    # --- END OF MULTI-CLAUSE CONCEPT ---

    # Simplified: Draw first clause
    cv_to_draw = clause_verbs[0]
    width, verb_x, verb_y = _draw_single_clause_diagram(ax, cv_to_draw, current_y)
    if width is None: # Diagramming failed for the clause
        return None, None
    max_width = max(max_width, width)
    drawn_clause_info.append({'verb_token': cv_to_draw, 'x': verb_x, 'y': verb_y, 'cc_token': None})


    # If there were conjunctions identified, find the one related to the drawn clauses
    # This part is highly simplified and only considers first conjunction for first two clauses.
    if conjunctions and len(clause_verbs) > 1:
        first_conj = conjunctions[0]
        # Try to find the info for the verbs involved in the first conjunction
        v1_info = next((info for info in drawn_clause_info if info['verb_token'] == first_conj['v1']), None)
        
        # Placeholder for drawing the second clause if we were to implement it fully here
        # For now, imagine the second clause *would have been drawn* at current_y + CLAUSE_Y_SPACING
        # And we'd need its verb_x position.
        # This is where drawing the stepped conjunction line would happen.
        # Example: ax.plot([v1_info['x'], v1_info['x']], [v1_info['y'], v1_info['y'] - 0.1], 'k:') # Step down
        #          ax.plot([v1_info['x'], v2_x_placeholder], [v1_info['y'] - 0.1, v1_info['y'] - 0.1], 'k:') # Horizontal
        #          ax.text((v1_info['x'] + v2_x_placeholder)/2, v1_info['y'] - 0.1 + TEXT_Y_OFFSET, first_conj['cc'].text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        #          ax.plot([v2_x_placeholder, v2_x_placeholder], [v1_info['y'] - 0.1, v2_y_placeholder], 'k:') # Step up/down to verb2
        pass # End of conceptual conjunction drawing


    if max_width == 0:
         st.warning("Failed to generate diagram components.")
         return None, None

    return max_width, total_height_needed


# --- Streamlit App Main Function ---
def main():
    nlp = load_nlp_model()
    st.title("Reed-Kellogg Sentence Diagrammer 📊")

    with st.expander("ℹ️ About Reed-Kellogg Diagrams"):
        st.markdown("""
        **Reed-Kellogg sentence diagrams** graphically represent sentence structure.
        - **Baseline:** Holds subject, verb, object/complement.
        - **Dividers:** Full vertical lines separate Subject|Verb and Verb|Object. Slanted line for Verb\\Complement.
        - **Modifiers:** On slanted lines below words they modify.
        - **Prepositional Phrases:** Below words, on slanted lines to preposition, then horizontal for object.
        *Compound sentences (multiple clauses) are complex. This version focuses on single clauses correctly.*
        """)

    if 'text' not in st.session_state:
        st.session_state.text = "The parents ate the cake." # Simpler default for now

    text = st.text_area("Enter sentence:", value=st.session_state.text, height=100, key="sentence_input")
    st.session_state.text = text

    if st.button("Diagram Sentence", key="diagram_button") or text:
        if text.strip():
            try:
                doc = nlp(text.strip())
                has_subject = any(t.dep_ in ["nsubj", "nsubjpass"] for t in doc)
                has_root_verb = any(t.dep_ == "ROOT" and t.pos_ in ["VERB", "AUX"] for t in doc)

                if not (has_subject and has_root_verb):
                    # Allow if conjunctions are present, indicating potential multiple clauses
                    is_compound_candidate = any(t.dep_ == 'cc' for t in doc)
                    if not is_compound_candidate:
                        st.warning("Please enter a complete sentence with a clear subject and main verb.")
                        return # Stop processing if not a clear simple sentence or compound candidate
                
                fig, ax = plt.subplots(figsize=(12, 6))
                diagram_width, diagram_height = draw_reed_kellogg(doc, fig, ax)

                if diagram_width and diagram_height:
                    ax.set_xlim(0, diagram_width)
                    ax.set_ylim(0, diagram_height)
                    ax.axis('off')
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig)

                    with st.expander("View Sentence Analysis Details"):
                        st.subheader("Dependency Parse")
                        cols_headers = st.columns(4)
                        headers = ["Text", "POS (fine)", "Dependency", "Head (Text, Index)"]
                        for col, header in zip(cols_headers, headers):
                            col.markdown(f"**{header}**")
                        for token in doc:
                            cols_data = st.columns(4)
                            cols_data[0].write(token.text)
                            cols_data[1].write(token.tag_)
                            cols_data[2].write(token.dep_)
                            cols_data[3].write(f"{token.head.text} ({token.head.i})")
                # else: (Warnings should be handled within draw_reed_kellogg or _draw_single_clause_diagram)
                #    pass
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        else:
            st.info("Please enter a sentence.")

    st.subheader("Examples")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "My friend's parents gave me a beautiful gift yesterday.",
        "The students in the classroom studied diligently.",
        "She is very happy.",
        "The parents ate the cake and the children ate the cookies." # Compound example
    ]
    cols = st.columns(min(len(examples), 5)) # Max 5 columns for examples
    for i, example in enumerate(examples):
        if cols[i % 5].button(f"Ex {i+1}", key=f"ex_btn_{i}", help=example):
            st.session_state.text = example
            st.rerun()

if __name__ == "__main__":
    main()
