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
        case_marker = next((child for child in modifier_token.children if child.dep_ == 'case'), None)
        if case_marker:
            return modifier_token.text + case_marker.text
        if modifier_token.text.endswith("'"):
             return modifier_token.text
    return modifier_token.text

def get_modifiers(token, dep_types=None):
    """Gets specified types of modifier tokens for a given token."""
    if dep_types is None:
        dep_types = ["det", "amod", "compound", "nummod", "poss", "advmod"]
    return [child for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in dep_types and not (child.dep_ == 'case' and token.dep_ == 'poss')]

def get_full_text(token):
    """Gets the primary text of a token meant for the baseline."""
    return token.text

def get_verb_phrase(verb_token):
    """Constructs the verb phrase including auxiliaries and negation, ordered correctly."""
    verb_parts = {verb_token.i: verb_token.text}
    for child in verb_token.children:
        if child.dep_ in ['aux', 'auxpass', 'neg']:
            verb_parts[child.i] = child.text
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
Y_BASE_START = 0.8 # Start higher to accommodate multiple clauses stacked downwards
CLAUSE_Y_SPACING = 0.7 # Increased vertical space between compound clauses
PP_Y_OFFSET = 0.15
PP_X_SPACING = 0.4
TEXT_Y_OFFSET = 0.02
DIVIDER_EXTENSION = 0.04
CONJUNCTION_STEP_Y = 0.1 # How much the conjunction line steps down/up
CONJUNCTION_H_MARGIN = 0.05 # Horizontal margin for conjunction text on its line

def estimate_text_width(text, fontsize=FONT_SIZE_MAIN):
    return len(text) * fontsize * 0.0065

def draw_modifiers_recursive(ax, x_anchor, y_anchor, mod_token):
    sub_mods = get_modifiers(mod_token, dep_types=["advmod"])
    if sub_mods:
        sub_anchor_x = x_anchor - estimate_text_width(get_modifier_text(mod_token), FONT_SIZE_MOD) / 2
        draw_modifiers_on_baseline(ax, sub_anchor_x, y_anchor, sub_mods, is_sub_modifier=True)

def draw_modifiers_on_baseline(ax, x_baseline_center, y_baseline, modifiers, is_sub_modifier=False):
    num_mods = len(modifiers)
    for i, mod_token in enumerate(modifiers):
        mod_text = get_modifier_text(mod_token)
        mod_text_width = estimate_text_width(mod_text, FONT_SIZE_MOD)
        slant_factor = 0.6 if is_sub_modifier else 1.0
        horizontal_offset_step = (mod_text_width + MODIFIER_SLANT_OFFSET_X) * slant_factor * 0.8
        base_x_offset = x_baseline_center + (i - (num_mods - 1) / 2) * horizontal_offset_step * 0.5
        sl_y = y_baseline - (PP_Y_OFFSET / 2) - (i % 2) * MODIFIER_SLANT_OFFSET_Y * 0.7
        sl_x = base_x_offset - MODIFIER_SLANT_OFFSET_X * (i + 1) * slant_factor
        ax.plot([base_x_offset, sl_x], [y_baseline, sl_y], color='black', linewidth=LINE_WIDTH)
        ax.text(sl_x, sl_y + TEXT_Y_OFFSET, mod_text, ha='right', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, sl_x, sl_y, mod_token)

def draw_adverbial_modifiers(ax, x_verb_center, y_baseline, adverbs):
    for i, adv_token in enumerate(adverbs):
        adv_text = get_modifier_text(adv_token)
        text_width = estimate_text_width(adv_text, FONT_SIZE_MOD)
        line_len = max(text_width * 1.1, 0.2)
        y_pos = y_baseline - PP_Y_OFFSET - i * MODIFIER_SLANT_OFFSET_Y * 1.5
        x_adv_center = x_verb_center
        x_adv_line_start = x_adv_center - line_len / 2
        ax.plot([x_verb_center, x_adv_center], [y_baseline, y_pos], color='black', linewidth=LINE_WIDTH)
        ax.plot([x_adv_line_start, x_adv_line_start + line_len], [y_pos, y_pos], color='black', linewidth=LINE_WIDTH)
        ax.text(x_adv_center, y_pos + TEXT_Y_OFFSET, adv_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x_adv_center, y_pos, adv_token)

def draw_prep_phrases(ax, x_anchor, y_baseline, prep_phrases):
    num_phrases = len(prep_phrases)
    for i, (prep_token, obj_token) in enumerate(prep_phrases):
        prep_text = prep_token.text
        obj_display_text = get_full_text(obj_token)
        obj_text_width_est = estimate_text_width(obj_display_text, FONT_SIZE_MOD)
        x_pp_group_center = x_anchor + (i - (num_phrases - 1) / 2) * PP_X_SPACING
        y_prep = y_baseline - PP_Y_OFFSET
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
        if obj_mods: draw_modifiers_on_baseline(ax, x_obj_center, y_prep, obj_mods)
        obj_pps = get_prep_phrases(obj_token)
        if obj_pps: draw_prep_phrases(ax, x_obj_center, y_prep, obj_pps)

def _draw_single_clause_diagram(ax, clause_verb, current_y_base, current_x_offset=0):
    subj = next((t for t in clause_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
    if not subj:
        head_verb = clause_verb
        while head_verb.dep_ == 'conj' and head_verb.head.pos_ == 'VERB':
            head_verb = head_verb.head
            s = next((t for t in head_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
            if s: subj = s; break
    if not subj:
        # Try finding subject connected to the ROOT if this is a conjoined verb from ROOT
        if clause_verb.head.dep_ == "ROOT" and clause_verb.dep_ == "conj":
            subj = next((t for t in clause_verb.doc if t.dep_ in ("nsubj", "nsubjpass") and t.head == clause_verb.head), None)
    if not subj:
        # Fallback: if this clause_verb is ROOT, find its subject
        if clause_verb.dep_ == "ROOT":
            subj = next((t for t in clause_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)

    if not subj: return None, None, None, current_y_base

    dobj = get_direct_object(clause_verb)
    subj_complement = get_subject_complement(clause_verb) if not dobj else None
    subj_text = get_full_text(subj); verb_text = get_verb_phrase(clause_verb)
    subj_width = estimate_text_width(subj_text); verb_width = estimate_text_width(verb_text)
    h_padding = 0.3 + current_x_offset; text_sep = 0.05
    x_subj_start = h_padding
    x_subj_baseline_end = x_subj_start + subj_width
    x_subj_text_center = x_subj_start + subj_width / 2
    subj_verb_divider_x = x_subj_baseline_end + text_sep
    x_verb_baseline_start = subj_verb_divider_x + text_sep
    x_verb_baseline_end = x_verb_baseline_start + verb_width
    x_verb_text_center = x_verb_baseline_start + verb_width / 2
    obj_comp = dobj or subj_complement
    final_x_pos = x_verb_baseline_end
    ax.plot([x_subj_start, x_subj_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
    ax.plot([x_verb_baseline_start, x_verb_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
    ax.plot([subj_verb_divider_x, subj_verb_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], 'k', lw=LINE_WIDTH)
    if obj_comp:
        obj_comp_text = get_full_text(obj_comp)
        obj_comp_width_est = estimate_text_width(obj_comp_text)
        verb_obj_divider_x = x_verb_baseline_end + text_sep
        x_obj_comp_baseline_start = verb_obj_divider_x + text_sep
        x_obj_comp_baseline_end = x_obj_comp_baseline_start + obj_comp_width_est
        x_obj_comp_text_center = x_obj_comp_baseline_start + obj_comp_width_est / 2
        final_x_pos = x_obj_comp_baseline_end
        ax.plot([x_obj_comp_baseline_start, x_obj_comp_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
        if dobj:
            ax.plot([verb_obj_divider_x, verb_obj_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], 'k', lw=LINE_WIDTH)
        elif subj_complement:
            ax.plot([verb_obj_divider_x, verb_obj_divider_x + 0.15], [current_y_base, current_y_base - 0.15], 'k', lw=LINE_WIDTH) # Adjusted slant
        ax.text(x_obj_comp_text_center, current_y_base + TEXT_Y_OFFSET, obj_comp_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
        obj_comp_mods = get_modifiers(obj_comp)
        if obj_comp_mods: draw_modifiers_on_baseline(ax, x_obj_comp_text_center, current_y_base, obj_comp_mods)
        obj_comp_pps = get_prep_phrases(obj_comp)
        if obj_comp_pps: draw_prep_phrases(ax, x_obj_comp_text_center, current_y_base, obj_comp_pps)
    ax.text(x_subj_text_center, current_y_base + TEXT_Y_OFFSET, subj_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    ax.text(x_verb_text_center, current_y_base + TEXT_Y_OFFSET, verb_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    subj_mods = get_modifiers(subj);
    if subj_mods: draw_modifiers_on_baseline(ax, x_subj_text_center, current_y_base, subj_mods)
    subj_pps = get_prep_phrases(subj);
    if subj_pps: draw_prep_phrases(ax, x_subj_text_center, current_y_base, subj_pps)
    verb_adverbs = get_adverbial_modifiers_of_verb(clause_verb)
    if verb_adverbs: draw_adverbial_modifiers(ax, x_verb_text_center, current_y_base, verb_adverbs)
    verb_pps = get_prep_phrases(clause_verb)
    if verb_pps: draw_prep_phrases(ax, x_verb_text_center, current_y_base, verb_pps)
    return final_x_pos + h_padding, x_verb_text_center, current_y_base

def draw_reed_kellogg(doc, fig, ax):
    clause_verbs = []
    identified_conjunctions = []
    root_token = next((t for t in doc if t.dep_ == "ROOT"), doc[0] if len(doc)>0 else None)

    # Find all verbs that could start a clause (ROOT or conjoined to another main verb)
    # This needs to be robust for various compound structures.
    
    # Attempt 1: Find ROOT and its direct conjunctions
    if root_token and root_token.pos_ in ("VERB", "AUX"):
        clause_verbs.append(root_token)
        for child in root_token.children:
            if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
                clause_verbs.append(child)
                # Find the cc associated with this conjunction
                cc = next((c for c in child.children if c.dep_ == "cc"), 
                          next((c for c in child.lefts if c.dep_ == "cc" and c.i < child.i and c.i > root_token.i), None)) # CC between verbs
                if cc:
                    identified_conjunctions.append({'cc_token': cc, 'verb1': root_token, 'verb2': child})
    
    # Attempt 2: If no conjoined verbs from ROOT, look for clauses conjoined higher up if ROOT isn't a verb.
    # Or if sentence starts with a conjunction.
    if not clause_verbs or (len(clause_verbs) == 1 and not identified_conjunctions):
        clause_verbs = [] # Reset
        potential_clause_starts = [token for token in doc if token.pos_ in ("VERB", "AUX") and (token.dep_ == "ROOT" or token.dep_ == "conj")]
        for verb_idx, verb in enumerate(potential_clause_starts):
            is_already_part_of_conj = any(conj['verb2'] == verb or conj['verb1'] == verb for conj in identified_conjunctions)
            if not is_already_part_of_conj:
                 if verb not in clause_verbs: clause_verbs.append(verb)
            if verb.dep_ == "conj":
                cc = next((c for c in verb.children if c.dep_ == "cc"), 
                          next((c for c in verb.lefts if c.dep_ == "cc" and c.i < verb.i and c.head == verb), None))
                if cc and verb.head.pos_ in ("VERB", "AUX"): # Ensure cc connects to a verb
                    # Check if this conjunction pair is already added
                    is_new_conj = not any(conj['cc_token'] == cc for conj in identified_conjunctions)
                    if is_new_conj:
                        v1 = verb.head
                        v2 = verb
                        if v1 not in clause_verbs : clause_verbs.append(v1)
                        if v2 not in clause_verbs : clause_verbs.append(v2)
                        identified_conjunctions.append({'cc_token': cc, 'verb1': v1, 'verb2': v2})


    if not clause_verbs:
        st.warning("Could not identify main verb(s) for diagramming.")
        return None, None
    
    # Ensure unique verbs, sorted by appearance in sentence
    seen_verb_indices = set()
    unique_clause_verbs = []
    for v in sorted(clause_verbs, key=lambda x: x.i):
        if v.i not in seen_verb_indices:
            unique_clause_verbs.append(v)
            seen_verb_indices.add(v.i)
    clause_verbs = unique_clause_verbs
    
    max_total_width = 0
    current_y_base_for_clause = Y_BASE_START
    drawn_clauses_verb_coords = {} # Store verb_x, verb_y by verb token index

    if not clause_verbs:
        st.warning("No clauses identified to diagram.")
        return None, None

    for i, verb_token in enumerate(clause_verbs):
        clause_width, verb_x, verb_y = _draw_single_clause_diagram(ax, verb_token, current_y_base_for_clause)
        if clause_width is None: # Problem drawing this clause
            st.warning(f"Could not diagram clause starting with verb '{verb_token.text}'. Skipping.")
            continue
        max_total_width = max(max_total_width, clause_width)
        drawn_clauses_verb_coords[verb_token.i] = (verb_x, verb_y)
        if i < len(clause_verbs) - 1: # If there are more clauses
            current_y_base_for_clause -= CLAUSE_Y_SPACING # Move down for next clause

    # Draw conjunction lines
    for conj_info in identified_conjunctions:
        cc_token = conj_info['cc_token']
        v1_idx = conj_info['verb1'].i
        v2_idx = conj_info['verb2'].i

        if v1_idx in drawn_clauses_verb_coords and v2_idx in drawn_clauses_verb_coords:
            v1_x, v1_y = drawn_clauses_verb_coords[v1_idx]
            v2_x, v2_y = drawn_clauses_verb_coords[v2_idx]

            # Midpoints for the horizontal line of the conjunction
            # Ensure horizontal line is roughly between the two verb x-coordinates
            # And at a y-level between the two clauses
            conj_line_y = (v1_y + v2_y) / 2
            if abs(v1_y - v2_y) < CLAUSE_Y_SPACING * 0.5 : # If clauses are too close, adjust conj_line_y
                conj_line_y = v1_y - CLAUSE_Y_SPACING / 2.5


            # Determine step direction and horizontal line points
            # Line starts from verb1, steps to conj_line_y, goes horizontal, then steps to verb2
            
            # Horizontal extent of the conjunction line
            # Make it shorter, e.g., 1/3 of the distance between verb centers, or a fixed small length
            h_line_start_x = min(v1_x, v2_x) + abs(v1_x - v2_x) * 0.3
            h_line_end_x = max(v1_x, v2_x) - abs(v1_x - v2_x) * 0.3
            # if verbs are vertically aligned, shift the horizontal line slightly
            if abs(v1_x - v2_x) < 0.2: # small threshold
                avg_x = (v1_x + v2_x) /2
                h_line_start_x = avg_x - 0.15
                h_line_end_x = avg_x + 0.15


            # Vertical lines (steps)
            ax.plot([v1_x, v1_x], [v1_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH) # Step from v1
            ax.plot([v2_x, v2_x], [v2_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH) # Step from v2
            
            # Horizontal line for conjunction text
            ax.plot([h_line_start_x, h_line_end_x], [conj_line_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH)
            
            cc_text = cc_token.text
            ax.text((h_line_start_x + h_line_end_x) / 2, conj_line_y + TEXT_Y_OFFSET, cc_text,
                    ha='center', va='bottom', fontsize=FONT_SIZE_MOD, backgroundcolor='white', zorder=3)


    final_diagram_height = Y_BASE_START + 0.5 if not clause_verbs else Y_BASE_START - current_y_base_for_clause + CLAUSE_Y_SPACING + 0.2
    if max_total_width == 0: return None, None
    return max_total_width, final_diagram_height


# --- Streamlit App Main Function --- (NO CHANGES BELOW THIS LINE from previous version)
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
        *Compound sentences are now supported with stacked clauses and conjunction lines.*
        """)

    if 'text' not in st.session_state:
        st.session_state.text = "The parents ate the cake and the children ate the cookies."

    text = st.text_area("Enter sentence:", value=st.session_state.text, height=100, key="sentence_input")
    st.session_state.text = text

    if st.button("Diagram Sentence", key="diagram_button") or text: # Process if button clicked OR text exists
        if text.strip():
            try:
                doc = nlp(text.strip())
                # No need for basic subject/verb check here anymore as draw_reed_kellogg handles clause identification
                
                fig, ax = plt.subplots(figsize=(12, 8)) # Increased default height for compound
                diagram_width, diagram_height = draw_reed_kellogg(doc, fig, ax)

                if diagram_width and diagram_height:
                    ax.set_xlim(0, diagram_width)
                    ax.set_ylim(0, diagram_height) # Use calculated height
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
                else:
                     # This case might occur if no valid clauses are found at all.
                     # Warnings should ideally be shown from within draw_reed_kellogg or _draw_single_clause_diagram
                     pass
            except Exception as e:
                st.error(f"An error occurred during diagram generation: {str(e)}")
                st.exception(e) # Provides full traceback for debugging
        else:
            st.info("Please enter a sentence to diagram.")

    st.subheader("Examples")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "My friend's parents gave me a beautiful gift yesterday.",
        "The students in the classroom studied diligently.",
        "She is very happy.",
        "The parents ate the cake and the children ate the cookies.",
        "He ran and she jumped.",
        "Mary designs and John codes."
    ]
    # Dynamically create columns for examples, up to a max of 4 per row for better layout
    num_examples = len(examples)
    max_cols = 4
    num_rows = (num_examples + max_cols - 1) // max_cols

    for r in range(num_rows):
        cols = st.columns(max_cols)
        for c in range(max_cols):
            idx = r * max_cols + c
            if idx < num_examples:
                example = examples[idx]
                if cols[c].button(f"Ex {idx+1}", key=f"ex_btn_{idx}", help=example):
                    st.session_state.text = example
                    st.rerun()

if __name__ == "__main__":
    main()
