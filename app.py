import streamlit as st
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

# --- spaCy Model Loading --- (Same as before)
@st.cache_resource
def load_nlp_model():
    model_name = "en_core_web_sm"
    try: return spacy.load(model_name)
    except OSError:
        st.info(f"Downloading spaCy model: {model_name}..."); from spacy.cli import download
        download(model_name); return spacy.load(model_name)

# --- Helper Functions for NLP Analysis --- (Largely same, minor tweaks if needed)
def get_modifier_text(modifier_token):
    if modifier_token.dep_ == 'poss':
        case_marker = next((child for child in modifier_token.children if child.dep_ == 'case'), None)
        if case_marker: return modifier_token.text + case_marker.text
        if modifier_token.text.endswith("'"): return modifier_token.text
    return modifier_token.text

def get_modifiers(token, dep_types=None):
    if dep_types is None: dep_types = ["det", "amod", "compound", "nummod", "poss", "advmod"]
    return [child for child in sorted(token.children, key=lambda c: c.i)
            if child.dep_ in dep_types and not (child.dep_ == 'case' and token.dep_ == 'poss')]

def get_full_text(token): return token.text

def get_verb_phrase(verb_token):
    verb_parts = {verb_token.i: verb_token.text}
    for child in verb_token.children:
        if child.dep_ in ['aux', 'auxpass', 'neg']: verb_parts[child.i] = child.text
    return " ".join(verb_parts[i] for i in sorted(verb_parts.keys()))

def get_prep_phrases(token):
    preps = []
    for child in token.children:
        if child.dep_ == "prep":
            pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
            if pobj: preps.append((child, pobj))
    return preps

def get_subject_complement(verb_token):
    return next((child for child in verb_token.children if child.dep_ in ["attr", "acomp"]), None)

def get_direct_object(verb_token):
    return next((child for child in verb_token.children if child.dep_ == "dobj"), None)

def get_adverbial_modifiers_of_verb(verb_token):
     return [child for child in verb_token.children if child.dep_ == "advmod"]

# --- Drawing Functions using Matplotlib --- (Largely same)
MODIFIER_SLANT_OFFSET_X = 0.03; MODIFIER_SLANT_OFFSET_Y = 0.08; LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 12; FONT_SIZE_MOD = 10; Y_BASE_START = 0.8
CLAUSE_Y_SPACING = 0.7; PP_Y_OFFSET = 0.15; PP_X_SPACING = 0.4
TEXT_Y_OFFSET = 0.02; DIVIDER_EXTENSION = 0.04; CONJUNCTION_H_MARGIN = 0.05

def estimate_text_width(text, fontsize=FONT_SIZE_MAIN): return len(text) * fontsize * 0.0065

def draw_modifiers_recursive(ax, x_anchor, y_anchor, mod_token):
    sub_mods = get_modifiers(mod_token, dep_types=["advmod"])
    if sub_mods:
        sub_anchor_x = x_anchor - estimate_text_width(get_modifier_text(mod_token), FONT_SIZE_MOD) / 2
        draw_modifiers_on_baseline(ax, sub_anchor_x, y_anchor, sub_mods, is_sub_modifier=True)

def draw_modifiers_on_baseline(ax, x_baseline_center, y_baseline, modifiers, is_sub_modifier=False):
    num_mods = len(modifiers)
    for i, mod_token in enumerate(modifiers):
        mod_text = get_modifier_text(mod_token); mod_text_width = estimate_text_width(mod_text, FONT_SIZE_MOD)
        slant_factor = 0.6 if is_sub_modifier else 1.0
        horizontal_offset_step = (mod_text_width + MODIFIER_SLANT_OFFSET_X) * slant_factor * 0.8
        base_x_offset = x_baseline_center + (i - (num_mods - 1) / 2) * horizontal_offset_step * 0.5
        sl_y = y_baseline - (PP_Y_OFFSET / 2) - (i % 2) * MODIFIER_SLANT_OFFSET_Y * 0.7
        sl_x = base_x_offset - MODIFIER_SLANT_OFFSET_X * (i + 1) * slant_factor
        ax.plot([base_x_offset, sl_x], [y_baseline, sl_y], 'k', lw=LINE_WIDTH)
        ax.text(sl_x, sl_y + TEXT_Y_OFFSET, mod_text, ha='right', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, sl_x, sl_y, mod_token)

def draw_adverbial_modifiers(ax, x_verb_center, y_baseline, adverbs):
    for i, adv_token in enumerate(adverbs):
        adv_text = get_modifier_text(adv_token); text_width = estimate_text_width(adv_text, FONT_SIZE_MOD)
        line_len = max(text_width * 1.1, 0.2); y_pos = y_baseline - PP_Y_OFFSET - i * MODIFIER_SLANT_OFFSET_Y * 1.5
        x_adv_center = x_verb_center; x_adv_line_start = x_adv_center - line_len / 2
        ax.plot([x_verb_center, x_adv_center], [y_baseline, y_pos], 'k', lw=LINE_WIDTH)
        ax.plot([x_adv_line_start, x_adv_line_start + line_len], [y_pos, y_pos], 'k', lw=LINE_WIDTH)
        ax.text(x_adv_center, y_pos + TEXT_Y_OFFSET, adv_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x_adv_center, y_pos, adv_token)

def draw_prep_phrases(ax, x_anchor, y_baseline, prep_phrases):
    num_phrases = len(prep_phrases)
    for i, (prep_token, obj_token) in enumerate(prep_phrases):
        prep_text = prep_token.text; obj_display_text = get_full_text(obj_token)
        obj_text_width_est = estimate_text_width(obj_display_text, FONT_SIZE_MOD)
        x_pp_group_center = x_anchor + (i - (num_phrases - 1) / 2) * PP_X_SPACING
        y_prep = y_baseline - PP_Y_OFFSET
        ax.plot([x_anchor, x_pp_group_center], [y_baseline, y_prep], 'k', lw=LINE_WIDTH)
        prep_text_width_est = estimate_text_width(prep_text, FONT_SIZE_MOD)
        prep_line_len = max(prep_text_width_est * 1.1, 0.15)
        ax.plot([x_pp_group_center, x_pp_group_center + prep_line_len], [y_prep, y_prep], 'k', lw=LINE_WIDTH)
        ax.text(x_pp_group_center + prep_line_len / 2, y_prep + TEXT_Y_OFFSET, prep_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        x_obj_line_start = x_pp_group_center + prep_line_len
        obj_line_len = max(obj_text_width_est * 1.1, 0.2)
        ax.plot([x_obj_line_start, x_obj_line_start + obj_line_len], [y_prep, y_prep], 'k', lw=LINE_WIDTH)
        x_obj_center = x_obj_line_start + obj_line_len / 2
        ax.text(x_obj_center, y_prep + TEXT_Y_OFFSET, obj_display_text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        obj_mods = get_modifiers(obj_token)
        if obj_mods: draw_modifiers_on_baseline(ax, x_obj_center, y_prep, obj_mods)
        obj_pps = get_prep_phrases(obj_token)
        if obj_pps: draw_prep_phrases(ax, x_obj_center, y_prep, obj_pps)

def _draw_single_clause_diagram(ax, clause_verb, current_y_base, current_x_offset=0):
    # Enhanced Subject Finding Logic
    subj = None
    if clause_verb.dep_ in ("ROOT", "conj"): # Only try to find subjects for main/conjoined verbs
        # Direct subject
        subj = next((t for t in clause_verb.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
        
        # If conjoined verb, check its heads for a subject
        if not subj and clause_verb.dep_ == "conj":
            head_v = clause_verb.head
            while head_v.dep_ == "conj" and head_v.pos_ in ("VERB", "AUX"): # Traverse up conj chain
                subj_candidate = next((t for t in head_v.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
                if subj_candidate: subj = subj_candidate; break
                head_v = head_v.head
            if not subj and head_v.dep_ == "ROOT": # Check final head if it's ROOT
                 subj_candidate = next((t for t in head_v.lefts if t.dep_ in ("nsubj", "nsubjpass")), None)
                 if subj_candidate: subj = subj_candidate

    if not subj: return None, None, None # Cannot draw clause without subject

    dobj = get_direct_object(clause_verb); subj_complement = get_subject_complement(clause_verb) if not dobj else None
    subj_text = get_full_text(subj); verb_text = get_verb_phrase(clause_verb)
    subj_width = estimate_text_width(subj_text); verb_width = estimate_text_width(verb_text)
    h_padding = 0.3 + current_x_offset; text_sep = 0.05
    x_subj_start = h_padding; x_subj_baseline_end = x_subj_start + subj_width
    x_subj_text_center = x_subj_start + subj_width / 2
    subj_verb_divider_x = x_subj_baseline_end + text_sep
    x_verb_baseline_start = subj_verb_divider_x + text_sep
    x_verb_baseline_end = x_verb_baseline_start + verb_width
    x_verb_text_center = x_verb_baseline_start + verb_width / 2
    obj_comp = dobj or subj_complement; final_x_pos = x_verb_baseline_end
    ax.plot([x_subj_start, x_subj_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
    ax.plot([x_verb_baseline_start, x_verb_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
    ax.plot([subj_verb_divider_x, subj_verb_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], 'k', lw=LINE_WIDTH)
    if obj_comp:
        obj_comp_text = get_full_text(obj_comp); obj_comp_width_est = estimate_text_width(obj_comp_text)
        verb_obj_divider_x = x_verb_baseline_end + text_sep
        x_obj_comp_baseline_start = verb_obj_divider_x + text_sep
        x_obj_comp_baseline_end = x_obj_comp_baseline_start + obj_comp_width_est
        x_obj_comp_text_center = x_obj_comp_baseline_start + obj_comp_width_est / 2
        final_x_pos = x_obj_comp_baseline_end
        ax.plot([x_obj_comp_baseline_start, x_obj_comp_baseline_end], [current_y_base, current_y_base], 'k', lw=LINE_WIDTH)
        if dobj: ax.plot([verb_obj_divider_x, verb_obj_divider_x], [current_y_base - DIVIDER_EXTENSION, current_y_base + DIVIDER_EXTENSION], 'k', lw=LINE_WIDTH)
        elif subj_complement: ax.plot([verb_obj_divider_x, verb_obj_divider_x + 0.15], [current_y_base, current_y_base - 0.15], 'k', lw=LINE_WIDTH)
        ax.text(x_obj_comp_text_center, current_y_base + TEXT_Y_OFFSET, obj_comp_text, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
        obj_comp_mods = get_modifiers(obj_comp);
        if obj_comp_mods: draw_modifiers_on_baseline(ax, x_obj_comp_text_center, current_y_base, obj_comp_mods)
        obj_comp_pps = get_prep_phrases(obj_comp);
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
    return final_x_pos + h_padding, x_verb_text_center, current_y_base # Return width, verb_x_center, verb_y_baseline


def draw_reed_kellogg(doc, fig, ax):
    clause_verbs = []
    identified_conjunctions = []

    # Identify verbs that are ROOT or conjoined to another verb (potential clause heads)
    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            if token.dep_ == "ROOT":
                if token not in clause_verbs: clause_verbs.append(token)
            elif token.dep_ == "conj" and token.head.pos_ in ("VERB", "AUX"):
                # This is a conjoined verb. Ensure its head is also considered.
                if token.head not in clause_verbs: clause_verbs.append(token.head)
                if token not in clause_verbs: clause_verbs.append(token)
                
                # Try to find the cc (coordinating conjunction)
                # Typically, the 'cc' token's head is the second verb in the conjoined pair (token itself)
                cc_token = next((child for child in token.children if child.dep_ == "cc"), None)
                if not cc_token: # Sometimes it's a left sibling of the conjoined verb
                     cc_token = next((left for left in token.lefts if left.dep_ == "cc" and left.head == token),None)

                if cc_token:
                    verb1 = token.head # The verb it's conjoined to
                    verb2 = token      # This verb
                    # Ensure verb1 appears before verb2 in the sentence
                    if verb1.i > verb2.i: verb1, verb2 = verb2, verb1 
                    
                    # Avoid duplicate conjunctions
                    is_new_conj = not any(c['cc_token'].i == cc_token.i and c['verb1'].i == verb1.i and c['verb2'].i == verb2.i for c in identified_conjunctions)
                    if is_new_conj:
                        identified_conjunctions.append({'cc_token': cc_token, 'verb1': verb1, 'verb2': verb2})
    
    # If only one verb was found and it's ROOT, it's a simple sentence
    if not clause_verbs:
        root_verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")), None)
        if root_verb: clause_verbs.append(root_verb)
        else: st.warning("Could not identify any main verb(s)."); return None, None

    # Deduplicate and sort clause_verbs by sentence order
    clause_verbs = sorted(list(set(clause_verbs)), key=lambda v: v.i)
    
    if not clause_verbs: st.warning("No clauses identified for diagramming."); return None, None

    max_total_width = 0
    current_y_base_for_clause = Y_BASE_START
    drawn_clauses_verb_coords = {} # Store verb_x_center, verb_y_baseline by verb token index

    for i, verb_token in enumerate(clause_verbs):
        # Add a small x_offset for subsequent clauses if desired for visual staggering (optional)
        # current_x_offset = i * 0.1 
        clause_width, verb_x_center, verb_y_baseline = _draw_single_clause_diagram(ax, verb_token, current_y_base_for_clause)
        
        if clause_width is None:
            st.warning(f"Could not diagram clause for verb '{verb_token.text}' (index {verb_token.i}). Skipping.")
            continue
            
        max_total_width = max(max_total_width, clause_width)
        drawn_clauses_verb_coords[verb_token.i] = (verb_x_center, verb_y_baseline)
        
        if i < len(clause_verbs) - 1:
            current_y_base_for_clause -= CLAUSE_Y_SPACING

    # Draw conjunction lines
    if identified_conjunctions:
        st.write(f"Debug: Identified Conjunctions: {[{'cc':c['cc_token'].text, 'v1':c['verb1'].text, 'v2':c['verb2'].text} for c in identified_conjunctions]}")
        st.write(f"Debug: Drawn Clause Verb Coords: { {k:v for k,v in drawn_clauses_verb_coords.items()} }")

    for conj_info in identified_conjunctions:
        cc_token = conj_info['cc_token']
        verb1 = conj_info['verb1']
        verb2 = conj_info['verb2']

        if verb1.i in drawn_clauses_verb_coords and verb2.i in drawn_clauses_verb_coords:
            v1_x, v1_y = drawn_clauses_verb_coords[verb1.i]
            v2_x, v2_y = drawn_clauses_verb_coords[verb2.i]

            # Y-coordinate for the horizontal part of the conjunction line
            conj_line_y = (v1_y + v2_y) / 2.0
            # If clauses are too close or y-values are inverted from expectation, adjust
            if abs(v1_y - v2_y) < CLAUSE_Y_SPACING * 0.4: # If baselines are close
                 conj_line_y = min(v1_y, v2_y) - CLAUSE_Y_SPACING * 0.3 # Place it below the lower clause's midpoint to ensure separation


            # 1. Vertical line from verb1 (at v1_x) to conj_line_y
            ax.plot([v1_x, v1_x], [v1_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH)
            # 2. Vertical line from verb2 (at v2_x) to conj_line_y
            ax.plot([v2_x, v2_x], [v2_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH)
            # 3. Horizontal line connecting the two vertical steps at their x-positions
            ax.plot([min(v1_x, v2_x), max(v1_x, v2_x)], [conj_line_y, conj_line_y], color='black', linestyle=':', linewidth=LINE_WIDTH)
            
            cc_text = cc_token.text
            text_x_pos = (v1_x + v2_x) / 2.0 
            ax.text(text_x_pos, conj_line_y + TEXT_Y_OFFSET, cc_text,
                    ha='center', va='bottom', fontsize=FONT_SIZE_MOD,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=0.7), zorder=3) # semi-transparent bbox
        else:
            missing_verbs_msg = []
            if verb1.i not in drawn_clauses_verb_coords: missing_verbs_msg.append(f"{verb1.text} (idx {verb1.i})")
            if verb2.i not in drawn_clauses_verb_coords: missing_verbs_msg.append(f"{verb2.text} (idx {verb2.i})")
            st.warning(f"Skipping conjunction '{cc_token.text}': Coords missing for {', '.join(missing_verbs_msg)}.")


    # Calculate the total height based on the last clause's y position
    # If no clauses drawn, current_y_base_for_clause remains Y_BASE_START
    final_diagram_height = Y_BASE_START + 0.5 # Default for one clause or if loop didn't run
    if clause_verbs and len(drawn_clauses_verb_coords) > 0: # If at least one clause was drawn
        # The last current_y_base_for_clause is the baseline of the LOWEST clause
        # So the total height is from Y_BASE_START down to that, plus some padding below it.
        # Since y decreases, height is Y_BASE_START - (last_y_baseline - some_padding_below)
        min_y_coord_drawn = min(coord[1] for coord in drawn_clauses_verb_coords.values()) if drawn_clauses_verb_coords else current_y_base_for_clause
        final_diagram_height = Y_BASE_START + (Y_BASE_START - min_y_coord_drawn) + 0.5 # Add some padding below the lowest element

    if max_total_width == 0 and not clause_verbs : return None, None # Nothing to draw
    if max_total_width == 0 and clause_verbs : max_total_width = 5.0 # Default width if clauses identified but somehow not drawn with width
    
    return max_total_width, final_diagram_height


# --- Streamlit App Main Function --- (Same as before)
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

    if st.button("Diagram Sentence", key="diagram_button") or text: 
        if text.strip():
            try:
                doc = nlp(text.strip())
                
                fig, ax = plt.subplots(figsize=(12, 8)) 
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
            except Exception as e:
                st.error(f"An error occurred during diagram generation: {str(e)}")
                st.exception(e) 
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
    num_examples = len(examples); max_cols = 4
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
