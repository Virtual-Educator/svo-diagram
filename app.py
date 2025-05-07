import streamlit as st
import spacy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

# --- spaCy Model Loading ---
@st.cache_resource
def load_nlp_model():
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)

# --- Diagram Constants ---
DIVIDER_ABOVE = 0.8    # percent of extension above baseline
DIVIDER_BELOW = 0.2    # percent of extension below baseline
MODIFIER_SLANT_OFFSET_X = 0.03
MODIFIER_SLANT_OFFSET_Y = 0.08
LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 12
FONT_SIZE_MOD = 10
Y_BASE_START = 0.8    # top clause baseline
CLAUSE_Y_SPACING = 1.0  # vertical distance between stacked clauses
PP_Y_OFFSET = 0.15
PP_X_SPACING = 0.4
TEXT_Y_OFFSET = 0.02

# --- NLP Helpers ---
def get_modifier_text(token):
    if token.dep_ == 'poss':
        case = next((c for c in token.children if c.dep_ == 'case'), None)
        if case:
            return token.text + case.text
    return token.text


def get_modifiers(token, dep_types=None):
    if dep_types is None:
        dep_types = ["det","amod","compound","nummod","poss","advmod","appos","partmod"]
    return [c for c in sorted(token.children, key=lambda x: x.i) if c.dep_ in dep_types]


def get_full_text(token):
    return token.text


def get_verb_phrase(verb):
    parts = {verb.i: verb.text}
    for c in verb.children:
        if c.dep_ in ['aux','auxpass','neg']:
            parts[c.i] = c.text
    return " ".join(parts[i] for i in sorted(parts))


def get_prep_phrases(token):
    results = []
    for c in token.children:
        if c.dep_ == 'prep':
            obj = next((x for x in c.children if x.dep_ == 'pobj'), None)
            if obj:
                results.append((c,obj))
    return results


def get_direct_object(verb):
    return next((c for c in verb.children if c.dep_ == 'dobj'), None)


def get_indirect_object(verb):
    return next((c for c in verb.children if c.dep_ == 'iobj'), None)


def get_subject_complement(verb):
    return next((c for c in verb.children if c.dep_ in ['attr','acomp']), None)


def get_appositives(noun):
    return [c for c in noun.children if c.dep_ == 'appos']


def get_adverbial_modifiers(verb):
    return [c for c in verb.children if c.dep_ == 'advmod']


# --- Drawing Utilities ---
def estimate_text_width(text, fontsize=FONT_SIZE_MAIN):
    return len(text) * fontsize * 0.0065


def draw_modifiers_recursive(ax, x, y, token):
    subs = [c for c in token.children if c.dep_ == 'advmod']
    if subs:
        draw_modifiers_on_baseline(ax, x, y, subs, True)


def draw_modifiers_on_baseline(ax, x_center, y_base, mods, is_sub=False):
    for i, m in enumerate(mods):
        txt = get_modifier_text(m)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        factor = 0.6 if is_sub else 1.0
        base = x_center + (i - (len(mods)-1)/2) * (w + MODIFIER_SLANT_OFFSET_X) * factor * 0.8
        sl_x = base + MODIFIER_SLANT_OFFSET_X * (i+1) * factor
        sl_y = y_base - MODIFIER_SLANT_OFFSET_Y * (i+1) * factor
        ax.plot([base, sl_x],[y_base, sl_y],'k',lw=LINE_WIDTH)
        ax.text(sl_x, sl_y+TEXT_Y_OFFSET, txt, ha='left', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, sl_x, sl_y, m)


def draw_adverbial_modifiers(ax, x, y_base, advs):
    for i, adv in enumerate(advs):
        txt = get_modifier_text(adv)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        length = max(w*1.1,0.2)
        y = y_base - PP_Y_OFFSET - i*MODIFIER_SLANT_OFFSET_Y*1.5
        ax.plot([x,x],[y_base,y],'k',lw=LINE_WIDTH)
        ax.plot([x-length/2, x+length/2],[y,y],'k',lw=LINE_WIDTH)
        ax.text(x, y+TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x, y, adv)


def draw_prep_phrases(ax, x, y_base, pps):
    for i,(prep,obj) in enumerate(pps):
        txt_p, txt_o = prep.text, get_full_text(obj)
        pw = estimate_text_width(txt_p, FONT_SIZE_MOD)
        ow = estimate_text_width(txt_o, FONT_SIZE_MOD)
        x0 = x + (i - (len(pps)-1)/2)*PP_X_SPACING
        y = y_base - PP_Y_OFFSET
        ax.plot([x, x0],[y_base,y],'k',lw=LINE_WIDTH)
        plen = max(pw*1.1,0.15)
        ax.plot([x0, x0+plen],[y,y],'k',lw=LINE_WIDTH)
        ax.text(x0+plen/2, y+TEXT_Y_OFFSET, txt_p, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        ox0 = x0+plen
        olen = max(ow*1.1,0.2)
        ax.plot([ox0, ox0+olen],[y,y],'k',lw=LINE_WIDTH)
        ax.text(ox0+olen/2, y+TEXT_Y_OFFSET, txt_o, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        # modifiers and nested PPs
        mods = get_modifiers(obj)
        if mods: draw_modifiers_on_baseline(ax, ox0+olen/2, y, mods)
        nested = get_prep_phrases(obj)
        if nested: draw_prep_phrases(ax, ox0+olen/2, y, nested)


def draw_appositives(ax, x_center, y_base, appos_list):
    for app in appos_list:
        txt = get_full_text(app)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        sl_x = x_center + w/2
        sl_y = y_base - PP_Y_OFFSET
        ax.plot([x_center, sl_x],[y_base,sl_y],'k',lw=LINE_WIDTH)
        ax.plot([sl_x, sl_x+w],[sl_y,sl_y],'k',lw=LINE_WIDTH)
        ax.text(sl_x+w/2, sl_y+TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        # modifiers on appositive
        mods = get_modifiers(app)
        if mods: draw_modifiers_on_baseline(ax, sl_x+w/2, sl_y, mods)


def draw_subordinate_clause(ax, vx, vy, clause, y_offset):
    marker = next((c for c in clause.children if c.dep_=='mark'), None)
    stem_x = vx
    stem_y = vy - DIVIDER_BELOW
    end_y = vy - y_offset
    ax.plot([stem_x, stem_x],[stem_y,end_y],'k',lw=LINE_WIDTH, linestyle=(0,(4,2)))
    if marker:
        ax.text(stem_x, stem_y+TEXT_Y_OFFSET, marker.text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
    # draw clause baseline below
    return _draw_single_clause_diagram(ax, clause, end_y)


# --- Core Clause Drawing ---
def _draw_single_clause_diagram(ax, verb, y_base, x_offset=0):
    # find subject
    subj = next((t for t in verb.lefts if t.dep_ in ['nsubj','nsubjpass']), None)
    if verb.dep_=='conj' and not subj:
        head=verb.head
        while head and head.dep_=='conj': head=head.head
        subj = next((t for t in head.lefts if t.dep_ in ['nsubj','nsubjpass']), None)
    if not subj:
        return None, None, None
    # texts and widths
    s_txt = get_full_text(subj)
    v_txt = get_verb_phrase(verb)
    s_w = estimate_text_width(s_txt)
    v_w = estimate_text_width(v_txt)
    # positions
    x_subj_start = 0.3 + x_offset
    x_subj_end = x_subj_start + s_w
    x_div = x_subj_end
    x_verb_start = x_div
    x_verb_end = x_verb_start + v_w
    x_verb_center = (x_verb_start + x_verb_end)/2
    # baseline lines
    ax.plot([x_subj_start,x_subj_end],[y_base,y_base],'k',lw=LINE_WIDTH)
    ax.plot([x_verb_start,x_verb_end],[y_base,y_base],'k',lw=LINE_WIDTH)
    # divider
    ax.plot([x_div,x_div],[y_base-DIVIDER_BELOW,y_base+DIVIDER_ABOVE],'k',lw=LINE_WIDTH)
    # indirect object
    iobj = get_indirect_object(verb)
    if iobj:
        io_txt = get_full_text(iobj)
        io_w = estimate_text_width(io_txt)
        sl_x = x_verb_center - io_w/2
        sl_y = y_base - PP_Y_OFFSET
        ax.plot([x_verb_center, sl_x],[y_base,sl_y],'k',lw=LINE_WIDTH)
        ax.plot([sl_x, sl_x+io_w],[sl_y,sl_y],'k',lw=LINE_WIDTH)
        ax.text(sl_x+io_w/2, sl_y+TEXT_Y_OFFSET, io_txt, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    # direct obj or complement
    dobj = get_direct_object(verb)
    comp = dobj or get_subject_complement(verb)
    x_end = x_verb_end
    if comp:
        c_txt = get_full_text(comp)
        c_w = estimate_text_width(c_txt)
        x_c_start = x_verb_end
        x_c_end = x_c_start + c_w
        ax.plot([x_c_start,x_c_end],[y_base,y_base],'k',lw=LINE_WIDTH)
        # if direct obj, full divider
        if dobj:
            ax.plot([x_verb_end,x_verb_end],[y_base-DIVIDER_BELOW,y_base+DIVIDER_ABOVE],'k',lw=LINE_WIDTH)
        else:
            # slanted for complement
            ax.plot([x_verb_end, x_verb_end+0.15],[y_base,y_base-0.15],'k',lw=LINE_WIDTH)
        ax.text((x_c_start+x_c_end)/2, y_base+TEXT_Y_OFFSET, c_txt, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
        x_end = x_c_end
        # appositives on object
        appos = get_appositives(comp)
        if appos:
            draw_appositives(ax, (x_c_start+x_c_end)/2, y_base, appos)
        # nested modifiers & PPs on comp
        mods_c = get_modifiers(comp)
        if mods_c: draw_modifiers_on_baseline(ax, (x_c_start+x_c_end)/2, y_base, mods_c)
        pps_c = get_prep_phrases(comp)
        if pps_c: draw_prep_phrases(ax, (x_c_start+x_c_end)/2, y_base, pps_c)
    # labels
    ax.text((x_subj_start+x_subj_end)/2, y_base+TEXT_Y_OFFSET, s_txt, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    ax.text(x_verb_center, y_base+TEXT_Y_OFFSET, v_txt, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)
    # subject modifiers & PPs
    subs = get_modifiers(subj)
    if subs: draw_modifiers_on_baseline(ax, (x_subj_start+x_subj_end)/2, y_base, subs)
    pps_s = get_prep_phrases(subj)
    if pps_s: draw_prep_phrases(ax, (x_subj_start+x_subj_end)/2, y_base, pps_s)
    # adverbs & PPs on verb
    advs = get_adverbial_modifiers(verb)
    if advs: draw_adverbial_modifiers(ax, x_verb_center, y_base, advs)
    pps_v = get_prep_phrases(verb)
    if pps_v: draw_prep_phrases(ax, x_verb_center, y_base, pps_v)
    return x_end+0.3+x_offset, x_verb_center, y_base


# --- Main Diagram Function ---
def draw_reed_kellogg(doc, fig, ax):
    # identify main and conjoined verbs
    verbs = []
    conjunctions = []
    for tok in doc:
        if tok.pos_ in ['VERB','AUX']:
            if tok.dep_ == 'ROOT': verbs.append(tok)
            elif tok.dep_ == 'conj' and tok.head.pos_ in ['VERB','AUX']:
                if tok.head not in verbs: verbs.append(tok.head)
                verbs.append(tok)
                cc = next((c for c in tok.children if c.dep_=='cc'), None) or next((l for l in tok.lefts if l.dep_=='cc'), None)
                if cc:
                    v1,v2 = tok.head, tok
                    if v1.i>v2.i: v1,v2=v2,v1
                    conjunctions.append((cc,v1,v2))
    verbs = sorted(set(verbs), key=lambda t: t.i)
    # draw clauses stacked
    y = Y_BASE_START
    coords = {}
    width = 0
    for i,v in enumerate(verbs):
        w, cx, cy = _draw_single_clause_diagram(ax, v, y)
        if w:
            width = max(width, w)
            coords[v.i] = (cx, cy)
        y -= CLAUSE_Y_SPACING
    # draw conjunctions as dashed lines
    style = (0,(4,2))
    for cc,v1,v2 in conjunctions:
        if v1.i in coords and v2.i in coords:
            x1,y1 = coords[v1.i]
            x2,y2 = coords[v2.i]
            mid_y = (y1+y2)/2
            ax.plot([x1,x1],[y1,mid_y], lw=LINE_WIDTH, linestyle=style)
            ax.plot([x2,x2],[y2,mid_y], lw=LINE_WIDTH, linestyle=style)
            ax.plot([x1,x2],[mid_y,mid_y], lw=LINE_WIDTH, linestyle=style)
            ax.text((x1+x2)/2, mid_y+TEXT_Y_OFFSET, cc.text, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
    # adjust canvas
    height = Y_BASE_START - y + 0.5
    return width, height


# --- Streamlit App ---
def main():
    nlp = load_nlp_model()
    st.set_page_config(layout="wide", page_title="Reed-Kellogg Diagrammer")
    st.title("Reed-Kellogg Sentence Diagrammer")
    text = st.text_area("Enter sentence:", value="The parents ate the cake and the children ate the cookies.")
    if st.button("Diagram Sentence") and text.strip():
        doc = nlp(text.strip())
        fig, ax = plt.subplots(figsize=(12, 8))
        w, h = draw_reed_kellogg(doc, fig, ax)
        if w and h:
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.axis('off')
            st.pyplot(fig)
    st.subheader("Examples")
    for ex in [
        "He wrote a letter to his friend.",
        "After dinner, the dog chased the cat that lived next door.",
        "My friend, a talented artist, painted a mural.",
        "They will plan and she will execute the project."
    ]:
        if st.button(ex):
            st.experimental_rerun()

if __name__ == '__main__':
    main()
