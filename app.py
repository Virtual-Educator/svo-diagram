import streamlit as st
# Must be the first Streamlit command in your script
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

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
DIVIDER_ABOVE = 0.1    # how far divider extends above baseline
DIVIDER_BELOW = 0.05   # how far divider extends below baseline
MODIFIER_SLANT_OFFSET_X = 0.03
MODIFIER_SLANT_OFFSET_Y = 0.08
LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 12
FONT_SIZE_MOD = 10
Y_BASE_START = 0.8    # y-coordinate for top clause
CLAUSE_Y_SPACING = 0.7  # vertical spacing between clauses
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
                results.append((c, obj))
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
        draw_modifiers_on_baseline(ax, x, y, subs, is_sub=True)


def draw_modifiers_on_baseline(ax, x_center, y_base, mods, is_sub=False):
    for i, m in enumerate(mods):
        txt = get_modifier_text(m)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        factor = 0.6 if is_sub else 1.0
        base = x_center + (i - (len(mods)-1)/2) * (w + MODIFIER_SLANT_OFFSET_X) * factor * 0.8
        sl_x = base + MODIFIER_SLANT_OFFSET_X * (i+1) * factor
        sl_y = y_base - MODIFIER_SLANT_OFFSET_Y * (i+1) * factor
        ax.plot([base, sl_x], [y_base, sl_y], 'k', lw=LINE_WIDTH)
        ax.text(sl_x, sl_y + TEXT_Y_OFFSET, txt, ha='left', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, sl_x, sl_y, m)


def draw_adverbial_modifiers(ax, x, y_base, advs):
    for i, adv in enumerate(advs):
        txt = get_modifier_text(adv)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        length = max(w * 1.1, 0.2)
        y = y_base - PP_Y_OFFSET - i * MODIFIER_SLANT_OFFSET_Y * 1.5
        ax.plot([x, x], [y_base, y], 'k', lw=LINE_WIDTH)
        ax.plot([x - length/2, x + length/2], [y, y], 'k', lw=LINE_WIDTH)
        ax.text(x, y + TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x, y, adv)


def draw_prep_phrases(ax, x, y_base, pps):
    for i, (prep, obj) in enumerate(pps):
        pt, ot = prep.text, get_full_text(obj)
        pw = estimate_text_width(pt, FONT_SIZE_MOD)
        ow = estimate_text_width(ot, FONT_SIZE_MOD)
        x0 = x + (i - (len(pps)-1)/2) * PP_X_SPACING
        y = y_base - PP_Y_OFFSET
        ax.plot([x, x0], [y_base, y], 'k', lw=LINE_WIDTH)
        plen = max(pw * 1.1, 0.15)
        ax.plot([x0, x0 + plen], [y, y], 'k', lw=LINE_WIDTH)
        ax.text(x0 + plen/2, y + TEXT_Y_OFFSET, pt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        ox0 = x0 + plen
        olen = max(ow * 1.1, 0.2)
        ax.plot([ox0, ox0 + olen], [y, y], 'k', lw=LINE_WIDTH)
        ax.text(ox0 + olen/2, y + TEXT_Y_OFFSET, ot, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        mods = get_modifiers(obj)
        if mods: draw_modifiers_on_baseline(ax, ox0 + olen/2, y, mods)
        nested = get_prep_phrases(obj)
        if nested: draw_prep_phrases(ax, ox0 + olen/2, y, nested)


def draw_appositives(ax, x_center, y_base, appos_list):
    for app in appos_list:
        txt = get_full_text(app)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        sl_x = x_center + w/2
        sl_y = y_base - PP_Y_OFFSET
        ax.plot([x_center, sl_x], [y_base, sl_y], 'k', lw=LINE_WIDTH)
        ax.plot([sl_x, sl_x + w], [sl_y, sl_y], 'k', lw=LINE_WIDTH)
        ax.text(sl_x + w/2, sl_y + TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        mods = get_modifiers(app)
        if mods: draw_modifiers_on_baseline(ax, sl_x + w/2, sl_y, mods)

# --- Core Clause Drawing ---
def _draw_single_clause_diagram(ax, verb, y_base, x_offset=0):
    subj = next((t for t in verb.lefts if t.dep_ in ['nsubj','nsubjpass']), None)
    if verb.dep_ == 'conj' and not subj:
        h = verb.head
        while h and h.dep_ == 'conj': h = h.head
        subj = next((t for t in h.lefts if t.dep_ in ['nsubj','nsubjpass']), None)
    if not subj:
        return None, None, None

    s_txt = get_full_text(subj)
    v_txt = get_verb_phrase(verb)
    s_w = estimate_text_width(s_txt)
    v_w = estimate_text_width(v_txt)

    x_s0 = 0.3 + x_offset
    x_s1 = x_s0 + s_w
    x_div_sv = x_s1
    x_v0 = x_div_sv
    x_v1 = x_v0 + v_w
    x_vc = (x_v0 + x_v1) / 2

    ax.plot([x_s0, x_s1], [y_base, y_base], 'k', lw=LINE_WIDTH)
    ax.plot([x_v0, x_v1], [y_base, y_base], 'k', lw=LINE_WIDTH)
    ax.plot([x_div_sv, x_div_sv], [y_base - DIVIDER_BELOW, y_base + DIVIDER_ABOVE], 'k', lw=LINE_WIDTH)

    iobj = get_indirect_object(verb)
    if iobj:
        io_txt = get_full_text(iobj)
        io_w = estimate_text_width(io_txt)
        sl_x0 = x_vc - io_w/2
        sl_y0 = y_base - PP_Y_OFFSET
        ax.plot([x_vc, sl_x0], [y_base, sl_y0], 'k', lw=LINE_WIDTH)
        ax.plot([sl_x0, sl_x0 + io_w], [sl_y0, sl_y0], 'k', lw=LINE_WIDTH)
        ax.text(sl_x0 + io_w/2, sl_y0 + TEXT_Y_OFFSET, io_txt, ha='center', va='bottom', fontsize=FONT_SIZE_MAIN)

    dobj = get_direct_object(verb)
    comp = dobj or get_subject_complement(verb)
    x_end = x_v1
    if comp:
        c_txt = get_full_text(comp)
        c_w = estimate_text
