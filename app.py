import streamlit as st
# Must be the first Streamlit command
st.set_page_config(layout="wide", page_title="Reed-Kellogg Sentence Diagrammer")

import spacy
import matplotlib.pyplot as plt
import numpy as np
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
DIVIDER_ABOVE = 0.1    # extent above baseline
DIVIDER_BELOW = 0.05   # extent below baseline
MODIFIER_SLANT_OFFSET_X = 0.03
MODIFIER_SLANT_OFFSET_Y = 0.08
LINE_WIDTH = 1.5
FONT_SIZE_MAIN = 12
FONT_SIZE_MOD = 10
Y_BASE_START = 0.8    # top clause baseline
CLAUSE_Y_SPACING = 0.7  # clause vertical spacing
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
    return [c for c in token.children if c.dep_ in dep_types]


def get_full_text(token):
    return token.text


def get_verb_phrase(verb):
    parts = {verb.i: verb.text}
    for c in verb.children:
        if c.dep_ in ['aux','auxpass','neg']:
            parts[c.i] = c.text
    return " ".join(parts[i] for i in sorted(parts))


def get_prep_phrases(token):
    pps = []
    for c in token.children:
        if c.dep_ == 'prep':
            pobj = next((x for x in c.children if x.dep_ == 'pobj'), None)
            if pobj:
                pps.append((c, pobj))
    return pps


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
    subs = get_modifiers(token, ['advmod'])
    if subs:
        draw_modifiers_on_baseline(ax, x, y, subs, is_sub=True)


def draw_modifiers_on_baseline(ax, x_center, y_base, mods, is_sub=False):
    for i, m in enumerate(mods):
        txt = get_modifier_text(m)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        factor = 0.6 if is_sub else 1.0
        base = x_center + (i - (len(mods)-1)/2)*(w+MODIFIER_SLANT_OFFSET_X)*factor*0.8
        sl_x = base + MODIFIER_SLANT_OFFSET_X*(i+1)*factor
        sl_y = y_base - MODIFIER_SLANT_OFFSET_Y*(i+1)*factor
        ax.plot([base, sl_x],[y_base, sl_y],'k',lw=LINE_WIDTH)
        ax.text(sl_x, sl_y+TEXT_Y_OFFSET, txt, ha='left', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, sl_x, sl_y, m)


def draw_adverbial_modifiers(ax, x, y, advs):
    for i, adv in enumerate(advs):
        txt = get_modifier_text(adv)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        length = max(w*1.1, 0.2)
        y_end = y - PP_Y_OFFSET - i*MODIFIER_SLANT_OFFSET_Y*1.5
        ax.plot([x, x],[y, y_end],'k',lw=LINE_WIDTH)
        ax.plot([x-length/2, x+length/2],[y_end, y_end],'k',lw=LINE_WIDTH)
        ax.text(x, y_end+TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        draw_modifiers_recursive(ax, x, y_end, adv)


def draw_prep_phrases(ax, x, y, pps):
    for i,(prep,obj) in enumerate(pps):
        pt, ot = prep.text, get_full_text(obj)
        pw = estimate_text_width(pt, FONT_SIZE_MOD)
        ow = estimate_text_width(ot, FONT_SIZE_MOD)
        x0 = x + (i-(len(pps)-1)/2)*PP_X_SPACING
        y0 = y - PP_Y_OFFSET
        ax.plot([x,x0],[y,y0],'k',lw=LINE_WIDTH)
        plen = max(pw*1.1,0.15)
        ax.plot([x0,x0+plen],[y0,y0],'k',lw=LINE_WIDTH)
        ax.text(x0+plen/2, y0+TEXT_Y_OFFSET, pt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        ox0 = x0+plen
        olen = max(ow*1.1,0.2)
        ax.plot([ox0, ox0+olen],[y0,y0],'k',lw=LINE_WIDTH)
        ax.text(ox0+olen/2, y0+TEXT_Y_OFFSET, ot, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        mods = get_modifiers(obj)
        if mods: draw_modifiers_on_baseline(ax, ox0+olen/2, y0, mods)
        nested = get_prep_phrases(obj)
        if nested: draw_prep_phrases(ax, ox0+olen/2, y0, nested)


def draw_appositives(ax, x, y, apps):
    for app in apps:
        txt = get_full_text(app)
        w = estimate_text_width(txt, FONT_SIZE_MOD)
        slx = x + w/2
        sly = y - PP_Y_OFFSET
        ax.plot([x,slx],[y,sly],'k',lw=LINE_WIDTH)
        ax.plot([slx,slx+w],[sly,sly],'k',lw=LINE_WIDTH)
        ax.text(slx+w/2, sly+TEXT_Y_OFFSET, txt, ha='center', va='bottom', fontsize=FONT_SIZE_MOD)
        mods = get_modifiers(app)
        if mods: draw_modifiers_on_baseline(ax, slx+w/2, sly, mods)

# --- Clause Diagram ---
def _draw_single_clause_diagram(ax, verb, y, x_offset=0):
    # Subject
    subj = next((t for t in verb.lefts if t.dep_ in ['nsubj','nsubjpass']), None)
    if verb.dep_=='conj' and not subj:
        h=verb.head
        while h.dep_=='conj': h=h.head
        subj=next((t for t in h.lefts if t.dep_ in ['nsubj','nsubjpass']),None)
    if not subj: return None,None,None
    # texts
    s= get_full_text(subj); v= get_verb_phrase(verb)
    sw=estimate_text_width(s); vw=estimate_text_width(v)
    # positions
    xs0=0.3+x_offset; xs1=xs0+sw; xdiv=xs1
    xv0=xdiv; xv1=xv0+vw; xvc=(xv0+xv1)/2
    # baselines
    ax.plot([xs0,xs1],[y,y],'k',lw=LINE_WIDTH)
    ax.plot([xv0,xv1],[y,y],'k',lw=LINE_WIDTH)
    # subject-verb divider
    ax.plot([xdiv,xdiv],[y-DIVIDER_BELOW*1.2,y+DIVIDER_ABOVE],'k',lw=LINE_WIDTH)
    # indirect object
    iobj=get_indirect_object(verb)
    if iobj:
        io=get_full_text(iobj); iow=estimate_text_width(io)
        slx= xvc-iow/2; sly=y-PP_Y_OFFSET
        ax.plot([xvc,slx],[y,sly],'k',lw=LINE_WIDTH)
        ax.plot([slx,slx+iow],[sly,sly],'k',lw=LINE_WIDTH)
        ax.text(slx+iow/2,sly+TEXT_Y_OFFSET,io,ha='center',va='bottom',fontsize=FONT_SIZE_MAIN)
    # object or complement
    dobj=get_direct_object(verb); comp=dobj or get_subject_complement(verb)
    xe=xv1
    if comp:
        ct=get_full_text(comp); cw=estimate_text_width(ct)
        xc0=xv1; xc1=xc0+cw
        ax.plot([xc0,xc1],[y,y],'k',lw=LINE_WIDTH)
        # verb-object divider up only
        ax.plot([xv1,xv1],[y,y+DIVIDER_ABOVE],'k',lw=LINE_WIDTH)
        ax.text((xc0+xc1)/2,y+TEXT_Y_OFFSET,ct,ha='center',va='bottom',fontsize=FONT_SIZE_MAIN)
        xe=xc1
    # labels
    ax.text((xs0+xs1)/2,y+TEXT_Y_OFFSET,s,ha='center',va='bottom',fontsize=FONT_SIZE_MAIN)
    ax.text(xvc,y+TEXT_Y_OFFSET,v,ha='center',va='bottom',fontsize=FONT_SIZE_MAIN)
    # modifiers & pps for subj, comp, verb
    mods=get_modifiers(subj)
    if mods: draw_modifiers_on_baseline(ax,(xs0+xs1)/2,y,mods)
    pps=get_prep_phrases(subj)
    if pps: draw_prep_phrases(ax,(xs0+xs1)/2,y,pps)
    advs=get_adverbial_modifiers(verb)
    if advs: draw_adverbial_modifiers(ax,xvc,y,advs)
    ppsv=get_prep_phrases(verb)
    if ppsv: draw_prep_phrases(ax,xvc,y,ppsv)
    return xe+0.3+x_offset, xvc, y

# --- Full Diagram ---
def draw_reed_kellogg(doc,fig,ax):
    verbs=[]; conj=[]
    for t in doc:
        if t.pos_ in ['VERB','AUX']:
            if t.dep_=='ROOT': verbs.append(t)
            elif t.dep_=='conj' and t.head.pos_ in ['VERB','AUX']:
                if t.head not in verbs: verbs.append(t.head)
                verbs.append(t)
                cc=next((c for c in t.children if c.dep_=='cc'),None)
                if cc: conj.append((cc,t.head,t))
    verbs=sorted(set(verbs),key=lambda x:x.i)
    y=Y_BASE_START; coords={}; mx=0
    for v in verbs:
        w,x,yc=_draw_single_clause_diagram(ax,v,y)
        if w: mx=max(mx,w); coords[v.i]=(x,yc)
        y-=CLAUSE_Y_SPACING
    dash=(0,(4,2))
    for cc,v1,v2 in conj:
        if v1.i in coords and v2.i in coords:
            x1,y1=coords[v1.i]; x2,y2=coords[v2.i]
            my=(y1+y2)/2
            ax.plot([x1,x1],[y1,my],'k',lw=LINE_WIDTH,linestyle=dash)
            ax.plot([x2,x2],[y2,my],'k',lw=LINE_WIDTH,linestyle=dash)
            ax.plot([x1,x2],[my,my],'k',lw=LINE_WIDTH,linestyle=dash)
            ax.text((x1+x2)/2,my+TEXT_Y_OFFSET,cc.text,ha='center',va='bottom',fontsize=FONT_SIZE_MOD)
    return mx,Y_BASE_START-y+0.5

# --- Streamlit App ---
def main():
    nlp=load_nlp_model()
    st.title("Reed-Kellogg Sentence Diagrammer")
    text=st.text_area("Enter sentence:",value="The parents ate the cake and the children ate the cookies.")
    if st.button("Diagram Sentence") and text.strip():
        doc=nlp(text.strip())
        fig,ax=plt.subplots(figsize=(12,8))
        w,h=draw_reed_kellogg(doc,fig,ax)
        pad=w*0.1; vpad=h*0.1
        ax.set_xlim(-pad,w+pad); ax.set_ylim(-vpad,h+vpad)
        ax.axis('off'); st.pyplot(fig)
    st.subheader("Examples")
    for ex in [
        "He wrote a letter to his friend.",
        "After dinner, the dog chased the cat that lived next door.",
        "My friend, a talented artist, painted a mural.",
        "They ate and she danced."
    ]:
        if st.button(ex):
            st.session_state['sentence_input']=ex; st.experimental_rerun()

if __name__=='__main__': main()
