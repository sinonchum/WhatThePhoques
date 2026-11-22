"""Infer model groups from cached multiple-choice and integer replies.

Compute per-prompt departures from typical answer syntax within each proposed
family, then account for whether explanations precede or follow a commitment.
Endpoint scores above the upper third of the family's distribution become
individual variants. Ties at the boundary remain with the shared checkpoint.
This exploratory assumption was evaluated on nine reference deployments.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np

OPT = re.compile(r"\bA:\s*(.*?);\s*B:\s*(.*?);\s*C:\s*(.*?);\s*D:\s*(.*?)\.\s*Reply", re.I)

FIELDS = ("altered_separator", "terminal_punctuation", "copy_case_change",
          "choice_answer_label", "prose_answer_scaffold", "reprinted_multiple_labels",
          "answer_then_further_text")
OPTS = re.compile(r"\bA: (.*?); B: (.*?); C: (.*?); D: (.*?)\. Reply\b")
LABEL = re.compile(r"^\(?([ABCD])\s*([:.)\-])\s*(.+)", re.S)
SCALAR = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|[A-D])([.!]?)$")


def grammar(row, prompt):
    """Each statistic is a binary event given a well-defined opportunity.

    A long answer does not automatically count against an endpoint. If there
    is no eligible construction, the affected feature is missing entirely.
    """
    text = (row["response"] or "").strip().replace("**", "").strip("`").strip()
    x = np.full(len(FIELDS), np.nan)
    options = OPTS.search(prompt["prompt"])
    letter = LABEL.match(text)
    if options and letter:
        x[0] = float(letter[2] != ":")
        payload = letter[3].strip()
        # Compare to any copied option; nominal choice correctness is irrelevant.
        for option in sorted(options.groups(), key=len, reverse=True):
            if payload[:len(option)].casefold() != option.casefold():
                continue
            tail = payload[len(option):].strip()
            if tail and tail[0].isalnum():
                continue
            if tail in ("", ".", "!"):
                x[1] = float(bool(tail))
            if option and option[0].isalpha() and len(option) > 1:
                x[2] = float(payload[:len(option)] != option)
            break
    short = SCALAR.fullmatch(text)
    if short:
        x[1] = float(bool(short[1]))
    # Prose-only features keep raw length and missingness outside the score.
    if len(text.split()) >= 8:
        if options:
            x[3] = float(bool(re.search(r"\b(?:correct (?:option|letter)|option [ABCD]\b|"
                                        r"(?:answer|option) is [ABCD]\b)", text, re.I)))
        x[4] = float(bool(re.search(r"(?m)^\s*(?:explanation|reasoning|solution|answer|"
                                    r"final answer|step \d+)\s*[:.)]", text, re.I)))
        if options:
            x[5] = float(len(set(re.findall(r"\b([ABCD])\s*[:)]", text))) > 1)
    # Among declarative answer openings, distinguish a final answer from a
    # short answer followed by extra explanation. A long derivation that never
    # commits to an answer near the start supplies no opportunity.
    clauses = re.split(r"(?<!\d)[.!?](?=\s|$)|(?<=\d)\.(?=\s|$)|\n",text,maxsplit=1)
    first = clauses[0]
    if (4 <= len(first.split()) <= 14 and
        re.search(r"\b(?:is|are|equals|prints?|produces?)\s+[`\x22(]*[-+]?\d+\b|"
                  r"\b(?:answer|option) is [ABCD]\b",first,re.I)):
        x[6] = float(len(clauses)>1 and len(clauses[1].split())>=4)
    return x


def score(x, kind, feature_count):
    # x shape is batch × endpoint × prompt × repeat × feature.
    x = x[..., :feature_count]
    events = np.nansum(x,axis=3)
    opps = np.sum(~np.isnan(x),axis=3)
    peer_y = events.sum(axis=1,keepdims=True)-events
    peer_n = opps.sum(axis=1,keepdims=True)-opps
    expect = np.zeros_like(events)
    for k in set(kind):
        jj=np.flatnonzero(kind==k)
        # Four pooled pseudo-observations shrink per-prompt probabilities toward
        # the leave-endpoint-out average for other prompts of that format.
        sy,sn=peer_y[:,:,jj],peer_n[:,:,jj]
        p=(sy.sum(axis=2,keepdims=True)-sy+.5)/(sn.sum(axis=2,keepdims=True)-sn+1)
        expect[:,:,jj]=(sy+4*p)/(sn+4)
    residual=(events-opps*expect).sum(axis=2)
    variance=(opps*expect*(1-expect)).sum(axis=2)
    # Signed score: typed departures from source format in the same direction
    # across questions. A unit noise floor regularizes features with tiny exposure.
    return residual.sum(axis=-1)/np.sqrt(1+variance.sum(axis=-1))


def words(s):
    return re.findall(r"\w+|[^\w\s]",s.casefold())

@lru_cache(None)
def observation(pid,prompt,text):
    options=OPT.search(prompt)
    kind='choice' if options else 'integer'
    row={'id':pid,'prompt':prompt,'response':text}
    f=grammar(row,{'format':kind,'prompt':prompt})
    extra=np.full(4,np.nan)
    text=' '.join((text or '').split())
    if options:
        m=re.match(r'^\(?([ABCD])\)?([.:;)])\s*([^.!?;\n]+)',text,re.I)
        if m:
            opts=[words(s.strip(' .,;:')) for s in options.groups()]
            expected=opts[ord(m[1].upper())-ord('A')]
            rendered=words(m[3])
            prefix=rendered[:len(expected)]
            same=prefix==expected
            overlap=[SequenceMatcher(None,rendered[:len(o)],o,autojunk=False).ratio() for o in opts]
            # Partial paraphrase versus selecting a different supplied option.
            extra[0]=float(not same and .2<max(overlap)<.99)
            extra[1]=float(not same and len(expected)>1 and set(prefix)==set(expected))
            extra[2]=float(len(rendered)>len(expected) and not same)
            extra[3]=float(len(rendered)>0 and len(expected)>0 and rendered[0]!=expected[0] and max(overlap)<.99)
    else:
        # Short initial numerical commitment modified by a unit or a restatement.
        m=re.match(r'^[-+]?\d+(?:\.\d+)?\b(.*)',text)
        if m:
            tail=m[1].strip()
            extra[0]=float(bool(re.match('^[A-Za-z]',tail)))
            extra[1]=float(bool(re.match(r'[.]\s+[A-Za-z]',tail)))
            extra[2]=float(bool(re.search(r'\b(?:because|means|answer|example|therefore)\b',tail,re.I)))
    return np.r_[f,extra]

def scores(rows):
    eps=sorted({r['endpoint'] for r in rows}); qs=sorted({r['id'] for r in rows}); idx={e:i for i,e in enumerate(eps)}
    qi={q:j for j,q in enumerate(qs)}; orig=defaultdict(list); kinds={}
    for r in rows:
        orig[r['endpoint'],r['id']].append(observation(r['id'],r['prompt'],r['response']))
        kinds[r['id']]='choice' if OPT.search(r['prompt']) else 'integer'
    rep=max(map(len,orig.values()))
    S=np.full((1,len(eps),len(qs),rep,11),np.nan)
    for (e,q),vs in orig.items():
        S[0,idx[e],qi[q],:len(vs),:]=vs
    return eps,score(S,np.array([kinds[q] for q in qs]),11)[0]

@lru_cache(None)
def order(prompt,text):
    s=' '.join((text or '').split())
    words=s.split()
    choice=OPT.search(prompt)
    if choice:
        lead=re.match(r'^\(?[A-D](?:\b|[):;.])',s)
        # Long responses that commit to a choice immediately vs explanations
        # that delay the choice are distinct transitions even at equal length.
        if lead:
            return np.log1p(max(len(words)-10,0))/2, 0
        if len(words)>3: return 0, 1
        return 0, 0
    # A direct numeric answer followed by an explanation.
    num=re.match(r'^[-+]?\d+(?:\.\d+)?(?:[.)]\s+[A-Za-z]|\s+[a-zA-Z]{2,})',s)
    if num: return min(1,np.log1p(len(words))/3),0
    return 0,0

def predict(rows):
    eps,base=scores(rows)
    by_key=defaultdict(list)
    for r in rows:
        by_key[r['endpoint'],r['id']].append(order(r['prompt'],r['response']))
    qs=sorted({r['id'] for r in rows})
    A=np.asarray([[np.mean(by_key[e,q],axis=0) for q in qs] for e in eps])
    # Prompt-specific expectation, summed to retain small independent deviations.
    residual=A-np.median(A,axis=0)
    evidence=np.sum(residual[:,:,0]-residual[:,:,1],axis=1)/np.sqrt(len(qs)/5)
    s=base+evidence
    t=np.quantile(s,2/3)
    return {e:('core' if score<=t else e) for e,score in zip(eps,s)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompts.csv'))
    parser.add_argument('--families', type=Path, help='CSV mapping formal endpoints to candidate families')
    parser.add_argument('--split', choices=['dev', 'holdout'])
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.out.resolve() in {p.resolve() for p in (args.answers,args.prompts,args.families) if p}:
        parser.error('Choose a separate output file.')
    with args.prompts.open(newline='',encoding='utf-8-sig') as f:
        bank={row['id']:row for row in csv.DictReader(f)}
    rows=[]
    slots=defaultdict(set)
    with args.answers.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            q=bank[r['id']]
            if args.split and args.split!=q['split']:
                continue
            slot=(r['id'],r['repeat'])
            if r.get('synthetic') or slot in slots[r['endpoint']] or r['prompt']!=q['prompt']:
                raise ValueError('Unexpected duplicate, synthetic or changed prompt in cache.')
            if not isinstance(r['response'],str) or not r['response'].strip():
                raise ValueError('Incomplete cached response.')
            slots[r['endpoint']].add(slot)
            rows.append(r)
    if not slots or len({frozenset(s) for s in slots.values()})!=1:
        raise ValueError('Selected endpoints need the same prompt and repeat slots.')
    if args.families:
        with args.families.open(newline='',encoding='utf-8-sig') as f:
            fr=list(csv.DictReader(f))
        family={r['endpoint']:r['family'] for r in fr}
        if len(fr)!=len(family) or set(family)!=set(slots) or not all(family.values()):
            raise ValueError('Family map must cover each endpoint exactly once.')
    elif all(e.startswith('ref-') for e in slots):
        family=dict.fromkeys(slots,'P')
    else:
        parser.error('Formal replies require a family map.')
    result=[]
    for name in sorted(set(family.values())):
        labels=predict([r for r in rows if family[r['endpoint']]==name])
        result.extend((ep,name,f'{name}/{label}') for ep,label in labels.items())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['endpoint','family','model'])
        w.writerows(sorted(result))

if __name__=='__main__':
    main()
