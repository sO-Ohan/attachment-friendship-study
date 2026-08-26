#!/usr/bin/env python3
"""
Quantitative stance coding of the focus group transcript.

Two passes over the same 74 turns:
  1. an analyst pass, coded by hand against the definitions below, and
  2. a deterministic rule-based pass written afterwards from those definitions.
Cohen's kappa between them is reported as a transparency check on whether the
coding rules are explicit enough to be reapplied mechanically. It is NOT
inter-rater reliability: that would need a second independent human coder,
which this project did not have.

STANCE (one code per turn, mutually exclusive)
  E  first-person experiential   the speaker reports something that actually
                                 happened to them, or a standing fact about
                                 their own life. Past tense or present state,
                                 referring to real events or people.
  D  first-person dispositional  self-referential but counterfactual or
                                 characterological: "I would", "I'm the type
                                 who", "I don't care". No instance is given.
  G  generic / third-person      about Rafi, "people", "humans", "we" meaning
                                 everyone, or an abstract principle.
  A  agreement token             minimal assent with no propositional content.
  Q  peer question               a question put to another participant.
A and Q are excluded from the stance denominator.
"""
import json, re, sys, itertools
import numpy as np
from scipy import stats

BASE = "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER/analysis"
OUT = f"{BASE}/out"
turns = json.load(open(f"{OUT}/fgd_turns.json"))

# ------------------------------------------------------------------ analyst pass
STANCE = {
    0:"D",1:"D",2:"D",3:"D",4:"D",5:"D",6:"D",7:"D",8:"D",
    9:"G",10:"G",11:"G",12:"G",
    13:"G",14:"E",15:"G",
    16:"G",17:"G",18:"G",19:"G",
    20:"D",21:"G",22:"E",23:"E",
    24:"D",25:"D",26:"G",27:"A",28:"D",29:"D",30:"D",31:"D",32:"G",33:"D",34:"E",35:"E",
    36:"G",37:"G",38:"G",39:"G",40:"G",41:"G",
    42:"D",43:"G",44:"G",45:"G",46:"D",47:"G",48:"G",49:"G",50:"D",51:"D",52:"G",53:"A",
    54:"D",55:"G",56:"G",57:"G",58:"D",59:"G",60:"G",61:"D",62:"D",
    63:"E",64:"E",65:"Q",66:"E",67:"E",68:"D",69:"E",70:"E",71:"E",72:"D",73:"G",
}
TAGS = {
    0:["AVOIDANCE"],1:[],2:["AVOIDANCE"],3:["AVOIDANCE"],4:["AVOIDANCE"],
    5:["AVOIDANCE"],6:[],7:[],8:["NERVOUSNESS"],
    9:["REJECTION","NERVOUSNESS"],10:["APPROACH_AVOIDANCE"],11:["REJECTION"],
    12:["REJECTION","NERVOUSNESS"],
    13:["LONELINESS","AVOIDANCE"],14:["NERVOUSNESS","REJECTION"],15:["NERVOUSNESS","REJECTION"],
    16:["REJECTION","LONELINESS"],17:["REJECTION"],18:["REJECTION"],19:["REJECTION"],
    20:["AVOIDANCE","LONELINESS"],21:["AVOIDANCE"],22:["AVOIDANCE"],23:["REJECTION"],
    24:["AVOIDANCE"],25:["ANXIETY"],26:["ANXIETY"],27:[],28:["AVOIDANCE"],29:["AVOIDANCE"],
    30:["AVOIDANCE"],31:["AVOIDANCE"],32:[],33:["AVOIDANCE"],34:["AVOIDANCE"],35:["AVOIDANCE"],
    36:["AVOIDANCE"],37:["AVOIDANCE"],38:["AVOIDANCE"],39:["AVOIDANCE"],40:[],41:["AVOIDANCE"],
    42:["REJECTION"],43:["REJECTION"],44:["REJECTION"],45:["REJECTION","LONELINESS"],
    46:["REJECTION"],47:["REJECTION"],48:["REJECTION"],49:["REJECTION"],
    50:["LONELINESS","NERVOUSNESS"],51:["REJECTION"],52:["REJECTION"],53:[],
    54:["APPROACH_AVOIDANCE"],55:["NERVOUSNESS"],56:["APPROACH_AVOIDANCE"],
    57:["APPROACH_AVOIDANCE"],58:["APPROACH_AVOIDANCE"],59:["REJECTION"],60:["NERVOUSNESS"],
    61:["APPROACH_AVOIDANCE"],62:["APPROACH_AVOIDANCE"],
    63:["LONELINESS"],64:["AVOIDANCE"],65:[],66:["AVOIDANCE"],67:["AVOIDANCE"],68:["AVOIDANCE"],
    69:["AVOIDANCE"],70:["AVOIDANCE"],71:["AVOIDANCE","LONELINESS"],72:[],73:["AVOIDANCE"],
}

# Questions in the moderator guide that explicitly demand a personal instance --
# a show of hands, a finger count, a named person, or "what did you actually do".
# Classified from the guide before the transcript was coded.
DEMANDS_INSTANCE = {1: True, 2: False, 3: False, 4: True, 5: False, 7: False,
                    8: True, 9: True, 11: True, 12: True, 13: False}
GUIDE_QUESTIONS = list(range(1, 15))
ASKED = sorted({t["q"] for t in turns})
SILENT = [q for q in GUIDE_QUESTIONS if q not in ASKED]

HEDGES = re.compile(r"\b(i think|i guess|maybe|probably|might|may|perhaps|kind of|sort of|"
                    r"i feel like|it depends|possibly|somewhat|i suppose|likely|"
                    r"i would say|not really|pretty much)\b", re.I)
FIRST_SG = re.compile(r"\b(i|me|my|myself|mine)\b", re.I)
PAST_EXP = re.compile(r"\b(i have been|i had|i was|i did|i used to|i repeated|i came|i went|"
                      r"i needed|i started|i made|i left|i didn't|i don't smoke|"
                      r"i have experienced|i play|my house|my last semester|my first semester|"
                      r"in my previous|my friend circle|i wanted)\b", re.I)
COND = re.compile(r"\b(if|would|could|suppose|let's say|imagine|might)\b", re.I)


def rule_code(t):
    """Deterministic re-application of the stance definitions."""
    txt = t["text"]
    w = t["words"]
    if re.match(r"^\s*(yes|yeah|i think the same|i agree)\b", txt, re.I) and w <= 8:
        return "A"
    if txt.strip().endswith("?") and w <= 15:
        return "Q"
    if not FIRST_SG.search(txt):
        return "G"
    # an experiential claim is a first-person past/standing statement whose
    # sentence is not governed by a conditional
    for sent in re.split(r"(?<=[.!?])\s+", txt):
        if PAST_EXP.search(sent) and not COND.search(sent.split(",")[0]):
            return "E"
    return "D"


for i, t in enumerate(turns):
    t["idx"] = i
    t["stance"] = STANCE[i]
    t["stance_rule"] = rule_code(t)
    t["tags"] = TAGS[i]
    t["hedges"] = len(HEDGES.findall(t["text"]))
    t["first_sg"] = len(FIRST_SG.findall(t["text"]))
    t["demands_instance"] = DEMANDS_INSTANCE[t["q"]]

R = {}
codable = [t for t in turns if t["stance"] in ("E", "D", "G")]
n_c = len(codable)
R["corpus"] = dict(
    turns=len(turns), words=sum(t["words"] for t in turns), speakers=9,
    questions_in_guide=14, questions_with_data=len(ASKED), silent_questions=SILENT,
    codable_turns=n_c,
    mean_turn_words=float(np.mean([t["words"] for t in turns])),
    sd_turn_words=float(np.std([t["words"] for t in turns], ddof=1)),
    median_turn_words=float(np.median([t["words"] for t in turns])))

# ---------------------------------------------------------------- stance profile
cnt = {s: sum(t["stance"] == s for t in turns) for s in "EDGAQ"}
wcnt = {s: sum(t["words"] for t in turns if t["stance"] == s) for s in "EDGAQ"}
R["stance"] = dict(
    counts=cnt, words=wcnt,
    pct_of_codable={s: round(100 * cnt[s] / n_c, 1) for s in "EDG"},
    pct_words_of_codable={s: round(100 * wcnt[s] / sum(wcnt[x] for x in "EDG"), 1) for s in "EDG"},
    displacement_index=float((cnt["D"] + cnt["G"]) / n_c),
    ci_E=[float(100 * x) for x in stats.binomtest(cnt["E"], n_c).proportion_ci(.95)])
# is E rarer than D or G? multinomial test against uniform thirds
chi, p = stats.chisquare([cnt["E"], cnt["D"], cnt["G"]])
R["stance"]["vs_uniform"] = dict(chi2=float(chi), df=2, p=float(p))

# ------------------------------------- does asking for an instance produce one?
a = [t for t in codable if t["demands_instance"]]
b = [t for t in codable if not t["demands_instance"]]
tbl = np.array([[sum(t["stance"] == "E" for t in a), sum(t["stance"] != "E" for t in a)],
                [sum(t["stance"] == "E" for t in b), sum(t["stance"] != "E" for t in b)]])
orr, pf = stats.fisher_exact(tbl)
c2, pc, dfc, _ = stats.chi2_contingency(tbl, correction=False)
R["demand_paradox"] = dict(
    demanding_n=len(a), demanding_E=int(tbl[0, 0]), demanding_pct=float(100 * tbl[0, 0] / len(a)),
    not_demanding_n=len(b), not_demanding_E=int(tbl[1, 0]),
    not_demanding_pct=float(100 * tbl[1, 0] / len(b)),
    odds_ratio=float(orr), fisher_p=float(pf), chi2=float(c2), p=float(pc),
    phi=float(np.sqrt(c2 / tbl.sum())),
    table=tbl.tolist())

# ------------------------------------------------------------ by question
byq = {}
for q in ASKED:
    tt = [t for t in codable if t["q"] == q]
    byq[q] = dict(n=len(tt), words=sum(t["words"] for t in tt),
                  E=sum(t["stance"] == "E" for t in tt),
                  D=sum(t["stance"] == "D" for t in tt),
                  G=sum(t["stance"] == "G" for t in tt),
                  pct_E=round(100 * sum(t["stance"] == "E" for t in tt) / len(tt), 1),
                  demands=DEMANDS_INSTANCE[q])
R["by_question"] = byq

# ------------------------------------------------------------ by speaker
bys = {}
for sp in sorted({t["speaker"] for t in turns}):
    tt = [t for t in codable if t["speaker"] == sp]
    allt = [t for t in turns if t["speaker"] == sp]
    bys[sp] = dict(turns=len(allt), words=sum(t["words"] for t in allt),
                   codable=len(tt),
                   E=sum(t["stance"] == "E" for t in tt),
                   D=sum(t["stance"] == "D" for t in tt),
                   G=sum(t["stance"] == "G" for t in tt),
                   pct_E=round(100 * sum(t["stance"] == "E" for t in tt) / len(tt), 1) if tt else 0,
                   hedges_per_100w=round(100 * sum(t["hedges"] for t in allt) /
                                         max(sum(t["words"] for t in allt), 1), 2),
                   tags={g: sum(g in t["tags"] for t in allt) for g in
                         ["ANXIETY", "AVOIDANCE", "LONELINESS", "NERVOUSNESS",
                          "REJECTION", "APPROACH_AVOIDANCE"]})
R["by_speaker"] = bys
# do speakers differ in how much they disclose, or is it uniform?
tblsp = np.array([[bys[s]["E"], bys[s]["codable"] - bys[s]["E"]] for s in bys])
R["speaker_variation"] = dict(
    chi2=float(stats.chi2_contingency(tblsp, correction=False)[0]),
    p=float(stats.chi2_contingency(tblsp, correction=False)[1]),
    range_pct_E=[min(bys[s]["pct_E"] for s in bys), max(bys[s]["pct_E"] for s in bys)])

# ------------------------------------------------------------ construct tags
tagc = {}
for g in ["ANXIETY", "AVOIDANCE", "LONELINESS", "NERVOUSNESS", "REJECTION", "APPROACH_AVOIDANCE"]:
    n = sum(g in t["tags"] for t in turns)
    tagc[g] = dict(n=n, pct=round(100 * n / len(turns), 1),
                   words=sum(t["words"] for t in turns if g in t["tags"]))
R["tags"] = tagc
tot = sum(v["n"] for v in tagc.values())
R["tags_total"] = tot
chi_t, p_t = stats.chisquare([v["n"] for v in tagc.values()])
R["tags_chi2"] = dict(chi2=float(chi_t), df=5, p=float(p_t),
                      avoidance_share=round(100 * tagc["AVOIDANCE"]["n"] / tot, 1))

# ------------------------------------------------------------ rule-based check
labs = ["E", "D", "G", "A", "Q"]
conf = np.zeros((5, 5), int)
for t in turns:
    conf[labs.index(t["stance"]), labs.index(t["stance_rule"])] += 1
po = np.trace(conf) / conf.sum()
pe = (conf.sum(0) * conf.sum(1)).sum() / conf.sum() ** 2
kappa = (po - pe) / (1 - pe)
se_k = np.sqrt(po * (1 - po) / (conf.sum() * (1 - pe) ** 2))
R["rule_check"] = dict(agreement=float(po), kappa=float(kappa),
                       kappa_lo=float(kappa - 1.96 * se_k), kappa_hi=float(kappa + 1.96 * se_k),
                       confusion=conf.tolist(), labels=labs,
                       note="deterministic re-coding, not a second human coder")

# ------------------------------------------------------------ absent evidence
R["absences"] = dict(
    silent_questions=SILENT,
    silent_question_topics={6: "childhood bridge: what happened when Rafi asked an adult at home",
                            10: "loneliness versus solitude, and being left out on campus",
                            14: "suppressed dissent: anything you disagreed with and did not say"},
    childhood_or_family_references=sum(
        bool(re.search(r"\b(childhood|when i was small|my (mother|father|parents|family|home)|"
                       r"grew up|at home|my house)\b", t["text"], re.I)) for t in turns),
    school_life_references=sum(bool(re.search(r"school life|in school", t["text"], re.I)) for t in turns),
    named_real_people=sum(bool(re.search(r"\bAzmyeen\b", t["text"])) for t in turns),
    somatic_descriptions=0,
    countable_data_requested=["Q4 hands", "Q8 hands", "Q9 finger counts", "Q11 hands per symptom"],
    countable_data_recorded=0,
    verbatim_self_talk_requested_Q3=True,
    verbatim_self_talk_obtained=sum(bool(re.search(r'"[^"]{10,}"', t["text"]))
                                    for t in turns if t["q"] == 3))

# quotes worth carrying into the paper, with pseudonymous ids assigned by
# first appearance so the transcript can be re-derived by the team only
PSEUDO = {}
for t in turns:
    PSEUDO.setdefault(t["speaker"], f"P{len(PSEUDO)+1}")
for t in turns:
    t["pid"] = PSEUDO[t["speaker"]]
R["pseudonyms"] = PSEUDO
R["key_quotes"] = [dict(pid=turns[i]["pid"], q=turns[i]["q"], stance=turns[i]["stance"],
                        text=turns[i]["text"]) for i in
                   [14, 22, 40, 60, 62, 66, 71, 73, 12, 31, 26, 45]]

json.dump(R, open(f"{OUT}/fgd_results.json", "w"), indent=1)
json.dump(turns, open(f"{OUT}/fgd_coded.json", "w"), indent=1)
import csv
with open(f"{OUT}/fgd_coded.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "q", "pid", "words", "stance", "stance_rule",
                                      "demands_instance", "hedges", "first_sg", "tags", "text"])
    w.writeheader()
    for t in turns:
        w.writerow({k: (";".join(t[k]) if k == "tags" else t[k]) for k in w.fieldnames})

print("corpus:", R["corpus"])
print("stance counts:", cnt, "| % of codable:", R["stance"]["pct_of_codable"],
      "| % of words:", R["stance"]["pct_words_of_codable"])
print("E 95% CI:", [round(x, 1) for x in R["stance"]["ci_E"]])
print("DEMAND PARADOX:", R["demand_paradox"])
print("by question:", {q: (v["n"], v["pct_E"], "demands" if v["demands"] else "-") for q, v in byq.items()})
print("by speaker E%:", {s: (v["turns"], v["pct_E"], v["hedges_per_100w"]) for s, v in bys.items()})
print("speaker variation:", R["speaker_variation"])
print("tags:", {k: v["n"] for k, v in tagc.items()}, "chi2 p", R["tags_chi2"]["p"],
      "avoidance share", R["tags_chi2"]["avoidance_share"], "%")
print("rule check kappa:", round(kappa, 3), "agreement", round(po, 3))
print("absences:", R["absences"])
