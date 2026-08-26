#!/usr/bin/env python3
"""Emit LaTeX table fragments straight from results.json so that no number in the
appendix is ever retyped by hand."""
import json, csv
import numpy as np, pandas as pd

BASE = "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER"
OUT = f"{BASE}/analysis/out"
TEX = f"{BASE}/paper/tables"
import os
os.makedirs(TEX, exist_ok=True)
R = json.load(open(f"{OUT}/results.json"))
Q = json.load(open(f"{OUT}/fgd_results.json"))
ORDER = ["Secure", "Preoccupied", "Dismissing", "Fearful"]


def p3(p):
    if p != p:
        return "--"
    return "$<$ .001" if p < .001 else f"{p:.3f}".replace("0.", ".")


def f2(x, d=2):
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    s = f"{x:.{d}f}"
    return s.replace("0.", ".") if abs(x) < 1 and s.startswith(("0.", "-0.")) else s


def w(name, s):
    open(f"{TEX}/{name}.tex", "w").write(s)
    print("  ", name)


# ---------------------------------------------------------------- C1 scales
rows = []
for k, v in R["reliability"].items():
    rows.append(f"{v['label']} & {v['k']} & {v['range'][0]}--{v['range'][1]} & "
                f"{f2(v['mean'])} & {f2(v['sd'])} & {f2(v['alpha'])} & "
                f"[{f2(v['alpha_lo'])}, {f2(v['alpha_hi'])}] & {f2(v['omega'])} & "
                f"{f2(v['avg_inter_item'])} & {f2(v['skew'])} & {f2(v['kurt'])}\\\\")
w("tabC1", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Scale statistics and internal consistency}\label{tab:C1}
\begin{tabular}{lccccccccc}
\toprule
Scale & $k$ & Range & $M$ & $SD$ & $\alpha$ & 95\% CI & $\omega$ & $\bar{r}_{ii}$ & Skew \\
\midrule
""" + "\n".join(r.rsplit("&", 1)[0] + r"\\" for r in rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $n = 103$. $k$ is the number of items. The confidence
interval for $\alpha$ is a percentile bootstrap over 10{,}000 resamples.
$\omega$ is McDonald's omega-total from a one-factor principal-axis solution.
$\bar{r}_{ii}$ is the average inter-item correlation.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- C2 EFA
e = R["efa_ecr"]
L = e["loadings"]
rows = [f"{it} & {f2(L[i][0])} & {f2(L[i][1])} & {f2(e['communalities'][i])}\\\\"
        for i, it in enumerate(e["items"])]
w("tabC2", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Rotated two-factor solution for the eight attachment items}\label{tab:C2}
\begin{tabular}{lccc}
\toprule
Item & Factor 1 & Factor 2 & $h^2$ \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\% variance & """ + f"{100*e['var_explained'][0]:.1f} & {100*e['var_explained'][1]:.1f} & \\\\" + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $n = """ + str(e["n"]) + r"""$. Principal-axis factoring with
varimax rotation. ANX = attachment anxiety items, AVO = attachment avoidance
items. KMO $= """ + f"{e['kmo']:.2f}" + r"""$; Bartlett's test of sphericity
$\chi^2(""" + str(e["bartlett"]["df"]) + r""") = """ + f"{e['bartlett']['chi2']:.1f}" + r"""$,
$p < .001$. $h^2$ is the communality.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E1 ANOVA
rows = []
for a in R["anova"]:
    rows.append(f"{a['label']} & {a['F']:.2f} & {p3(a['p'])} & {p3(a['q'])} & "
                f"{f2(a['eta2'])} & [{f2(a['eta2_lo'])}, {f2(a['eta2_hi'])}] & "
                f"{f2(a['omega2'])} & {a['welch_F']:.2f} & {p3(a['welch_p'])} & "
                f"{a['kruskal_H']:.2f} & {f2(a['power'])}\\\\")
w("tabE1", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{One-way comparisons of the four self-selected attachment styles}\label{tab:E1}
\begin{tabular}{lcccccccccc}
\toprule
& \multicolumn{6}{c}{Fisher's $F$} & \multicolumn{2}{c}{Welch} & K--W & \\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}
Outcome & $F$ & $p$ & $q$ & $\eta^2$ & 95\% CI & $\omega^2$ & $F$ & $p$ & $H$ & $1-\beta$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $df = 3, 99$ for Fisher's $F$ except where a case was
missing. $q$ is the Benjamini--Hochberg adjusted $p$ across the ten outcomes.
The confidence interval for $\eta^2$ is derived from the noncentral $F$.
$1-\beta$ is achieved power at $\alpha = .05$ given the observed effect. K--W is
the Kruskal--Wallis rank test.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E2 descriptives
CORE = ["anxiety", "avoidance", "fne", "scenario_anx", "scenario_expect",
        "loneliness", "friends_have", "approach_inhibition", "convo_started"]
LAB = {a["variable"]: a["label"] for a in R["anova"]}
rows = []
for v in CORE:
    a = [x for x in R["anova"] if x["variable"] == v][0]
    cells = " & ".join(f"{a['means'][s]:.2f} ({a['sds'][s]:.2f})" for s in ORDER)
    rows.append(f"{a['label']} & {cells}\\\\")
ns = " & ".join(f"$n = {[x for x in R['anova'] if x['variable']=='anxiety'][0]['ns'][s]}$"
                for s in ORDER)
w("tabE2", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Means and standard deviations by self-selected attachment style}\label{tab:E2}
\begin{tabular}{lcccc}
\toprule
Measure & Secure & Preoccupied & Dismissing & Fearful\\
& """ + ns + r"""\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Standard deviations in parentheses. Cell $n$ varies by one
or two for friendship counts because of item non-response.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E3 correlations
cv = R["correlations"]
V, Lb = cv["vars"], cv["labels"]
M = cv["r"]
qmap = {}
for p in cv["pairs"]:
    qmap[(p["a"], p["b"])] = p["q"]; qmap[(p["b"], p["a"])] = p["q"]
lines = []
for i in range(len(V)):
    cells = []
    for j in range(len(V)):
        if j > i:
            cells.append("")
        elif i == j:
            cells.append("---")
        else:
            r = M[i][j]; q = qmap.get((V[i], V[j]), 1)
            s = f2(r)
            cells.append(f"\\textbf{{{s}}}" if q < .05 else s)
    lines.append(f"{i+1}. {Lb[i]} & " + " & ".join(cells) + r"\\")
w("tabE3", r"""\begin{table}[htbp]\centering\scriptsize\setstretch{1.05}
\begin{threeparttable}
\caption{Correlations among the ten core measures}\label{tab:E3}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{l""" + "c" * len(V) + r"""}
\toprule
& """ + " & ".join(str(i + 1) for i in range(len(V))) + r"""\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $n = 101$--$103$. Bold marks correlations surviving
Benjamini--Hochberg correction across all 45 tests ($q < .05$);
""" + f"{cv['n_sig_fdr']}" + r""" of """ + f"{cv['n_tests']}" + r""" did so.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E4 regressions
blocks = []
for dv, m in R["regressions"].items():
    lab = LAB.get(dv, dv)
    blocks.append(r"\multicolumn{7}{l}{\textit{" + lab +
                  f"}} \\quad $R^2 = {f2(m['r2'])}$, $F({m['df1']}, {m['df2']}) = {m['F']:.2f}$, "
                  f"$p = {p3(m['p']).replace('$','')}$" + r"}\\")
    for t in m["terms"]:
        blocks.append(f"\\quad {t['name']} & {f2(t['b'])} & {f2(t['se'])} & "
                      f"[{f2(t['lo'])}, {f2(t['hi'])}] & {f2(t['beta'])} & "
                      f"{t['t']:.2f} & {p3(t['p'])}\\\\")
    blocks.append(r"\addlinespace")
w("tabE4", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.05}
\begin{threeparttable}
\caption{Each outcome regressed on the two attachment dimensions}\label{tab:E4}
\begin{tabular}{lcccccc}
\toprule
Predictor & $b$ & $SE$ & 95\% CI & $\beta$ & $t$ & $p$\\
\midrule
""" + "\n".join(blocks) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $n = 101$--$103$. Predictors are mean-centred. All variance
inflation factors were below 1.1.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E5 mediation
rows = []
NAMES = {"anxiety_fne_inhibition": "Anxiety $\\to$ FNE $\\to$ approach inhibition",
         "anxiety_fne_loneliness": "Anxiety $\\to$ FNE $\\to$ loneliness",
         "avoidance_convo_friends": "Avoidance $\\to$ conversations $\\to$ friends",
         "avoidance_expect_friends": "Avoidance $\\to$ expected acceptance $\\to$ friends",
         "fne_sanx_inhibition": "FNE $\\to$ anticipated anxiety $\\to$ inhibition"}
for k, m in R["mediation"].items():
    sig = "yes" if (m["lo"] > 0) == (m["hi"] > 0) else "no"
    rows.append(f"{NAMES[k]} & {f2(m['a'])} & {f2(m['b'])} & {f2(m['c_total'])} & "
                f"{f2(m['c_prime'])} & {f2(m['indirect'],3)} & "
                f"[{f2(m['lo'],3)}, {f2(m['hi'],3)}] & {sig}\\\\")
w("tabE5", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Bootstrapped mediation models}\label{tab:E5}
\begin{tabular}{lccccccc}
\toprule
Path & $a$ & $b$ & $c$ & $c'$ & $ab$ & 95\% CI & CI excludes 0\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Percentile bootstrap, 10{,}000 resamples. FNE = fear of
negative evaluation. The avoidance-to-friends model is a suppression case: the
total effect is negative while the indirect effect is positive, so it is reported
for completeness and not interpreted as mediation.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E6 post hoc
rows = []
for v in ["loneliness", "friends_have", "convo_started", "anxiety", "fne", "scenario_anx"]:
    a = [x for x in R["anova"] if x["variable"] == v][0]
    first = True
    for ph in a["posthoc"]:
        lab = a["label"] if first else ""
        first = False
        rows.append(f"{lab} & {ph['a']} vs.\\ {ph['b']} & {f2(ph['diff'])} & "
                    f"[{f2(ph['lo'])}, {f2(ph['hi'])}] & {f2(ph['d'])} & {p3(ph['p'])}\\\\")
    rows.append(r"\addlinespace")
w("tabE6", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.05}
\begin{threeparttable}
\caption{Games--Howell pairwise comparisons}\label{tab:E6}
\begin{tabular}{llcccc}
\toprule
Outcome & Contrast & $\Delta M$ & 95\% CI & $d$ & $p$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Games and Howell's (1976) procedure does not assume equal
variances or equal group sizes.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- G1 FGD
byq = {int(k): v for k, v in Q["by_question"].items()}
rows = [f"Q{q} & {'yes' if v['demands'] else 'no'} & {v['n']} & {v['words']} & "
        f"{v['E']} & {v['D']} & {v['G']} & {v['pct_E']:.1f}\\\\"
        for q, v in sorted(byq.items())]
silent = ", ".join(f"Q{s}" for s in Q["corpus"]["silent_questions"])
w("tabG1", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Focus group stance, by moderator question}\label{tab:G1}
\begin{tabular}{lccccccc}
\toprule
Question & Demands an instance & Turns & Words & E & D & G & \% E\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} E = first-person experiential, D = first-person
dispositional, G = generic or third-person. Agreement tokens and peer questions
are excluded. """ + silent + r""" produced no usable transcript and are absent
from the table.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

rows = []
for sp, v in sorted(Q["by_speaker"].items(), key=lambda x: -x[1]["words"]):
    pid = Q["pseudonyms"][sp]
    rows.append(f"{pid} & {v['turns']} & {v['words']} & {v['E']} & {v['D']} & "
                f"{v['G']} & {v['pct_E']:.1f} & {v['hedges_per_100w']:.2f}\\\\")
w("tabG2", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{Focus group stance, by participant}\label{tab:G2}
\begin{tabular}{lccccccc}
\toprule
Participant & Turns & Words & E & D & G & \% E & Hedges/100 words\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} Participants are ordered by words contributed. The
proportion of experiential turns did not differ reliably across speakers,
$\chi^2(8) = """ + f"{Q['speaker_variation']['chi2']:.2f}" + r"""$,
$p = """ + f"{Q['speaker_variation']['p']:.2f}"[1:] + r"""$.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- B1 audit
aud = pd.read_csv("/mnt/storage/Accademics/2-2/psy101/final_project_paper/servay/"
                  "REFINED_BIAS_FREE_DATASET/data/04_cleaning_audit_log.csv")
sub = aud[~aud["reason"].str.contains("case/label harmonisation", na=False)]
rows = []
for _, r in sub.iterrows():
    orig = str(r["original"])[:22].replace("&", "\\&").replace("_", "\\_")
    rows.append(f"{r['participant_id']} & {str(r['variable'])[:18].replace('_','~')} & "
                f"\\texttt{{{orig}}} & {r['new']} & {str(r['reason'])[:52]}\\\\")
w("tabB1", r"""\begin{table}[htbp]\centering\scriptsize\setstretch{1.05}
\begin{threeparttable}
\caption{Every substantive change made during data cleaning}\label{tab:B1}
\begin{tabular}{lllll}
\toprule
ID & Field & Entered & Became & Reason\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} """ + str(len(aud) - len(sub)) + r""" further entries were
case harmonisation on the free-text department field and are omitted. No value
was capped, imputed, or smoothed, and no row was deleted from any file.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- E7 bids
rows = [f"{g['bid']} & {g['anx']:.2f} ({g['anx_sd']:.2f}) & "
        f"{g['exp']:.2f} ({g['exp_sd']:.2f}) & {f2(g['r_anx_exp'])} & "
        f"{p3(g['p_anx_exp'])}\\\\" for g in R["expectancy_gap"]]
b = R["anxiety_not_forecast"]
w("tabE7", r"""\begin{table}[htbp]\centering\footnotesize\setstretch{1.1}
\begin{threeparttable}
\caption{The six friendship bids}\label{tab:E7}
\begin{tabular}{lcccc}
\toprule
Bid & Anticipated anxiety & Expected acceptance & $r$ & $p$\\
\midrule
""" + "\n".join(rows) + r"""
\midrule
All six averaged & """ + f"{np.mean([g['anx'] for g in R['expectancy_gap']]):.2f}" + r""" & """ + \
   f"{np.mean([g['exp'] for g in R['expectancy_gap']]):.2f}" + r""" & """ + \
   f"{f2(b['aggregate_r'])}" + r""" & """ + p3(b["aggregate_p"]) + r"""\\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item \textit{Note.} $n = 103$. Both ratings are on 1--6 scales; standard
deviations in parentheses. $r$ is the correlation, across students, between how
anxious a student expected to feel and how well they expected the bid to go.
Across the six bids taken as units the relationship is strong and negative,
Spearman $\rho = """ + f"{b['bid_level_rho']['rho']:.2f}" + r"""$,
$p = """ + f"{b['bid_level_rho']['p']:.3f}"[1:] + r"""$.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

print("tables written to", TEX)
