#!/usr/bin/env python3
"""
Statistical engine, part 2 of 2. Group differences, prediction, and the two
constructs the paper introduces:

  (a) the distress-outcome dissociation score D, which asks whether a student is
      paying a social cost larger than the distress they report; and
  (b) disclosure refusal, the decision to decline a follow-up conversation, which
      is treated here as behaviour rather than as attrition.
"""
import json, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
import sys
sys.path.insert(0, "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER/analysis")
from stats_core import (d, N, ORDER, CORE, LABEL, SCALES, items, desc, welch_anova,
                        brown_forsythe_var, games_howell, hedges_g, ols, hierarchical,
                        mediate, logistic, pillai_manova, kmeans, adj_rand, silhouette,
                        cochran_q, power_oneway, sensitivity_oneway, power_r,
                        sensitivity_r, bh_fdr, boot_ci, RNG, BOOT, OUT)

warnings.filterwarnings("ignore")
R = json.load(open(f"{OUT}/results_part1.json"))


def z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std(ddof=1)


# ============================================================ 5. GROUP DIFFERENCES
groups_of = lambda v: [d.loc[d.attachment_style == s, v].dropna().values for s in ORDER]
anova = []
for v in CORE:
    g = groups_of(v)
    F, p = stats.f_oneway(*g)
    allx = np.concatenate(g); gm = allx.mean()
    ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in g)
    sst = ((allx - gm) ** 2).sum()
    sse = sst - ssb
    k, n = len(g), len(allx)
    dfb, dfw = k - 1, n - k
    msw = sse / dfw
    eta2 = ssb / sst
    om2 = (ssb - dfb * msw) / (sst + msw)
    # eta2 CI via noncentral F
    def nc_ci(F, dfb, dfw, a):
        from scipy.optimize import brentq
        try:
            lo = brentq(lambda L: stats.ncf.sf(F, dfb, dfw, L) - (1 - a / 2), 0, 500)
        except Exception:
            lo = 0.0
        try:
            hi = brentq(lambda L: stats.ncf.sf(F, dfb, dfw, L) - a / 2, 0, 2000)
        except Exception:
            hi = np.nan
        return lo / (lo + n) if np.isfinite(lo) else 0.0, hi / (hi + n) if np.isfinite(hi) else np.nan
    e_lo, e_hi = nc_ci(F, dfb, dfw, .05)
    WF, wdf1, wdf2, wp = welch_anova(g)
    LV, LP = brown_forsythe_var(g)
    KW = stats.kruskal(*g)
    anova.append(dict(
        variable=v, label=LABEL[v], F=float(F), df1=int(dfb), df2=int(dfw), p=float(p),
        eta2=float(eta2), eta2_lo=float(e_lo), eta2_hi=float(e_hi), omega2=float(om2),
        welch_F=WF, welch_df1=wdf1, welch_df2=wdf2, welch_p=wp,
        levene_F=LV, levene_p=LP,
        kruskal_H=float(KW.statistic), kruskal_p=float(KW.pvalue),
        power=power_oneway(max(eta2, 1e-6), n, k), n=int(n),
        means={s: float(x.mean()) for s, x in zip(ORDER, g)},
        sds={s: float(x.std(ddof=1)) for s, x in zip(ORDER, g)},
        ns={s: int(len(x)) for s, x in zip(ORDER, g)},
        posthoc=games_howell(g, ORDER)))
qa = bh_fdr([a["p"] for a in anova])
for a, qq in zip(anova, qa):
    a["q"] = float(qq); a["sig_fdr"] = bool(qq < .05)
R["anova"] = anova
R["power"] = dict(sensitivity_f_oneway=sensitivity_oneway(N, 4),
                  sensitivity_eta2=float(sensitivity_oneway(N, 4) ** 2 / (1 + sensitivity_oneway(N, 4) ** 2)),
                  sensitivity_r=sensitivity_r(N), n=N, alpha=.05, target_power=.80)

# MANOVA over the outcome battery
Yb = d[["fne", "scenario_anx", "loneliness", "friends_have", "convo_started",
        "approach_inhibition"]].values
R["manova"] = pillai_manova(Yb, d.attachment_style.astype(str).values)

# secure vs each insecure style on every core variable
sec = d[d.attachment_style == "Secure"]
contrasts = []
for s in ORDER[1:]:
    o = d[d.attachment_style == s]
    for v in CORE:
        a, b = sec[v].dropna().values, o[v].dropna().values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        g_, gl, gh = hedges_g(a, b)
        u = stats.mannwhitneyu(a, b)
        contrasts.append(dict(style=s, variable=v, label=LABEL[v],
                              t=float(t), p=float(p), g=g_, g_lo=gl, g_hi=gh,
                              mw_U=float(u.statistic), mw_p=float(u.pvalue),
                              m_secure=float(a.mean()), m_other=float(b.mean()),
                              n_secure=int(len(a)), n_other=int(len(b))))
qc = bh_fdr([c["p"] for c in contrasts])
for c, qq in zip(contrasts, qc):
    c["q"] = float(qq); c["sig_fdr"] = bool(qq < .05)
R["secure_contrasts"] = contrasts

# ============================================================ 6. THE EXPECTANCY GAP
BIDS = ["Sitting together", "Joining a project team", "Suggesting lunch",
        "Asking for an explanation", "Asking about an outing", "Naming an upset"]
gap = []
for j in range(6):
    a = d[f"scenario_anx_{j+1}"]
    e = d[f"scenario_expect_{j+1}"]
    m = pd.concat([a, e], axis=1).dropna()
    t, p = stats.ttest_rel(m.iloc[:, 0], m.iloc[:, 1])
    diff = (m.iloc[:, 0] - m.iloc[:, 1])
    dz = diff.mean() / diff.std(ddof=1)
    r_ = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    gap.append(dict(bid=BIDS[j], anx=float(m.iloc[:, 0].mean()), anx_sd=float(m.iloc[:, 0].std(ddof=1)),
                    exp=float(m.iloc[:, 1].mean()), exp_sd=float(m.iloc[:, 1].std(ddof=1)),
                    diff=float(diff.mean()), t=float(t), p=float(p), dz=float(dz),
                    r_anx_exp=float(r_[0]), p_anx_exp=float(r_[1]), n=int(len(m))))
qg = bh_fdr([g["p"] for g in gap])
for g_, qq in zip(gap, qg):
    g_["q"] = float(qq)
R["expectancy_gap"] = gap
# repeated-measures over the six bids
long = d[[f"scenario_anx_{j+1}" for j in range(6)]].dropna()
Fq, pq = stats.friedmanchisquare(*[long.iloc[:, j] for j in range(6)])
R["bid_friedman"] = dict(chi2=float(Fq), df=5, p=float(pq), n=int(len(long)))

# ============================================================ 7. THE DISSOCIATION SCORE
d["z_fne"] = z(d.fne); d["z_sanx"] = z(d.scenario_anx)
d["z_lone"] = z(d.loneliness); d["z_fri"] = -z(d.friends_have)
d["z_inh"] = z(d.approach_inhibition); d["z_conv"] = -z(d.convo_started)
d["reported_distress"] = d[["z_fne", "z_sanx"]].mean(1)
d["social_cost"] = d[["z_lone", "z_fri", "z_inh", "z_conv"]].mean(1)
d["D"] = d.social_cost - d.reported_distress

rc = d[["z_fne", "z_sanx"]].dropna(); sc = d[["z_lone", "z_fri", "z_inh", "z_conv"]].dropna()
from stats_core import cronbach
R["dissociation"] = dict(
    definition="D = z(social cost composite) - z(reported distress composite)",
    distress_items=["fne", "scenario_anx"], cost_items=["loneliness", "-friends_have",
                                                        "approach_inhibition", "-convo_started"],
    alpha_distress=float(cronbach(rc.values)), alpha_cost=float(cronbach(sc.values)),
    r_distress_cost=float(stats.pearsonr(*d[["reported_distress", "social_cost"]].dropna().values.T)[0]),
    overall=desc(d.D))
gD = [d.loc[d.attachment_style == s, "D"].dropna().values for s in ORDER]
F, p = stats.f_oneway(*gD)
allx = np.concatenate(gD); gm = allx.mean()
ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in gD); sst = ((allx - gm) ** 2).sum()
WF, wd1, wd2, wp = welch_anova(gD)
R["dissociation"]["anova"] = dict(F=float(F), df1=3, df2=int(len(allx) - 4), p=float(p),
                                  eta2=float(ssb / sst), welch_F=WF, welch_p=wp,
                                  means={s: float(x.mean()) for s, x in zip(ORDER, gD)},
                                  sds={s: float(x.std(ddof=1)) for s, x in zip(ORDER, gD)},
                                  ns={s: int(len(x)) for s, x in zip(ORDER, gD)},
                                  posthoc=games_howell(gD, ORDER))
# is D reliably above zero within each style?
R["dissociation"]["one_sample"] = {}
for s, x in zip(ORDER, gD):
    t, p = stats.ttest_1samp(x, 0)
    ci = stats.t.interval(.95, len(x) - 1, x.mean(), stats.sem(x))
    R["dissociation"]["one_sample"][s] = dict(mean=float(x.mean()), t=float(t), p=float(p),
                                              df=int(len(x) - 1), lo=float(ci[0]), hi=float(ci[1]),
                                              d=float(x.mean() / x.std(ddof=1)))
# D against the dimensional scores
R["dissociation"]["dimensional"] = ols(
    d.D.values, d[["anxiety", "avoidance"]].values, ["Attachment anxiety", "Attachment avoidance"])

# ============================================================ 8. REGRESSION
d["male"] = (d.gender == "Man").astype(float)
d["anx_c"] = d.anxiety - d.anxiety.mean()
d["avo_c"] = d.avoidance - d.avoidance.mean()
d["anx_x_avo"] = d.anx_c * d.avo_c

DVS = ["loneliness", "friends_have", "approach_inhibition", "convo_started", "fne", "scenario_anx"]
R["regressions"] = {}
R["hierarchical"] = {}
R["moderation"] = {}
for dv in DVS:
    y = d[dv].values.astype(float)
    R["regressions"][dv] = ols(y, d[["anx_c", "avo_c"]].values,
                               ["Attachment anxiety", "Attachment avoidance"])
    R["hierarchical"][dv] = hierarchical(
        y, [("Block 1: age, gender, semester", d[["age", "male", "semester"]].values.astype(float)),
            ("Block 2: attachment dimensions", d[["anx_c", "avo_c"]].values.astype(float)),
            ("Block 3: anxiety x avoidance", d[["anx_x_avo"]].values.astype(float))],
        None)
    R["moderation"][dv] = ols(y, d[["anx_c", "avo_c", "anx_x_avo"]].values,
                              ["Attachment anxiety", "Attachment avoidance", "Anxiety x Avoidance"])
    # simple slopes of anxiety at +-1 SD of avoidance
    sd_av = d.avoidance.std(ddof=1)
    ss = {}
    for lab, off in (("Low avoidance (-1 SD)", -sd_av), ("Mean avoidance", 0.0),
                     ("High avoidance (+1 SD)", sd_av)):
        Xs = np.column_stack([d.anx_c.values, (d.avo_c - off).values,
                              (d.anx_c * (d.avo_c - off)).values]).astype(float)
        f = ols(y, Xs, ["anx", "avo", "int"])
        ss[lab] = {k2: f["terms"][0][k2] for k2 in ("b", "se", "t", "p", "lo", "hi")}
    R["moderation"][dv]["simple_slopes"] = ss

# ============================================================ 9. MEDIATION
R["mediation"] = {
    "anxiety_fne_inhibition": mediate("anxiety", "fne", "approach_inhibition"),
    "anxiety_fne_loneliness": mediate("anxiety", "fne", "loneliness"),
    "avoidance_convo_friends": mediate("avoidance", "convo_started", "friends_have"),
    "avoidance_expect_friends": mediate("avoidance", "scenario_expect", "friends_have"),
    "fne_sanx_inhibition": mediate("fne", "scenario_anx", "approach_inhibition"),
}

# ============================================================ 10. DISCLOSURE REFUSAL
d["refused"] = (d.willing_followup == "No, thank you").astype(float)
R["refusal"] = dict(
    n_refused=int(d.refused.sum()), n=int(len(d)), prop=float(d.refused.mean()),
    ci=[float(x) for x in stats.binomtest(int(d.refused.sum()), len(d)).proportion_ci(.95)],
    p_vs_half=float(stats.binomtest(int(d.refused.sum()), len(d), .5).pvalue),
    by_style={s: dict(n=int((d.attachment_style == s).sum()),
                      refused=int(d.loc[d.attachment_style == s, "refused"].sum()),
                      prop=float(d.loc[d.attachment_style == s, "refused"].mean()))
              for s in ORDER})
ctr = pd.crosstab(d.attachment_style, d.refused)
c2r, pr, dfr, _ = stats.chi2_contingency(ctr)
R["refusal"]["chi2"] = dict(chi2=float(c2r), df=int(dfr), p=float(pr),
                            cramers_v=float(np.sqrt(c2r / len(d) / min(ctr.shape[0] - 1, ctr.shape[1] - 1))))
R["refusal"]["model"] = logistic(d.refused.values,
                                 d[["anxiety", "avoidance", "fne", "loneliness"]].values.astype(float),
                                 ["Attachment anxiety", "Attachment avoidance",
                                  "Fear of negative evaluation", "Loneliness"])
R["refusal"]["model_min"] = logistic(d.refused.values, d[["avoidance"]].values.astype(float),
                                     ["Attachment avoidance"])
for v in ["anxiety", "avoidance", "fne", "loneliness", "friends_have", "convo_started", "D"]:
    a = d.loc[d.refused == 1, v].dropna(); b = d.loc[d.refused == 0, v].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    g_, gl, gh = hedges_g(a.values, b.values)
    R["refusal"].setdefault("contrasts", {})[v] = dict(
        m_refused=float(a.mean()), m_willing=float(b.mean()), t=float(t), p=float(p),
        g=g_, g_lo=gl, g_hi=gh, n_refused=int(len(a)), n_willing=int(len(b)))

# ============================================================ 11. EMPIRICAL CLUSTERS
X = d[["anxiety", "avoidance"]].dropna()
Xz = np.column_stack([z(X.anxiety), z(X.avoidance)])
lab, cent, wss = kmeans(Xz, 4)
# name clusters by quadrant of their centroid
names = []
for c in cent:
    hi_a, hi_v = c[0] > 0, c[1] > 0
    names.append("Fearful-like" if (hi_a and hi_v) else "Preoccupied-like" if hi_a
                 else "Dismissing-like" if hi_v else "Secure-like")
selfsel = d.loc[X.index, "attachment_style"].astype(str).values
ct = pd.crosstab(pd.Series([names[i] for i in lab], name="empirical"),
                 pd.Series(selfsel, name="self_selected"))
c2k, pk, dfk, _ = stats.chi2_contingency(ct)
sils = {k: silhouette(Xz, kmeans(Xz, k)[0]) for k in (2, 3, 4, 5)}
R["clusters"] = dict(
    k=4, n=int(len(Xz)), wss=wss, ari=adj_rand([names[i] for i in lab], selfsel),
    silhouette=silhouette(Xz, lab), silhouette_by_k=sils,
    centroids={names[i]: [float(cent[i, 0]), float(cent[i, 1])] for i in range(4)},
    sizes={n_: int(c) for n_, c in pd.Series([names[i] for i in lab]).value_counts().items()},
    crosstab=ct.to_dict(),
    chi2=float(c2k), df=int(dfk), p=float(pk),
    cramers_v=float(np.sqrt(c2k / len(Xz) / 3)),
    agreement=float(np.mean([names[i].replace("-like", "") == s for i, s in zip(lab, selfsel)])))
# does the self-selected label add anything over the two dimensions?
dummies = pd.get_dummies(d.attachment_style, drop_first=True).astype(float)
for dv in ["loneliness", "friends_have"]:
    st = hierarchical(d[dv].values.astype(float),
                      [("Block 1: anxiety + avoidance", d[["anx_c", "avo_c"]].values.astype(float)),
                       ("Block 2: + self-selected style", dummies.values)], None)
    R["clusters"].setdefault("incremental", {})[dv] = st

# ============================================================ 12. AVOIDED SITUATIONS
SITS = ["Speaking in class", "Joining a cafeteria group", "Club or society events",
        "Group project meetings", "Messaging someone first", "Asking a classmate for help"]
M = d[[f"avoid_sel_{j+1}" for j in range(6)]].values
Q, dfq, pq2 = cochran_q(M)
sit = []
for j, s in enumerate(SITS):
    col = M[:, j]
    k_ = int(col.sum())
    ci = stats.binomtest(k_, len(col)).proportion_ci(.95)
    ranks = d[f"avoid_rank_{j+1}"].dropna()
    by = {}
    for st_ in ORDER:
        sub = d.loc[d.attachment_style == st_, f"avoid_sel_{j+1}"]
        by[st_] = float(sub.mean())
    ctj = pd.crosstab(d.attachment_style, d[f"avoid_sel_{j+1}"])
    c2j, pj, dfj, _ = stats.chi2_contingency(ctj) if ctj.shape[1] > 1 else (np.nan,) * 4
    sit.append(dict(situation=s, n=k_, pct=float(100 * k_ / len(col)),
                    lo=float(100 * ci[0]), hi=float(100 * ci[1]),
                    mean_rank=float(ranks.mean()) if len(ranks) else np.nan,
                    by_style=by, chi2=float(c2j), p=float(pj)))
R["situations"] = dict(items=sit, cochran_Q=Q, df=dfq, p=pq2,
                       n_avoided=desc(d.n_situations_avoided))
gn = [d.loc[d.attachment_style == s, "n_situations_avoided"].dropna().values for s in ORDER]
Fn, pn = stats.f_oneway(*gn)
R["situations"]["n_anova"] = dict(F=float(Fn), p=float(pn),
                                  means={s: float(x.mean()) for s, x in zip(ORDER, gn)})

# ============================================================ 13. HOUSEHOLD BACKGROUND
R["background"] = {}
for v in ["anxiety", "avoidance", "loneliness", "fne"]:
    g = [d.loc[d.caregiver == c, v].dropna().values for c in
         ["Both parents", "One parent", "Grandparents / other relatives", "Other"]]
    g = [x for x in g if len(x) > 1]
    F, p = stats.f_oneway(*g)
    rho, prho = stats.spearmanr(*d[["family_moves", v]].dropna().values.T)
    R["background"][v] = dict(caregiver_F=float(F), caregiver_p=float(p),
                              moves_rho=float(rho), moves_p=float(prho),
                              n=int(d[["family_moves", v]].dropna().shape[0]))
ctb = pd.crosstab(d.caregiver, d.attachment_style)
c2b, pb, dfb2, _ = stats.chi2_contingency(ctb)
R["background"]["caregiver_style_chi2"] = dict(chi2=float(c2b), df=int(dfb2), p=float(pb),
                                               cramers_v=float(np.sqrt(c2b / len(d) / min(ctb.shape[0] - 1, ctb.shape[1] - 1))))

# ============================================================ 14. FRIENDSHIP COUNTS
R["friends"] = dict(
    have=desc(d.friends_have), want=desc(d.friends_want), gap=desc(d.friend_gap),
    t_gap=dict(**{k: float(v) for k, v in zip(
        ("t", "p"), stats.ttest_1samp(d.friend_gap.dropna(), 0))},
        d=float(d.friend_gap.mean() / d.friend_gap.std(ddof=1)),
        n=int(d.friend_gap.notna().sum())),
    zero_friends=int((d.friends_have == 0).sum()),
    zero_pct=float(100 * (d.friends_have == 0).mean()),
    zero_by_style={s: float(100 * (d.loc[d.attachment_style == s, "friends_have"] == 0).mean())
                   for s in ORDER},
    wilcoxon_gap=float(stats.wilcoxon(d.friend_gap.dropna()).pvalue))

d.to_csv(f"{OUT}/analytic_with_derived.csv", index=False)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)

print("ANOVA (raw p / FDR q / eta2):")
for a in R["anova"]:
    print(f"  {a['label']:<28} F={a['F']:6.3f} p={a['p']:.4f} q={a['q']:.4f} eta2={a['eta2']:.3f} power={a['power']:.2f}")
print("MANOVA Pillai", round(R["manova"]["pillai"], 3), "p", R["manova"]["p"])
print("Dissociation D by style:", {k: round(v, 3) for k, v in R["dissociation"]["anova"]["means"].items()},
      "F", round(R["dissociation"]["anova"]["F"], 3), "p", round(R["dissociation"]["anova"]["p"], 5))
print("Refusal:", R["refusal"]["prop"], R["refusal"]["by_style"])
print("Refusal model AUC", round(R["refusal"]["model"]["auc"], 3), "p", round(R["refusal"]["model"]["p"], 4))
print("Cluster ARI", round(R["clusters"]["ari"], 3), "agreement", round(R["clusters"]["agreement"], 3))
print("Sensitivity: f =", round(R["power"]["sensitivity_f_oneway"], 3),
      " r =", round(R["power"]["sensitivity_r"], 3))
