#!/usr/bin/env python3
"""
Statistical engine, part 3. Confirmatory tests for the three claims the paper
actually rests on, plus the tests that decide whether the exploratory
dissociation index survives.

  A. Recognition beats report: does the self-selected prototype predict social
     outcomes over and above the two Likert dimensions, and is that increment
     robust to bootstrapping and cross-validation?
  B. Anxiety is not a forecast: is anticipated anxiety independent of expected
     acceptance, aggregated and bid by bid, with an equivalence test rather than
     a bare null.
  C. The zero-friend gradient across styles, tested for trend.
"""
import json, warnings, sys
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER/analysis")
from stats_core import (d, N, ORDER, OUT, RNG, BOOT, desc, hedges_g, cronbach)

warnings.filterwarnings("ignore")
R = json.load(open(f"{OUT}/results.json"))
d = pd.read_csv(f"{OUT}/analytic_with_derived.csv")
d["attachment_style"] = pd.Categorical(d["attachment_style"], categories=ORDER)


def r2(y, X):
    m = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[m], X[m]
    Xd = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(Xd, y, rcond=None)[0]
    return 1 - ((y - Xd @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# =============================================== A. RECOGNITION VS REPORT
dum = pd.get_dummies(d.attachment_style, drop_first=True).astype(float).values
DIM = d[["anxiety", "avoidance"]].values.astype(float)
rec = {}
for dv in ["friends_have", "loneliness", "convo_started", "approach_inhibition", "fne"]:
    y = d[dv].values.astype(float)
    m = np.isfinite(y) & np.isfinite(DIM).all(1) & np.isfinite(dum).all(1)
    yv, Dv, Cv = y[m], DIM[m], dum[m]
    n = m.sum()
    r2d, r2c = r2(yv, Dv), r2(yv, Cv)
    r2b = r2(yv, np.column_stack([Dv, Cv]))

    # bootstrap the difference in explanatory power
    diff = np.empty(BOOT)
    for b_ in range(BOOT):
        i = RNG.integers(0, n, n)
        diff[b_] = r2(yv[i], Cv[i]) - r2(yv[i], Dv[i])
    lo, hi = np.percentile(diff, [2.5, 97.5])

    # 10-fold cross-validated R2, so the increment is not just extra parameters
    def cv_r2(X, folds=10, reps=20):
        outv = []
        for _ in range(reps):
            idx = RNG.permutation(n)
            pred = np.empty(n)
            for f in range(folds):
                te = idx[f::folds]
                tr = np.setdiff1d(idx, te)
                Xt = np.column_stack([np.ones(len(tr)), X[tr]])
                bb = np.linalg.lstsq(Xt, yv[tr], rcond=None)[0]
                pred[te] = np.column_stack([np.ones(len(te)), X[te]]) @ bb
            outv.append(1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum())
        return float(np.mean(outv)), float(np.std(outv))

    cvd, cvd_s = cv_r2(Dv)
    cvc, cvc_s = cv_r2(Cv)
    cvb, cvb_s = cv_r2(np.column_stack([Dv, Cv]))

    # nested F test: category over dimensions
    def sse(X):
        Xd_ = np.column_stack([np.ones(n), X])
        bb = np.linalg.lstsq(Xd_, yv, rcond=None)[0]
        return ((yv - Xd_ @ bb) ** 2).sum()
    s_red, s_full = sse(Dv), sse(np.column_stack([Dv, Cv]))
    dfn, dfd = Cv.shape[1], n - Dv.shape[1] - Cv.shape[1] - 1
    Fc = ((s_red - s_full) / dfn) / (s_full / dfd)
    # and the reverse: dimensions over category
    s_red2 = sse(Cv)
    dfn2 = Dv.shape[1]
    Fc2 = ((s_red2 - s_full) / dfn2) / (s_full / dfd)

    rec[dv] = dict(n=int(n), r2_dimensions=float(r2d), r2_category=float(r2c),
                   r2_both=float(r2b), diff=float(r2c - r2d), diff_lo=float(lo), diff_hi=float(hi),
                   cv_r2_dimensions=cvd, cv_r2_category=cvc, cv_r2_both=cvb,
                   cv_sd_dimensions=cvd_s, cv_sd_category=cvc_s, cv_sd_both=cvb_s,
                   F_category_over_dimensions=float(Fc), df1=int(dfn), df2=int(dfd),
                   p_category_over_dimensions=float(stats.f.sf(Fc, dfn, dfd)),
                   F_dimensions_over_category=float(Fc2),
                   p_dimensions_over_category=float(stats.f.sf(Fc2, dfn2, dfd)))
R["recognition_vs_report"] = rec

# how well does the prototype choice line up with the dimension the scales measure?
R["prototype_validity"] = {}
for dim in ["anxiety", "avoidance"]:
    g = [d.loc[d.attachment_style == s, dim].dropna().values for s in ORDER]
    # theory says: anxiety high in Preoccupied+Fearful; avoidance high in Dismissing+Fearful
    if dim == "anxiety":
        hi = np.concatenate([g[1], g[3]]); lo_ = np.concatenate([g[0], g[2]])
    else:
        hi = np.concatenate([g[2], g[3]]); lo_ = np.concatenate([g[0], g[1]])
    t, p = stats.ttest_ind(hi, lo_, equal_var=False)
    gg, gl, gh = hedges_g(hi, lo_)
    R["prototype_validity"][dim] = dict(
        m_theory_high=float(hi.mean()), m_theory_low=float(lo_.mean()),
        n_high=int(len(hi)), n_low=int(len(lo_)),
        t=float(t), p=float(p), g=gg, g_lo=gl, g_hi=gh)
# self-rating of each of the 4 vignettes vs the one chosen
vg = {}
for s, col in zip(ORDER, ["vignette_secure", "vignette_preoccupied",
                          "vignette_dismissing", "vignette_fearful"]):
    vg[s] = {t: float(d.loc[d.attachment_style == t, col].mean()) for t in ORDER}
R["prototype_validity"]["vignette_ratings"] = vg
hits = sum(int(d.loc[i, {"Secure": "vignette_secure", "Preoccupied": "vignette_preoccupied",
                         "Dismissing": "vignette_dismissing", "Fearful": "vignette_fearful"}[
    d.loc[i, "attachment_style"]]] ==
    d.loc[i, ["vignette_secure", "vignette_preoccupied",
              "vignette_dismissing", "vignette_fearful"]].max())
    for i in d.index if pd.notna(d.loc[i, "attachment_style"]))
R["prototype_validity"]["chose_own_highest_rated"] = dict(
    n=int(hits), of=int(N), pct=float(100 * hits / N),
    p_vs_chance=float(stats.binomtest(hits, N, 0.25).pvalue))

# =============================================== B. ANXIETY IS NOT A FORECAST
A = d[[f"scenario_anx_{j+1}" for j in range(6)]]
E = d[[f"scenario_expect_{j+1}" for j in range(6)]]
m = pd.concat([A.mean(1), E.mean(1)], axis=1).dropna()
r_agg, p_agg = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
# TOST equivalence test against bounds of +-.30 (a small-to-medium effect)
def tost_r(r, n, bound=.30):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); zb = np.arctanh(bound)
    p1 = stats.norm.sf((z - (-zb)) / se)      # H0: r <= -bound
    p2 = stats.norm.cdf((z - zb) / se)        # H0: r >= +bound
    return float(max(p1, p2))
R["anxiety_not_forecast"] = dict(
    aggregate_r=float(r_agg), aggregate_p=float(p_agg), n=int(len(m)),
    ci=[float(x) for x in np.tanh(np.arctanh(r_agg) + np.array([-1.96, 1.96]) / np.sqrt(len(m) - 3))],
    tost_p_bound_030=tost_r(r_agg, len(m), .30),
    tost_p_bound_020=tost_r(r_agg, len(m), .20),
    per_bid=[dict(bid=g["bid"], r=g["r_anx_exp"], p=g["p_anx_exp"], n=g["n"],
                  tost_p_030=tost_r(g["r_anx_exp"], g["n"], .30))
             for g in R["expectancy_gap"]],
    max_abs_r=float(max(abs(g["r_anx_exp"]) for g in R["expectancy_gap"])))
# the two scales are internally coherent, so the null is not measurement failure
R["anxiety_not_forecast"]["alpha_anx"] = R["reliability"]["scenario_anx"]["alpha"]
R["anxiety_not_forecast"]["alpha_exp"] = R["reliability"]["scenario_expect"]["alpha"]
# bid ordering
bids = sorted(R["expectancy_gap"], key=lambda g: g["anx"])
R["anxiety_not_forecast"]["cheapest"] = bids[0]
R["anxiety_not_forecast"]["dearest"] = bids[-1]
rk_a = stats.rankdata([g["anx"] for g in R["expectancy_gap"]])
rk_e = stats.rankdata([-g["exp"] for g in R["expectancy_gap"]])
rho_b, prho_b = stats.spearmanr(rk_a, rk_e)
R["anxiety_not_forecast"]["bid_level_rho"] = dict(rho=float(rho_b), p=float(prho_b), n=6)

# =============================================== C. THE ZERO-FRIEND GRADIENT
zf = [(s, int((d.loc[d.attachment_style == s, "friends_have"] == 0).sum()),
       int(d.loc[d.attachment_style == s, "friends_have"].notna().sum())) for s in ORDER]
tbl = np.array([[k, n - k] for _, k, n in zf])
c2z, pz, dfz, _ = stats.chi2_contingency(tbl)
# Cochran-Armitage trend across the ordered scores Secure<Preoccupied<Dismissing<Fearful
scores = np.array([0, 1, 2, 3], float)
nk = tbl.sum(1); xk = tbl[:, 0]
Nn = nk.sum(); p_ = xk.sum() / Nn
Tstat = (scores * (xk - nk * p_)).sum()
Var = p_ * (1 - p_) * ((nk * scores**2).sum() - (nk * scores).sum() ** 2 / Nn)
Zt = Tstat / np.sqrt(Var)
R["zero_friends"] = dict(
    by_style=[dict(style=s, zero=k, n=n, pct=100 * k / n) for s, k, n in zf],
    overall_n=int(tbl[:, 0].sum()), overall_pct=float(100 * tbl[:, 0].sum() / tbl.sum()),
    chi2=float(c2z), df=int(dfz), p=float(pz),
    fisher_p=float(stats.fisher_exact(tbl[[0, 3]])[1]),
    trend_z=float(Zt), trend_p=float(2 * stats.norm.sf(abs(Zt))),
    or_fearful_vs_secure=float((tbl[3, 0] * tbl[0, 1]) / max(tbl[3, 1] * tbl[0, 0], 1e-9)))

# =============================================== D. DISSOCIATION, RESPECIFIED
# The pre-specified index used four cost indicators and had poor internal
# consistency. The respecified version keeps only the two outcome indicators
# that the ANOVA showed to move, and is reported as exploratory.
z = lambda s: (s - s.mean()) / s.std(ddof=1)
d["cost2"] = pd.concat([z(d.loneliness), -z(d.friends_have)], axis=1).mean(1)
d["distress2"] = pd.concat([z(d.fne), z(d.scenario_anx)], axis=1).mean(1)
d["D2"] = d.cost2 - d.distress2
gD2 = [d.loc[d.attachment_style == s, "D2"].dropna().values for s in ORDER]
F2, p2 = stats.f_oneway(*gD2)
allx = np.concatenate(gD2); gm = allx.mean()
ssb = sum(len(x) * (x.mean() - gm) ** 2 for x in gD2); sst = ((allx - gm) ** 2).sum()
sd_pair = stats.ttest_ind(gD2[2], gD2[0], equal_var=False)
gg2, gl2, gh2 = hedges_g(gD2[2], gD2[0])
R["dissociation"]["respecified"] = dict(
    note="exploratory; two cost indicators only",
    alpha_cost=float(cronbach(pd.concat([z(d.loneliness), -z(d.friends_have)], axis=1).dropna().values)),
    F=float(F2), df1=3, df2=int(len(allx) - 4), p=float(p2), eta2=float(ssb / sst),
    means={s: float(x.mean()) for s, x in zip(ORDER, gD2)},
    sds={s: float(x.std(ddof=1)) for s, x in zip(ORDER, gD2)},
    ns={s: int(len(x)) for s, x in zip(ORDER, gD2)},
    dismissing_vs_secure=dict(t=float(sd_pair.statistic), p=float(sd_pair.pvalue),
                              g=gg2, g_lo=gl2, g_hi=gh2))
# the profile interaction, stated as a 2 x 2 mixed design on standardised scores
sub = d[d.attachment_style.isin(["Secure", "Dismissing"])].dropna(subset=["fne", "loneliness"])
within = pd.DataFrame({"distress": z(d.fne)[sub.index], "cost": z(d.loneliness)[sub.index],
                       "style": sub.attachment_style.astype(str)})
diff = within.cost - within.distress
t_int, p_int = stats.ttest_ind(diff[within.style == "Dismissing"],
                               diff[within.style == "Secure"], equal_var=False)
gi, gil, gih = hedges_g(diff[within.style == "Dismissing"].values,
                        diff[within.style == "Secure"].values)
R["dissociation"]["profile_interaction"] = dict(
    description="Secure vs Dismissing x (z-loneliness minus z-FNE)",
    t=float(t_int), p=float(p_int), g=gi, g_lo=gil, g_hi=gih,
    m_dismissing=float(diff[within.style == "Dismissing"].mean()),
    m_secure=float(diff[within.style == "Secure"].mean()),
    n_dismissing=int((within.style == "Dismissing").sum()),
    n_secure=int((within.style == "Secure").sum()))

d.to_csv(f"{OUT}/analytic_with_derived.csv", index=False)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)

print("=== A. recognition vs report ===")
for k, v in rec.items():
    print(f" {k:<20} R2dim={v['r2_dimensions']:.3f} R2cat={v['r2_category']:.3f} both={v['r2_both']:.3f}"
          f" | diff={v['diff']:+.3f} CI[{v['diff_lo']:+.3f},{v['diff_hi']:+.3f}]"
          f" | cvDim={v['cv_r2_dimensions']:+.3f} cvCat={v['cv_r2_category']:+.3f}"
          f" | F={v['F_category_over_dimensions']:.2f} p={v['p_category_over_dimensions']:.5f}")
print("=== prototype validity ===")
for k in ("anxiety", "avoidance"):
    v = R["prototype_validity"][k]
    print(f" {k:<10} theory-high {v['m_theory_high']:.2f} vs low {v['m_theory_low']:.2f} t={v['t']:+.2f} p={v['p']:.4f} g={v['g']:+.2f}")
print(" chose own highest-rated:", R["prototype_validity"]["chose_own_highest_rated"])
print("=== B. anxiety not a forecast ===")
b = R["anxiety_not_forecast"]
print(f" aggregate r={b['aggregate_r']:+.3f} CI{[round(x,3) for x in b['ci']]} p={b['aggregate_p']:.3f}"
      f" TOST(|r|<.30) p={b['tost_p_bound_030']:.5f}  max|r| per bid={b['max_abs_r']:.3f}")
print(f" cheapest={b['cheapest']['bid']} dearest={b['dearest']['bid']} bid-level rho={b['bid_level_rho']['rho']:+.2f} p={b['bid_level_rho']['p']:.3f}")
print("=== C. zero friends ===", {x['style']: round(x['pct'], 1) for x in R["zero_friends"]["by_style"]},
      "chi2 p", round(R["zero_friends"]["p"], 4), "trend z", round(R["zero_friends"]["trend_z"], 2),
      "p", round(R["zero_friends"]["trend_p"], 5), "OR F/S", round(R["zero_friends"]["or_fearful_vs_secure"], 2))
print("=== D. dissociation respecified ===",
      {k: round(v, 3) for k, v in R["dissociation"]["respecified"]["means"].items()},
      "F", round(R["dissociation"]["respecified"]["F"], 3), "p", round(R["dissociation"]["respecified"]["p"], 4),
      "alpha_cost", round(R["dissociation"]["respecified"]["alpha_cost"], 3))
print(" D vs S:", {k: round(v, 4) for k, v in R["dissociation"]["respecified"]["dismissing_vs_secure"].items()})
print(" profile interaction:", {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in R["dissociation"]["profile_interaction"].items() if k != "description"})
