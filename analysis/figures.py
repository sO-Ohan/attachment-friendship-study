#!/usr/bin/env python3
"""Vector figures for the paper. Every figure is written as PDF (for LaTeX) and
SVG (for the repository). Nothing is rasterised."""
import json, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")
BASE = "/mnt/storage/Accademics/2-2/psy101/final_project_paper/PAPER/analysis"
OUT, FIG = f"{BASE}/out", f"{BASE}/figures"
R = json.load(open(f"{OUT}/results.json"))
Q = json.load(open(f"{OUT}/fgd_results.json"))
d = pd.read_csv(f"{OUT}/analytic_with_derived.csv")
ORDER = ["Secure", "Preoccupied", "Dismissing", "Fearful"]

# A hand-picked print palette: petrol, burnt ochre, olive, claret. Chosen to
# separate by lightness as well as hue so the figures survive greyscale printing.
C = {"Secure": "#1F5673", "Preoccupied": "#C67B3E",
     "Dismissing": "#7D8C57", "Fearful": "#9E3B47"}
INK, MUTED, RULE, FAINT = "#22201E", "#6B6560", "#B9B2A6", "#E8E4DB"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.2,
    "axes.edgecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none",
    "legend.frameon": False, "axes.grid": False,
})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIG}/{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  wrote", name)


def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=FAINT, lw=0.6, zorder=0)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# ============================================================== FIG 2 headline
def fig_recognition():
    rec = R["recognition_vs_report"]
    keys = ["friends_have", "convo_started", "loneliness", "approach_inhibition", "fne"]
    lab = ["Close friends held", "Conversations initiated", "Loneliness",
           "Approach inhibition", "Fear of negative evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.09))
    y = np.arange(len(keys))[::-1]

    ax = axes[0]
    h = 0.26
    dim = [rec[k]["r2_dimensions"] for k in keys]
    cat = [rec[k]["r2_category"] for k in keys]
    both = [rec[k]["r2_both"] for k in keys]
    ax.barh(y + h, dim, h, color="white", edgecolor=INK, lw=0.8, hatch="/////", zorder=3,
            label="Two Likert dimensions")
    ax.barh(y, cat, h, color="#1F5673", edgecolor=INK, lw=0.5, zorder=3,
            label="Prototype chosen")
    ax.barh(y - h, both, h, color=FAINT, edgecolor=INK, lw=0.5, zorder=3, label="Both together")
    for i, k in enumerate(keys):
        p_ = rec[k]["p_category_over_dimensions"]
        ax.text(max(cat[i], both[i]) + .012, y[i] - h, stars(p_), ha="left", va="center",
                fontsize=7.6, color=INK if p_ < .05 else MUTED)
    grid(ax, "x")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("Variance explained  $R^2$")
    ax.set_xlim(0, .41)
    ax.legend(loc="lower right", handlelength=1.6, borderpad=0.3, labelspacing=0.38)
    ax.set_title("a  Two ways of measuring the same construct", loc="left",
                 fontweight="bold", pad=7)

    ax = axes[1]
    cvd = [rec[k]["cv_r2_dimensions"] for k in keys]
    cvc = [rec[k]["cv_r2_category"] for k in keys]
    ax.axvline(0, color=INK, lw=0.9, zorder=2)
    for i, yy in enumerate(y):
        ax.annotate("", xy=(cvc[i], yy), xytext=(cvd[i], yy), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                    shrinkA=3, shrinkB=3, mutation_scale=9))
        ax.scatter(cvd[i], yy, s=32, facecolor="white", edgecolor=INK, lw=0.9, zorder=4)
        ax.scatter(cvc[i], yy, s=32, color="#1F5673", edgecolor="white", lw=0.6, zorder=4)
    grid(ax, "x")
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Cross-validated $R^2$   (10-fold, 20 repeats)")
    ax.set_xlim(-0.13, 0.27)
    ax.set_ylim(-0.75, len(keys) - 0.25)
    ax.annotate("worse than predicting\nthe sample mean", xy=(-0.062, 4.0),
                xytext=(-0.125, 3.05), fontsize=6.8, color=MUTED, style="italic",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                connectionstyle="arc3,rad=-0.25"))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5.5,
                              label="Dimensions"),
                       Line2D([], [], marker="o", ls="", color="#1F5673", ms=5.5,
                              label="Prototype")],
              loc="lower right", handlelength=1, borderpad=0.3, labelspacing=0.38)
    ax.set_title("b  Held-out prediction", loc="left", fontweight="bold", pad=7)
    save(fig, "fig02_recognition_vs_report")


# ============================================================== FIG 3 profiles
def fig_profiles():
    VARS = ["anxiety", "avoidance", "fne", "scenario_anx", "loneliness",
            "friends_have", "convo_started", "approach_inhibition"]
    LAB = ["Attachment\nanxiety", "Attachment\navoidance", "Fear of neg.\nevaluation",
           "Anticipated\nanxiety", "Loneliness", "Close friends\nheld",
           "Conversations\ninitiated", "Approach\ninhibition"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(len(VARS))
    for s in ORDER:
        sub = d[d.attachment_style == s]
        m, lo, hi = [], [], []
        for v in VARS:
            vv = sub[v].dropna()
            z = (vv - d[v].mean()) / d[v].std(ddof=1)
            m.append(z.mean())
            ci = stats.t.interval(.95, len(z) - 1, z.mean(), stats.sem(z))
            lo.append(ci[0]); hi.append(ci[1])
        ax.errorbar(x, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
                    color=C[s], lw=1.5, marker="o", ms=4.5, capsize=2.2, elinewidth=0.8,
                    label=f"{s} ($n$ = {len(sub)})", zorder=4,
                    mec="white", mew=0.7)
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    for i in range(len(VARS)):
        a = R["anova"][[j["variable"] for j in R["anova"]].index(VARS[i])]
        ax.text(i, 1.06, stars(a["q"]), ha="center", fontsize=7.6,
                color=INK if a["q"] < .05 else MUTED)
    grid(ax)
    ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=7.4)
    ax.set_ylabel("Standardised score  ($z$, whole sample)")
    ax.set_ylim(-1.05, 1.22)
    ax.axvspan(3.5, 7.5, color="#F6F3EC", zorder=0)
    ax.text(5.5, -1.0, "outcomes", ha="center", fontsize=7.4, color=MUTED, style="italic")
    ax.text(1.5, -1.0, "reported disposition and affect", ha="center", fontsize=7.4,
            color=MUTED, style="italic")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=1.4)
    save(fig, "fig03_style_profiles")


# ============================================================== FIG 4 forecast
def fig_forecast():
    b = R["anxiety_not_forecast"]
    gapd = R["expectancy_gap"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.34))

    ax = axes[0]
    xs = [g["anx"] for g in gapd]; ys = [g["exp"] for g in gapd]
    ax.scatter(xs, ys, s=54, color="#1F5673", edgecolor="white", lw=0.9, zorder=5)
    sl, ic = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs) - .18, max(xs) + .18, 20)
    ax.plot(xr, sl * xr + ic, color=MUTED, lw=1.0, ls=(0, (5, 3)), zorder=3)
    off = {"Asking for an explanation": (9, -2), "Suggesting lunch": (-9, 7),
           "Joining a project team": (9, 3), "Naming an upset": (-9, -9),
           "Sitting together": (9, -9), "Asking about an outing": (-9, 5)}
    for g in gapd:
        ax.annotate(g["bid"], (g["anx"], g["exp"]),
                    textcoords="offset points", xytext=off.get(g["bid"], (7, 5)),
                    fontsize=7, color=INK,
                    ha="right" if off.get(g["bid"], (7, 5))[0] < 0 else "left")
    grid(ax, "both")
    ax.set_xlabel("Mean anticipated anxiety  (1–6)")
    ax.set_ylabel("Mean expected acceptance  (1–6)")
    ax.set_title("a  Between situations", loc="left", fontweight="bold", pad=7)
    ax.text(.97, .93, f"Spearman $\\rho$ = {b['bid_level_rho']['rho']:.2f}, "
                      f"$p$ = {b['bid_level_rho']['p']:.3f}\nacross the six situations",
            transform=ax.transAxes, fontsize=7.6, va="top", ha="right", color=INK)
    ax.set_xlim(1.95, 3.95); ax.set_ylim(3.18, 4.52)

    ax = axes[1]
    per = b["per_bid"]
    y = np.arange(len(per))[::-1]
    ax.axvline(0, color=INK, lw=0.8, zorder=3)
    ax.axvspan(-.30, .30, color="#F6F3EC", zorder=0)
    for i, g in enumerate(per):
        n = g["n"]; r = g["r"]
        lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) / np.sqrt(n - 3))
        ax.plot([lo, hi], [y[i], y[i]], color=MUTED, lw=1.1, solid_capstyle="round", zorder=4)
        ax.scatter(r, y[i], s=26, color="#1F5673", edgecolor="white", lw=0.7, zorder=5)
    ra = b["aggregate_r"]
    lo, hi = b["ci"]
    ax.plot([lo, hi], [-1, -1], color=INK, lw=1.4, solid_capstyle="round", zorder=4)
    ax.scatter(ra, -1, s=42, marker="D", color=INK, zorder=5)
    grid(ax, "x")
    ax.set_yticks(list(y) + [-1])
    ax.set_yticklabels([g["bid"] for g in per] + ["All six averaged"], fontsize=7.4)
    for t in ax.get_yticklabels()[-1:]:
        t.set_fontweight("bold")
    ax.set_xlabel("Correlation between a student's anxiety and their expectation")
    ax.set_xlim(-.42, .52)
    ax.set_title("b  Between students, within each situation", loc="left",
                 fontweight="bold", pad=7)
    ax.set_ylim(-1.7, 5.6)
    ax.text(0, -1.62, "shaded band: equivalence bound $|r|$ = .30", fontsize=6.6,
            color=MUTED, ha="center", va="bottom", style="italic")
    save(fig, "fig04_anxiety_not_forecast")


# ============================================================== FIG 5 friends
def fig_friends():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    for i, s in enumerate(ORDER):
        v = d.loc[d.attachment_style == s, "friends_have"].dropna()
        j = (np.random.default_rng(7 + i).random(len(v)) - .5) * .34
        ax.scatter(i + j, v, s=13, color=C[s], alpha=.45, edgecolor="none", zorder=3)
        m = v.mean(); ci = stats.t.interval(.95, len(v) - 1, m, stats.sem(v))
        ax.plot([i - .26, i + .26], [m, m], color=INK, lw=1.6, zorder=5)
        ax.plot([i, i], ci, color=INK, lw=1.0, zorder=5)
        ax.text(i, 15.6, f"$n$ = {len(v)}", ha="center", fontsize=7, color=MUTED)
    grid(ax)
    ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, fontsize=8)
    ax.set_ylabel("Close friends at BRAC University")
    ax.set_ylim(-0.8, 16.6)
    a = R["anova"][[j["variable"] for j in R["anova"]].index("friends_have")]
    ax.set_title(f"a  $F$(3, {a['df2']}) = {a['F']:.2f}, $p$ < .001, "
                 f"$\\eta^2$ = {a['eta2']:.2f}", loc="left", fontweight="bold", pad=7)

    ax = axes[1]
    zf = R["zero_friends"]["by_style"]
    pct = [z["pct"] for z in zf]
    ax.bar(range(4), pct, .56, color=[C[s] for s in ORDER], edgecolor=INK, lw=0.5, zorder=3)
    for i, z in enumerate(zf):
        ax.text(i, z["pct"] + .9, f"{z['zero']}/{z['n']}", ha="center", fontsize=7.4, color=INK)
    grid(ax)
    ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, fontsize=8)
    ax.set_ylabel("Reporting zero close friends (%)")
    ax.set_ylim(0, 37)
    ax.axhline(R["zero_friends"]["overall_pct"], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=4)
    ax.text(3.42, R["zero_friends"]["overall_pct"] + .7, "whole sample",
            fontsize=6.8, color=MUTED, ha="right")
    ax.set_title(f"b  trend $z$ = {R['zero_friends']['trend_z']:.2f}, "
                 f"$p$ = {R['zero_friends']['trend_p']:.3f}", loc="left",
                 fontweight="bold", pad=7)
    save(fig, "fig05_friendship_outcomes")


# ============================================================== FIG 6 correlations
def fig_corr():
    cv = R["correlations"]
    V = cv["vars"]; L = ["Attachment anxiety", "Attachment avoidance", "Fear of neg. evaluation",
                         "Anticipated anxiety", "Expected acceptance", "Loneliness",
                         "Close friends held", "Friendship deficit", "Approach inhibition",
                         "Conversations initiated"]
    M = np.array([[np.nan if x is None else x for x in row] for row in cv["r"]])
    P = np.array(cv["p"])
    qmap = {(p["a"], p["b"]): p["q"] for p in cv["pairs"]}
    n = len(V)
    fig, ax = plt.subplots(figsize=(6.0, 5.3))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pc", ["#9E3B47", "#D9C7A8", "white",
                                                    "#AFBE93", "#1F5673"])
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            if i == j:
                ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor="#F1EDE4",
                                       edgecolor="white", lw=1))
                continue
            r = M[i, j]
            q = qmap.get((V[j], V[i]), qmap.get((V[i], V[j]), 1))
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   facecolor=cmap((r + 1) / 2), edgecolor="white", lw=1))
            ax.text(j + .5, n - 1 - i + .5, f"{r:.2f}".replace("0.", "."),
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(r) > .34 else INK,
                    fontweight="bold" if q < .05 else "normal")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + .5); ax.set_yticks(np.arange(n) + .5)
    ax.set_xticklabels(L, rotation=42, ha="right", fontsize=7.2)
    ax.set_yticklabels(L[::-1], fontsize=7.2)
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=.028, pad=.03, ticks=[-1, -.5, 0, .5, 1])
    cb.set_label("Pearson $r$", fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7.4, length=2)
    ax.text(0, -0.03, "Bold: survives Benjamini–Hochberg correction across all 45 tests "
                      "($q$ < .05).  $n$ = 101–103.",
            transform=ax.transAxes, fontsize=6.9, color=MUTED, va="top")
    save(fig, "fig06_correlations")


# ============================================================== FIG 7 FGD stance
def fig_stance():
    byq = {int(k): v for k, v in Q["by_question"].items()}
    order = sorted(byq, key=lambda q: (-byq[q]["pct_E"], q))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6),
                             gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.34))
    ax = axes[0]
    COL = {"E": "#1F5673", "D": "#C67B3E", "G": "#E4DFD4"}
    y = np.arange(len(order))[::-1]
    for i, q in enumerate(order):
        v = byq[q]; tot = v["n"]; left = 0
        for s in ("E", "D", "G"):
            w = 100 * v[s] / tot
            ax.barh(y[i], w, left=left, height=.66, color=COL[s],
                    edgecolor=INK, lw=0.4, zorder=3)
            if w > 11:
                ax.text(left + w / 2, y[i], f"{v[s]}", ha="center", va="center",
                        fontsize=7, color="white" if s != "G" else INK)
            left += w
    grid(ax, "x")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Q{q}{'  ●' if byq[q]['demands'] else ''}   ($n$ = {byq[q]['n']})"
                        for q in order], fontsize=7.6)
    ax.set_xlabel("Percentage of codable turns")
    ax.set_xlim(0, 100)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=COL["E"], ec=INK, lw=.4,
                                 label="First-person experiential"),
                       Rectangle((0, 0), 1, 1, fc=COL["D"], ec=INK, lw=.4,
                                 label="First-person dispositional"),
                       Rectangle((0, 0), 1, 1, fc=COL["G"], ec=INK, lw=.4,
                                 label="Generic / third-person")],
              ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              handlelength=1.1, labelspacing=0.3)
    ax.set_title("a  Stance, by moderator question", loc="left", fontweight="bold", pad=7)
    ax.text(0.5, -0.44, "●  question demands a personal instance",
            transform=ax.transAxes, fontsize=6.9, color=MUTED, ha="center", va="top")

    ax = axes[1]
    dp = Q["demand_paradox"]
    vals = [dp["demanding_pct"], dp["not_demanding_pct"]]
    ns = [dp["demanding_n"], dp["not_demanding_n"]]
    ks = [dp["demanding_E"], dp["not_demanding_E"]]
    bars = ax.bar([0, 1], vals, .5, color=["#9E3B47", "#1F5673"], edgecolor=INK,
                  lw=0.5, zorder=3)
    for i in range(2):
        lo, hi = stats.binomtest(ks[i], ns[i]).proportion_ci(.95)
        ax.plot([i, i], [100 * lo, 100 * hi], color=INK, lw=1.0, zorder=5)
        ax.text(i, 100 * hi + 1.6, f"{ks[i]}/{ns[i]}", ha="center", fontsize=7.4)
    grid(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Question asks for\na personal instance",
                        "Question does not"], fontsize=7.8)
    ax.set_ylabel("Turns that were first-person experiential (%)")
    ax.set_ylim(0, 62)
    ax.set_title(f"b  Fisher's exact $p$ = {dp['fisher_p']:.3f}, "
                 f"OR = {dp['odds_ratio']:.2f}", loc="left", fontweight="bold", pad=7)
    save(fig, "fig07_fgd_stance")


# ============================================================== FIG 8 situations
def fig_situations():
    S = R["situations"]["items"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    o = sorted(range(len(S)), key=lambda i: S[i]["pct"])
    y = np.arange(len(S))
    for k, i in enumerate(o):
        s = S[i]
        ax.plot([s["lo"], s["hi"]], [y[k], y[k]], color=RULE, lw=3.2,
                solid_capstyle="round", zorder=3)
        for st in ORDER:
            ax.scatter(100 * s["by_style"][st], y[k], s=26, color=C[st],
                       edgecolor="white", lw=0.7, zorder=5)
        ax.scatter(s["pct"], y[k], s=64, marker="|", color=INK, lw=1.5, zorder=6)
    grid(ax, "x")
    ax.set_yticks(y)
    ax.set_yticklabels([S[i]["situation"] for i in o], fontsize=8)
    ax.set_xlabel("Students naming the situation as one they avoid (%)")
    ax.set_xlim(0, 85)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C[s], ms=5.5, label=s)
                       for s in ORDER] +
                      [Line2D([], [], marker="|", ls="", color=INK, ms=8, mew=1.5,
                              label="Whole sample")],
              ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.26), columnspacing=1.2,
              handletextpad=0.4)
    ax.set_title(f"Cochran's $Q$ = {R['situations']['cochran_Q']:.1f}, $p$ < .001 across "
                 f"situations; no attachment-style difference on any single situation "
                 f"(all $p$ > .38)", loc="left", fontweight="bold", pad=7, fontsize=8.2)
    save(fig, "fig08_situations")


# ============================================================== FIG 9 dissociation
def fig_dissociation():
    pi = R["dissociation"]["profile_interaction"]
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    dd = d.copy(); dd["zf"] = z(dd.fne); dd["zl"] = z(dd.loneliness)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.32,
                                                                       width_ratios=[1, 1.1]))
    ax = axes[0]
    for s in ORDER:
        sub = dd[dd.attachment_style == s]
        mf, ml = sub.zf.mean(), sub.zl.mean()
        cf = stats.t.interval(.95, len(sub) - 1, mf, stats.sem(sub.zf.dropna()))
        cl = stats.t.interval(.95, len(sub) - 1, ml, stats.sem(sub.zl.dropna()))
        ax.plot([0, 1], [mf, ml], color=C[s], lw=1.7, marker="o", ms=5,
                mec="white", mew=0.8, zorder=4, label=s)
        ax.plot([0, 0], cf, color=C[s], lw=0.9, zorder=3)
        ax.plot([1, 1], cl, color=C[s], lw=0.9, zorder=3)
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    grid(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Fear of negative\nevaluation\n(what they report)",
                        "Loneliness\n(what they live with)"], fontsize=7.8)
    ax.set_xlim(-.34, 1.34)
    ax.set_ylabel("Standardised score ($z$)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30), columnspacing=1.2)
    ax.set_title("a  Profile across the two measures", loc="left", fontweight="bold", pad=7)

    ax = axes[1]
    for s, col in (("Secure", C["Secure"]), ("Dismissing", C["Dismissing"])):
        v = (dd.loc[dd.attachment_style == s, "zl"] -
             dd.loc[dd.attachment_style == s, "zf"]).dropna()
        pos = 0 if s == "Secure" else 1
        j = (np.random.default_rng(3).random(len(v)) - .5) * .3
        ax.scatter(pos + j, v, s=15, color=col, alpha=.5, edgecolor="none", zorder=3)
        ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
        ax.plot([pos - .24, pos + .24], [v.mean()] * 2, color=INK, lw=1.7, zorder=5)
        ax.plot([pos, pos], ci, color=INK, lw=1.1, zorder=5)
    ax.axhline(0, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    grid(ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Secure\n($n$ = 33)", "Dismissing\n($n$ = 32)"],
                                              fontsize=8)
    ax.set_xlim(-.5, 1.5)
    ax.set_ylabel("Loneliness minus fear of negative evaluation ($z$)")
    ax.set_title(f"b  $t$ = {pi['t']:.2f}, $p$ = {pi['p']:.4f}, "
                 f"Hedges' $g$ = {pi['g']:.2f}", loc="left", fontweight="bold", pad=7)
    save(fig, "fig09_dissociation")


# ============================================================== APPENDIX FIGURES
def fig_efa():
    e = R["efa_ecr"]
    L = np.array(e["loadings"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(wspace=0.34))
    ax = axes[0]
    ev = e["eigenvalues"]
    ax.plot(range(1, len(ev) + 1), ev, marker="o", color="#1F5673", ms=5, lw=1.3,
            mec="white", mew=0.8, zorder=4)
    ax.axhline(1, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    grid(ax)
    ax.set_xlabel("Component"); ax.set_ylabel("Eigenvalue")
    ax.set_xticks(range(1, len(ev) + 1))
    ax.set_title(f"a  KMO = {e['kmo']:.2f}, Bartlett $p$ < .001", loc="left",
                 fontweight="bold", pad=7)
    ax = axes[1]
    lab = e["items"]
    y = np.arange(len(lab))[::-1]
    ax.barh(y + .18, L[:, 0], .34, color="#1F5673", edgecolor=INK, lw=.4,
            label="Factor 1", zorder=3)
    ax.barh(y - .18, L[:, 1], .34, color="#C67B3E", edgecolor=INK, lw=.4,
            label="Factor 2", zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    grid(ax, "x")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7.6)
    ax.set_xlabel("Rotated loading (varimax)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    ax.set_title("b  Two-factor solution", loc="left", fontweight="bold", pad=7)
    save(fig, "figA1_efa")


def fig_clusters():
    X = d[["anxiety", "avoidance"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    ma, mv = d.anxiety.mean(), d.avoidance.mean()
    ax.axvline(ma, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.axhline(mv, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    for s in ORDER:
        sub = d[d.attachment_style == s]
        ax.scatter(sub.anxiety, sub.avoidance, s=22, color=C[s], alpha=.75,
                   edgecolor="white", lw=0.5, label=s, zorder=4)
    grid(ax, "both")
    ax.set_xlabel("Attachment anxiety (1–7)")
    ax.set_ylabel("Attachment avoidance (1–7)")
    for txt, xy in (("Fearful\nquadrant", (6.3, 6.6)), ("Secure\nquadrant", (1.4, 1.5)),
                    ("Dismissing\nquadrant", (1.4, 6.6)), ("Preoccupied\nquadrant", (6.3, 1.5))):
        ax.text(xy[0], xy[1], txt, fontsize=6.8, color=MUTED, ha="center", va="center",
                style="italic")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), columnspacing=1.2)
    ax.set_title("a  Where each self-selected group actually sits", loc="left",
                 fontweight="bold", pad=7)

    ax = axes[1]
    ct = pd.DataFrame(R["clusters"]["crosstab"]).reindex(
        index=["Secure-like", "Preoccupied-like", "Dismissing-like", "Fearful-like"],
        columns=ORDER).fillna(0)
    im = ax.imshow(ct.values, cmap="BuPu", aspect="auto", vmin=0)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = int(ct.values[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=8.5,
                    color="white" if v > 9 else INK,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(4)); ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=7.6)
    ax.set_yticks(range(4)); ax.set_yticklabels(ct.index, fontsize=7.6)
    ax.set_xlabel("Prototype the student chose")
    ax.set_ylabel("Cluster the scales put them in")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"b  ARI = {R['clusters']['ari']:.3f}; agreement "
                 f"{100*R['clusters']['agreement']:.0f}%", loc="left",
                 fontweight="bold", pad=7)
    save(fig, "figA2_clusters")


def fig_reliability():
    rel = R["reliability"]
    ks = ["anxiety", "avoidance", "scenario_anx", "scenario_expect", "fne", "loneliness"]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    y = np.arange(len(ks))[::-1]
    for i, k in enumerate(ks):
        r = rel[k]
        ax.plot([r["alpha_lo"], r["alpha_hi"]], [y[i], y[i]], color=MUTED, lw=1.3,
                solid_capstyle="round", zorder=3)
        ax.scatter(r["alpha"], y[i], s=34, color="#1F5673", edgecolor="white",
                   lw=0.7, zorder=5, label="Cronbach's $\\alpha$" if i == 0 else None)
        ax.scatter(r["omega"], y[i], s=30, marker="D", facecolor="white",
                   edgecolor="#9E3B47", lw=1.0, zorder=5,
                   label="McDonald's $\\omega$" if i == 0 else None)
    ax.axvline(.70, color="#9E3B47", lw=0.9, ls=(0, (4, 3)), zorder=2)
    grid(ax, "x")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{rel[k]['label']}  ({rel[k]['k']} items)" for k in ks], fontsize=7.6)
    ax.set_xlabel("Internal consistency")
    ax.set_xlim(.45, 1.0)
    ax.text(.70, len(ks) - .35, "conventional floor", fontsize=6.8, color="#9E3B47",
            ha="center", va="bottom", style="italic")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, -0.42))
    save(fig, "figA3_reliability")


if __name__ == "__main__":
    print("figures:")
    fig_recognition(); fig_profiles(); fig_forecast(); fig_friends(); fig_corr()
    fig_stance(); fig_situations(); fig_dissociation()
    fig_efa(); fig_clusters(); fig_reliability()
    print("done")
