# Attachment Style, Fear of Rejection, and Friendship Formation

Data, analysis code, and instruments for:

> **How Attachment Style Shapes Fear of Rejection and Friendship Formation Among
> BRAC University Undergraduates**
> Bhuiyan, M. M. H., Abdullah, K., Nahian, A., Labib, K. M., Hasan, T.,
> & Mahjabin, M. (2026).
> PSY 101 final project, Section 11, Group 02, BRAC University.

## What the study found

Undergraduates at BRAC University (*n* = 103) completed a survey that measured
attachment style two ways at once: by picking one of four prototype paragraphs
(Bartholomew & Horowitz, 1991), and by rating four-item attachment anxiety and
avoidance scales. Nine students then discussed a vignette in a focus group.

1. **Recognition beats report.** The paragraph a student pointed to predicted how
   many close friends they had (*R²* = .19); the two Likert dimensions predicted
   nothing (*R²* = .01). Under 10-fold cross-validation the dimensions scored
   *R²* = −.06, worse than predicting the sample mean, against +.13 for the
   prototype. The effect is specific to behavioural outcomes; for self-reported
   affect the scales did better.
2. **Anxiety is not a forecast.** Within each of six concrete friendship bids, a
   student's anticipated anxiety was uncorrelated with their expected acceptance
   (*r* = −.004 to .147). Across the six bids as units the relationship was
   strong (ρ = −.83). Situations are ranked coherently; individuals are not.
3. **Situations, not persons, determine avoidance.** Cochran's *Q* = 82.6,
   *p* < 1e-15 across six situations, with no attachment-style difference on any
   single one.
4. **The room would not speak in the first person.** Focus-group questions
   written to extract a personal instance produced first-person disclosure in
   6.7% of turns; questions that asked for nothing of the kind produced 34.6%
   (Fisher's exact *p* = .006).

## Layout

| Path | Contents |
|---|---|
| `data/` | De-identified responses, the analytic sample, the codebook, and the cleaning audit |
| `analysis/` | The five Python scripts that reproduce every number and figure |
| `figures/` | All eleven figures as vector PDF |
| `fgd/` | The pseudonymised focus-group transcript, coded one row per turn |
| `instruments/` | The focus group moderator guide; survey items are in Appendix F of the report |
| `paper/` | The report |

## Reproducing

```bash
pip install pandas numpy scipy matplotlib openpyxl
python analysis/stats_core.py
python analysis/stats_model.py
python analysis/stats_extra.py
python analysis/fgd_coding.py
python analysis/figures.py
```

The scripts use absolute paths from the machine they were written on; change the
`BASE`, `SRC`, and `OUT` constants at the top of each file. The random seed is
fixed at 20260827 and every bootstrap uses 10,000 resamples, so the numbers
reproduce exactly.

Nothing here depends on statsmodels or pingouin. Bootstrap intervals for α,
McDonald's ω, principal-axis factoring with varimax rotation, Welch and
Brown–Forsythe tests, Games–Howell, Pillai's trace, k-means with the adjusted
Rand index, Cochran's Q, logistic regression, and the power calculations are all
implemented directly in `stats_core.py`.

## Privacy

Participation was voluntary and anonymous. The raw Google Forms export is **not**
published, because 43 respondents gave an optional name and 22 gave an optional
phone number or email address; that file never leaves the research team. Focus
group participants appear as P1–P9, and a third party named in passing during the
session has been replaced. The real-name mapping is not in this repository.

## Licence

Data and text: CC BY 4.0. Code: MIT. Please cite the archived release.
