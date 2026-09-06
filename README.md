# ML Driven Mortgage Prepayment and MBS Cash-Flow Analytics

**Independent portfolio case study using public Freddie Mac Single-Family Loan-Level Data (SFLLD) and Freddie Mac PMMS mortgage rates**

This project asks a mortgage-finance question rather than a generic classification question:

**Can borrower-level machine-learning models improve our understanding and forecasting of mortgage prepayments and how do those changing prepayment expectations affects the timing of cash flows received by MBS investor?**

The implementation code and development notebooks are intentionally not published in this repository. This public version focuses on the **problem framing, methodology, validation, mortgage interpretation, results, limitations and investor implications**

[**Read the full 7 page case study (PDF)**](....)

---

## Project at a glance
| Area | Result |
|---|---:|
| Longitudinal sample | **150,000 mortgages / 6.42M loan-months** |
| Final locked test period | **Jan 2025 - Feb 2026** |
| Test observations / events | **1,642,132 / 11,378** |
| LightGBM test ROC-AUC | **0.701** |
| LightGBM test PR-AUC | **0.026** |
| Top risk decile | **3.5x population lift; 35.1% of realized events** |
| Pool-level CPR MAE | **1.05 percentage points** |
| Average CPR forecast bias | **+0.01 percentage points** |
| Principal returned after 24 months | **50.0% high-refi vs 19.0% benchmark vs 13.8% low-refi** |

![End-to-end analytical flow]
<img width="1152" height="85" alt="00_workflow" src="https://github.com/user-attachments/assets/f4c4e333-51d3-4036-aaf4-e48189961aa1" />

## 1. Business problem
Mortgage investors do not fully control when principal is returned. Borrowers can refinance, sell or pay off their loans before contractual maturity.
This creates a risk for mortgage-backed securities:

- **Faster prepayment** returns principal sooner, reduces future interest income, and creates reinvestment / contraction risk.
- **Slower prepayment** keeps principal outstanding longer and creates extension exposure.

A useful prepayment model therefore needs to do more than rank risk borrowers. Its probabilities ultimately affect forecasts of **pool SMM/CPR and projected cashflows**.

---

## 2. Data and target constructions

### Public data

The project uses:

- Freddie Mac Single-Family Loan-Level Dataset origination and monthly performance data
- From 2020 to 2023 origination vintages across 16 quarters
- Freddie Mac PMMS 30-year mortgage rates

### Loan-level sampling

I sampled **9,375 unique mortgages from each of the 16 origination quarters**, producing **150,000 unique loans**. Sampling was performed at the mortgage level and the complete observed monthly history of each selected mortgage was retained.

This produced **6,418,477 monthly performance records** spanning January 2020 through March 2026.



