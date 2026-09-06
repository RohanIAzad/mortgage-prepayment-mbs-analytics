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

End-to-end analytical flow

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

## Prediction target

The primary target is **next month voluntary prepayment**

For each eligible active loan-month:

- `1` = the mortgage voluntarily prepays in the following month
- `0` = it remnains eligible without voluntary prepayment in the following month
- termination month observations and insufficiently oberved future periods are not treated as ordinary negative observations.
- The months on which voluntary payment occcurs are ommited from the observation because the target is 1 month prepayment.

<img width="242" height="537" alt="image" src="https://github.com/user-attachments/assets/4d696766-cf19-4f14-9fe8-b7080262d55c" />

This picture illustrates the structure of the target variable.

For a loan that originated in April and assuming currently we are in July, predicting if the loan will prepay in the month of August. In August will predict if the loan will prepay in September if it has not already prepaid in August.Some of teh input features use events from the last 3 months such as if there was any 30 plus days of delinquency in the last 3 months (any_30plus_dq_last_3m), That is why t-1, t-2, t-3 is there.

The feature panel contains approximately **6.27M eligible next month observations and 38,374 positive events**.

features include- 
borrower characteristics, loan structure, current balance/rate, seasoning, delinquency history, market rates, refinance incentive, and experimental burnout proxies.

--

## 3. Borrower behavior: refinance incentive

A core mortgage feature is:

**Refinance incentive = current borrower mortgage rate - prevailing PMMS rate**

Positive values mean the borrower's existing coupon is above the prevailing market mortgage rate, making refinancing more economically attractive.

Observed prepayment by refinance incentive

<img width="1262" height="365" alt="image" src="https://github.com/user-attachments/assets/76e966b6-71f7-4069-b36d-28d95a224080" />

Observed next-month voluntary prepayment rises sharply with refinance incentive:

- `< -300 bps`: approximately **0.35%**
- `0 to 50 bps`: approximately **0.95%**
- `100 to 150 bps`: approximately **2.46%**
- `150 to 200 bps`: approximately **3.10%**

## 4. Modeling and out-of-time validation

I compared an interpretable Logistic regression benchmark with LightGBM and XGBoost.

The project uses a **calendar-time split rather than random loan-month split**:

| Partition | Reporting period | Use |
|---|---|---|
| Train | Through Dec 2023 | Model fitting |
| Validation | Jan-Dec 2024 | Model selection / tuning |
| Locked test | Jan 2025-Feb 2026 | Final untouched evaluation |

The same surviving mortgage can appear in multiple calendar partitions as it ages. Loan ID is not a model feature. This design evaluates **future-period portfolio forecasting*.

The final LightGBM model achieved on the locked test set:

- **ROC-AUC: 0.701**
- **PR-AUC: 0.026**
- **Actual event rate: 0.693%**
- **Mean predicted probability: 0.730%**

Out-of-time risk decile validation:

<img width="662" height="392" alt="image" src="https://github.com/user-attachments/assets/ed1d3b8d-cc27-4140-8a95-18f3a16de762" />

The highest predicted-risk decile experienced about **3.5x the population-average prepayment rate** and contained about **35% of all realized test prepayments**.

Model selection emphasized **probability quality, downstream pool usefulness**, not ROC-AUC alone.

---

## 5. What drives the model?

SHAP analysis shows that the strongest global model driver is **refinance incentive**, followed by **loan age / seasoning** and **PMMS**

<img width="657" height="596" alt="image" src="https://github.com/user-attachments/assets/adac4344-b15f-429e-94ea-8a869f203bc5" />

SHAP is used here to explain model behavior, not causality.

---

## 6. From loan probabilities to pool SMM and CPR

For each reporting month, loan-level probabilities are aggregated using current UPB weights:

**Pool SMM(t) = sum[UPB(i,t) x p(i,t)] / sum[UPB(i,t)]**

Then:

**CPR(t) = 1 - (1 - SMM(t))^12**

UPB weighting is important because a $500,000 mortgage and a $50,000 mortgage should not contribute equally to expected pool principal runoff.

Three comparison pools were used:

- **Benchmark:** full eligible test population
- **High Refi Incentive:** refinance incentive >= 100 bps
- **Low Refi Incentive:** refinance incentive <= 0 bps

Predicted VS realized CPR:

<img width="665" height="395" alt="image" src="https://github.com/user-attachments/assets/fd587859-a5e0-48bb-8222-d7a4f3bcdf00" />

Average holdout-period results:

| Pool | Predicted CPR | Realized CPR |
|---|---:|---:|
| Benchmark | **8.03%** | **8.02%** |
| High Refi Incentive | **28.59%** | **29.62%** |
| Low Refi Incentive | **4.95%** | **4.97%** |

Across monthly holdout forecasts, CPR MAE was approximately **1.05 percentage points**, RMSE **1.43 pp**, and average bias **+0.01 pp**.


