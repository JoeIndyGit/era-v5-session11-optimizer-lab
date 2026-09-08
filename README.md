# Session 11 — Optimizers & Learning-Rate Schedules

> **A reproducible optimizer study, not a black-box training run.**  
> This submission reproduces Adam from first principles, isolates bias correction, instruments update-to-weight ratios layer by layer, compares independently tuned cosine and WSD schedules under an early-stop interruption, and empirically transfers learning rate across width.

---

## Executive summary

| Question | Result | Evidence |
|---|---:|---|
| Can I reproduce Adam exactly? | **Yes — max abs. error = 4.12×10⁻¹⁷** | `artifacts/01_adam_verification.csv` |
| When does bias correction stop mattering? | **Step 3916** under a sustained **<1%** multiplier-gap criterion | `02_bias_correction_*` |
| When does warmup stop directly changing U/W? | Warmup's last LR increase is **step 20**; **step 21** is the first plateau step | `03_update_weight_*` |
| At a forced stop at step 200, which tuned schedule wins? | **Cosine** on validation loss: **1.1036 vs 1.1422** | `04_schedule_tuning.csv`, `04_cosine_vs_wsd_curve.csv` |
| Best LR at width 256? | Discrete **0.008**; quadratic basin estimate **0.00868** | `05_width_lr_*` |
| Best LR at width 512? | Discrete **0.008**; quadratic basin estimate **0.00695** | `05_width_lr_*` |
| Best LR at width 1024? | Discrete **0.006**; quadratic basin estimate **0.00593** | `05_width_lr_*` |
| LR I would try first at width 4096 | **≈ 0.0040** | power-law fit: LR ∝ width⁻⁰·²⁷⁵ |
| Confidence in width-4096 extrapolation | **Moderate** | 3 widths × 3 seeds, but 4096 is still out-of-range extrapolation |

### One-sentence conclusion

The experiments show that the optimizer's early behavior is strongly shaped by bias correction and warmup, schedule comparisons can reverse when each schedule is tuned fairly, and the best learning rate in this parameterization drifts downward with width rather than remaining perfectly transferable.

---

# 1. Experimental philosophy

The assignment warning is the governing rule of this repository:

> **Tune both sides before accepting a comparison.**

Every comparison therefore follows four controls:

1. **Same task** — a fixed teacher-generated classification problem.
2. **Same initialization rule** — deterministic PyTorch seed before every run.
3. **Same mini-batch stream** — fixed generated index sequence.
4. **One manipulated variable at a time** — bias correction, schedule, warmup stage, width, or LR.

This matters because an optimizer experiment is easy to make visually convincing and scientifically wrong. If one run gets luckier initialization, different batches, a better LR, or a different horizon, the plot may attribute that advantage to the optimizer or schedule.

---

# 2. Reproducibility contract

## Environment

- Python 3.10+
- PyTorch
- NumPy
- pandas
- matplotlib
- pytest

Install:

```bash
python -m pip install -r requirements.txt
```

Run every experiment:

```bash
python run_all.py
```

Run tests:

```bash
pytest -q
```

Run one experiment:

```bash
python -m experiments.exp01_adam_by_hand
python -m experiments.exp02_bias_correction
python -m experiments.exp03_update_weight_ratio
python -m experiments.exp04_cosine_vs_wsd
python -m experiments.exp05_width_lr_sweep
```

All plots and raw tables are written to `artifacts/`.

---

# 3. Common model and task

For the model-training experiments I use a deterministic **teacher → student** classification benchmark.

### Teacher dataset

- Input dimension: **64**
- Teacher hidden dimension: **128**
- Classes: **20**
- Training examples: **10,000**
- Validation examples: **2,000**
- Teacher seed: fixed

The teacher creates labels with a frozen nonlinear network. This has two benefits:

- no network downloads or dataset-version drift;
- the training problem is non-trivial but fully reproducible.

### Student model

```text
64-dimensional input
      ↓
Linear(64 → width)
      ↓
GELU
      ↓
Linear(width → 20)
      ↓
Cross-entropy
```

Only `width` changes in Experiment 5.

### Shared optimizer settings

- AdamW unless the experiment explicitly studies Adam internals
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
- weight decay = 0.01 for model-training experiments
- batch size = 128
- warmup = 20 steps unless otherwise stated

---

# 4. Experiment 1 — Reproduce Adam by hand

## Goal

Take one scalar weight and five gradients, calculate Adam's internal state manually, and compare every quantity against PyTorch — not just the final weight.

### Setup

```text
initial weight = 1.0
gradients      = [0.10, -0.20, 0.05, -0.10, 0.15]
lr             = 0.001
β1             = 0.9
β2             = 0.999
ε              = 1e-8
```

## Equations

First moment:

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]

Second raw moment:

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

Bias corrections:

\[
\hat m_t = \frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t = \frac{v_t}{1-\beta_2^t}
\]

Signed parameter step:

\[
\Delta w_t = -\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

Parameter update:

\[
w_t = w_{t-1} + \Delta w_t
\]

## Manual results

| t | g | m | v | m̂ | v̂ | signed step | weight after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.010000 | 0.00001000 | 0.100000 | 0.010000 | -0.00100000 | 0.99900000 |
| 2 | -0.20 | -0.011000 | 0.00004999 | -0.057895 | 0.025008 | +0.00036610 | 0.99936610 |
| 3 | 0.05 | -0.004900 | 0.00005244 | -0.018081 | 0.017497 | +0.00013669 | 0.99950279 |
| 4 | -0.10 | -0.014410 | 0.00006239 | -0.041902 | 0.015620 | +0.00033526 | 0.99983806 |
| 5 | 0.15 | 0.002031 | 0.00008483 | 0.004960 | 0.016999 | -0.00003804 | 0.99980002 |

## PyTorch verification

The code inspects PyTorch's optimizer state after **every step**:

- `exp_avg` → `m`
- `exp_avg_sq` → `v`
- bias-corrected values reconstructed from state
- actual parameter delta → optimizer step

### Result

**Maximum absolute error across all compared quantities: `4.1220e-17`.**

That is far beyond “several decimal places”; the manual and PyTorch calculations agree to floating-point noise.

Raw proof: [`artifacts/01_adam_verification.csv`](artifacts/01_adam_verification.csv)

### Interpretation

Adam is not “momentum plus RMS scaling” in a vague sense. At each step it maintains two exponentially weighted histories, explicitly debiases their zero initialization, and uses their ratio to determine the update.

---

# 5. Experiment 2 — Disable bias correction

## Question

What actually happens if the `1 - βᵗ` correction terms are removed?

The custom optimizer `AdamNoBiasCorrection` changes exactly one thing: it uses raw `m_t` and `v_t` instead of `m̂_t` and `v̂_t`.

The first 20 steps repeat the same five-gradient stream used in Experiment 1, linking the two experiments directly.

## Why the early error is large

Without bias correction, the effective update contains the initialization bias from both EMAs. Ignoring ε, the relationship between the corrected and uncorrected adaptive term is controlled by:

\[
R_t = \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
\]

Crucially, this multiplicative correction depends on **t and the β values**, not on a lucky choice of gradient stream.

## Defining “stops mattering”

I do not eyeball the graph. I define a material effect as:

\[
|R_t - 1| \ge 1\%
\]

To avoid a one-step accidental crossing, the criterion must stay below 1% for **20 consecutive steps**.

### Result

**Bias correction first becomes sustainably <1% material at step 3916.**

That result is much later than the 20-step plot — which is exactly the point. The first 20 steps visualize the dramatic transient; the longer analytical check answers when the correction genuinely becomes negligible under the chosen threshold.

### Artifacts

- `artifacts/02_bias_correction_first20.csv`
- `artifacts/02_bias_correction_trajectory.png`
- `artifacts/02_bias_correction_gap.png`

![Bias correction trajectory](artifacts/02_bias_correction_trajectory.png)

![Bias correction gap](artifacts/02_bias_correction_gap.png)

### Interpretation

For β₂ = 0.999, the second-moment estimator has a very long memory. Removing bias correction does **not** merely perturb step 1; it changes the optimizer's effective scale for a substantial early portion of training.

---

# 6. Experiment 3 — Update-to-weight ratio for every layer

## Metric

For each parameter tensor:

\[
R_t = \frac{\|\Delta W_t\|_2}{\|W_t\|_2 + \epsilon}
\]

I also aggregate parameters by layer using norms, not by averaging unrelated ratios:

\[
R_{\text{layer}}
=
\frac{\sqrt{\sum_i \|\Delta W_i\|_2^2}}
{\sqrt{\sum_i \|W_i\|_2^2}+\epsilon}
\]

This prevents a tiny bias vector from receiving the same statistical weight as a large weight matrix.

## Instrumentation

For every step and parameter, the code logs:

```text
step
parameter
layer
learning rate
weight norm
update norm
update-to-weight ratio
```

Nothing is inferred after the fact; the parameter tensors are cloned immediately before `optimizer.step()` and compared with the post-step tensors.

## Warmup result

Warmup is linear for 20 steps:

\[
\eta_t = \eta_{max}\frac{t}{20}, \qquad t\le20
\]

Then LR becomes flat.

Therefore:

- **step 20** = final step on which warmup increases LR;
- **step 21** = first step with no warmup-induced LR increase.

Observed layer ratios:

| Layer | Step 20 U/W | Step 21 U/W |
|---|---:|---:|
| `fc1` | 0.01846 | 0.01817 |
| `fc2` | 0.04871 | 0.04500 |

The ratio does **not** become constant after warmup — gradients and optimizer state continue to evolve. The narrower claim is the correct one: after the transition to step 21, **warmup itself no longer changes the LR multiplier**.

### Artifacts

- `artifacts/03_update_weight_parameter.csv`
- `artifacts/03_update_weight_layer.csv`
- `artifacts/03_update_to_weight_ratio.png`
- `artifacts/03_warmup_lr.png`

![Update-to-weight ratio](artifacts/03_update_to_weight_ratio.png)

### Interpretation

Warmup controls update scale during the optimizer's most statistically fragile phase. U/W is useful because the same absolute update can be tiny for one layer and large for another.

---

# 7. Experiment 4 — Cosine vs WSD

## Fair-comparison rule

Both schedules use:

- the same student architecture;
- identical initialization seed;
- identical batch sequence;
- identical optimizer family and weight decay;
- the same 300-step planned horizon;
- the same 20-step warmup.

**The peak learning rate is tuned independently for each schedule.** This is the key control requested by the assignment warning.

### Schedule definitions

**Cosine:** warmup, then cosine decay across the remaining planned horizon.

**WSD:** warmup → stable LR → terminal decay. Here the stable phase lasts until step 240, then a linear decay finishes at step 300.

## Independent tuning

Candidate peak LRs:

```text
0.0015, 0.0020, 0.0030, 0.0040, 0.0055, 0.0070
```

Each candidate is evaluated at the assignment's interruption point, step 200.

Selected values:

| Schedule | tuned peak LR |
|---|---:|
| Cosine | **0.0055** |
| WSD | **0.0030** |

## Forced stop at step 200

| Schedule | train loss @200 | validation loss @200 | LR @200 |
|---|---:|---:|---:|
| Cosine | **0.7154** | **1.1036** | ~0.00199 |
| WSD | 0.8382 | 1.1422 | 0.00300 |

### Which checkpoint do I keep?

**I keep the cosine checkpoint at step 200.**

It has the lower training loss **and** the lower held-out validation loss under independently tuned configurations.

That is an empirical statement about this run and interruption point — not a claim that cosine is universally superior to WSD.

## What happens by step 300?

| Schedule | train loss @300 | validation loss @300 |
|---|---:|---:|
| Cosine | 0.6095 | **1.0597** |
| WSD | **0.6061** | 1.0797 |

WSD catches up strongly after its decay begins and edges cosine on training loss, while cosine remains slightly better on held-out loss in this setup.

That distinction is precisely why schedule claims need a specified horizon and metric.

### Artifacts

- `artifacts/04_schedule_tuning.csv`
- `artifacts/04_cosine_vs_wsd_curve.csv`
- `artifacts/04_cosine_vs_wsd_lr.png`
- `artifacts/04_cosine_vs_wsd_val_loss.png`

![Cosine vs WSD LR](artifacts/04_cosine_vs_wsd_lr.png)

![Cosine vs WSD validation loss](artifacts/04_cosine_vs_wsd_val_loss.png)

---

# 8. Experiment 5 — Learning-rate sweep across width

## Goal

Find the LR basin at widths:

- 256
- 512
- 1024

Then predict the first LR to try at width 4096.

## Two-stage sweep

A single narrow grid can lie by putting the minimum at an edge. I therefore use:

### Stage A — coarse search

```text
0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032
```

This locates the basin.

### Stage B — refined search

```text
0.004, 0.005, 0.006, 0.008, 0.010, 0.012
```

Each refined point is run over **three seeds: 41, 42, 43**.

The plotted value is mean validation loss with standard deviation error bars.

## Refined discrete minima

| Width | best sampled LR |
|---:|---:|
| 256 | **0.008** |
| 512 | **0.008** |
| 1024 | **0.006** |

A discrete grid only says “best among values I happened to test.” To estimate the basin minimum more smoothly, I fit a quadratic to validation loss as a function of `log(LR)` around the basin.

## Quadratic basin estimates

| Width | fitted LR minimum |
|---:|---:|
| 256 | **0.008684** |
| 512 | **0.006952** |
| 1024 | **0.005931** |

These values show a smooth downward drift as width increases.

![Width LR sweep](artifacts/05_width_lr_sweep.png)

## Width → LR scaling fit

I fit:

\[
\eta^*(d)=C d^\alpha
\]

Taking logs:

\[
\log \eta^*=\log C+\alpha\log d
\]

The fitted exponent is:

\[
\alpha = -0.2751
\]

So in this experiment:

\[
\eta^* \propto d^{-0.275}
\]

Extrapolating to width 4096 gives:

\[
\boxed{\eta^*_{4096}\approx0.00401}
\]

### Value I would use at width 4096

**I would start at `4e-3`.**

I would then bracket it locally, for example with nearby values around the prediction rather than pretending the extrapolation is exact.

### Confidence: moderate

Why not high?

1. We measured only three widths.
2. Width 4096 is four times larger than the largest measured width.
3. The scaling law is empirical and specific to this model parameterization, task, optimizer and training budget.
4. A quadratic basin fit adds resolution but does not create new observations.

Why not low?

1. The refined sweep uses three seeds per point.
2. The minima move smoothly rather than jumping randomly.
3. The coarse sweep verifies the basin is not an artifact of an obviously clipped search range.

![Width LR extrapolation](artifacts/05_width_lr_extrapolation.png)

### Artifacts

- `artifacts/05_width_lr_coarse.csv`
- `artifacts/05_width_lr_fine_all_seeds.csv`
- `artifacts/05_width_lr_fine_summary.csv`
- `artifacts/05_width_lr_minima.csv`
- `artifacts/05_width_lr_sweep.png`
- `artifacts/05_width_lr_extrapolation.png`

---

# 9. What the assignment taught me

## 9.1 Bias correction is not cosmetic

The first- and second-moment EMAs begin at zero. Their early estimates are therefore biased toward zero, and β₂ = 0.999 makes the second-moment transient especially long. The correction terms are mathematically part of Adam's intended update, not a numerical flourish.

## 9.2 Warmup is visible in parameter space

Learning rate is a scalar hyperparameter, but its practical effect is layer dependent. Logging U/W makes that visible: the same LR schedule produces different relative movement across layers.

## 9.3 Schedule rankings are horizon-dependent

At step 200, WSD has not yet entered decay, while cosine has already reduced LR substantially. The forced interruption therefore evaluates different points in each schedule's intended lifecycle. This is not unfair — it is the scenario the assignment asks us to analyze — but it must be stated explicitly.

## 9.4 “Best LR” is a basin, not a magic scalar

The refined sweep and seed variance show that nearby LRs can be statistically similar. Reporting only one winning scalar without the surrounding curve creates false precision.

## 9.5 Hyperparameter transfer is empirical

The width-4096 LR is an extrapolation, not a fact. The right scientific statement is: **4e-3 is the first value I would test, with moderate confidence, because it follows the measured trend.**

---

# 10. Beyond-the-rubric checks

These are deliberate extensions beyond the minimum assignment requirements.

### Numerical verification

- float64 scalar Adam comparison
- assertion on max manual-vs-PyTorch error
- optimizer state inspected directly

### Bias-correction materiality rule

- explicit 1% threshold
- must hold for 20 consecutive steps
- analytical correction multiplier separates bias correction from gradient luck

### U/W observability

- parameter-level logs
- layer-level norm aggregation
- pre/post optimizer snapshots rather than inferred updates

### Fair schedule comparison

- same planned horizon
- same model initialization
- same batches
- same optimizer/weight decay
- independent LR tuning for each schedule
- training **and** validation loss reported

### Width-sweep robustness

- coarse + refined search
- 3 random seeds per refined point
- error bars
- continuous log-LR basin fit
- power-law extrapolation
- explicit uncertainty statement

### Software quality

- experiment modules are independent
- raw CSVs are retained
- plots regenerate from code
- `pytest` tests key invariants
- one-command runner
- notebook included for interactive review

---

# 11. Repository structure

```text
session11_optimizer_assignment/
│
├── README.md
├── requirements.txt
├── run_all.py
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── model.py
│   ├── optimizers.py
│   ├── schedules.py
│   └── utils.py
│
├── experiments/
│   ├── exp01_adam_by_hand.py
│   ├── exp02_bias_correction.py
│   ├── exp03_update_weight_ratio.py
│   ├── exp04_cosine_vs_wsd.py
│   └── exp05_width_lr_sweep.py
│
├── tests/
│   └── test_assignment.py
│
├── notebooks/
│   └── session11_assignment.ipynb
│
└── artifacts/
    ├── 01_adam_verification.csv
    ├── 02_bias_correction_first20.csv
    ├── 02_bias_correction_trajectory.png
    ├── 02_bias_correction_gap.png
    ├── 03_update_weight_parameter.csv
    ├── 03_update_weight_layer.csv
    ├── 03_update_to_weight_ratio.png
    ├── 03_warmup_lr.png
    ├── 04_schedule_tuning.csv
    ├── 04_cosine_vs_wsd_curve.csv
    ├── 04_cosine_vs_wsd_lr.png
    ├── 04_cosine_vs_wsd_val_loss.png
    ├── 05_width_lr_coarse.csv
    ├── 05_width_lr_fine_all_seeds.csv
    ├── 05_width_lr_fine_summary.csv
    ├── 05_width_lr_minima.csv
    ├── 05_width_lr_sweep.png
    └── 05_width_lr_extrapolation.png
```

---

# 12. Rubric traceability

| Assignment requirement | Where it is satisfied |
|---|---|
| One weight + five gradients | Experiment 1 |
| Compute `m`, `v`, `m̂`, `v̂`, step manually | Experiment 1 table + code |
| Check each against PyTorch | `01_adam_verification.csv` + assertion |
| Disable bias correction | `AdamNoBiasCorrection` |
| Plot first 20 steps both ways | `02_bias_correction_trajectory.png` |
| Report when difference stops mattering | step **3916**, explicit 1% criterion |
| Log update-to-weight ratio every layer | Experiment 3 CSV + plot |
| Identify warmup transition | step 20 → first flat step 21 |
| Train cosine and WSD | Experiment 4 |
| Planned 300-step schedules | Experiment 4 |
| Evaluate forced stop at 200 | Experiment 4 table |
| State which model to keep | **cosine** |
| Sweep LR at 256/512/1024 | Experiment 5 |
| Plot loss against LR | `05_width_lr_sweep.png` |
| Mark minima | discrete + quadratic minima |
| State LR at 4096 | **≈0.0040** |
| State confidence | **moderate**, with reasons |
| Tune both sides | independent schedule LR grids |
| Detailed README + support code | this repository |

---

# 13. References

1. D. P. Kingma and J. Ba, **Adam: A Method for Stochastic Optimization**, ICLR 2015. https://arxiv.org/abs/1412.6980
2. PyTorch, **torch.optim.Adam documentation**. https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
3. I. Loshchilov and F. Hutter, **Decoupled Weight Decay Regularization**. https://arxiv.org/abs/1711.05101
4. K. Wen et al., **Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective**. https://arxiv.org/abs/2410.05192

---

# Final answer to the assignment

- **Adam reproduced by hand:** yes; all internal quantities match PyTorch to max absolute error **4.12×10⁻¹⁷**.
- **Bias correction:** under a sustained <1% multiplier-gap rule, it stops materially changing the adaptive scale at approximately **step 3916**.
- **Warmup/U-W:** the final LR increase occurs at **step 20**; **step 21** is the first step where warmup no longer changes LR directly.
- **Cosine vs WSD at forced step 200:** after tuning both independently, cosine validation loss is **1.1036**, WSD is **1.1422**; I keep **cosine**.
- **Width LR minima:** fitted minima are approximately **0.00868 @256**, **0.00695 @512**, **0.00593 @1024**.
- **Width 4096:** I would start at **0.0040**, with **moderate confidence**, then run a local bracketed sweep before treating it as tuned.

**The submission does not ask the plots to tell a story the experiment did not earn. Every headline result has a definition, a control, and a raw artifact behind it.**
