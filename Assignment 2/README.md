# Assignment 2: A Simple Model For Cancer Growth

Code and figures for the second assignment: a dimensionless three-species model
(tumor `x1`, healthy tissue `x2`, effector immune cells `x3`) based on Itik &
Banks (2010). The analysis follows the equilibria and their stability, a
continuation in the parameter `p1` to a Hopf bifurcation, a shooting method for
the resulting limit cycle, and a branch-switching routine that follows the
period-doubling cascade towards chaos.

All modules use the default parameters `p2..p8 = 2.5, 0.6, 1.5, 4.5, 1.0, 0.2,
0.5`, with `p1` as the continuation parameter. Requirements: `numpy`, `scipy`,
`matplotlib`.

## Code

### RootFinding.py

Equilibria and linear stability (exercises 1–2). Computes the six equilibrium
points (zero, tumor, healthy, tumor–healthy, and the two coexistence states)
in closed form, builds the analytic Jacobian `J(x, p1)`, and classifies each
point from the sign of the largest real part of its eigenvalues (`numpy.linalg.eig`,
a QR-based solver). Provides `get_coexistence_state`, `calculate_jacobian`,
`compute_eigenvalues` and `analyze_stability`, which are reused by the
continuation module.

### Continuation.py

Continuation of the stable coexistence equilibrium in `p1 ∈ [0, 1]` and
detection of the Hopf bifurcation (exercise 3). Uses natural parameter
continuation (1000 steps), evaluating the equilibrium and its eigenvalues at
each `p1`. The Hopf point is detected from a sign change of the test function
`ε(λ) = ∏_{k<j}(μ_k + μ_j)` and confirmed against the three criteria
(existence, non-degeneracy, transversality). At the located point it reports the
period of the nascent limit cycle from the imaginary eigenvalue, `T = 2π/ω`.
Includes plotting routines for the continuation profile, the stability test and
the frequency `ω(λ)`.

### LimitCycle.py

Shooting method for the limit cycle, its stability, and the first period
doubling (exercise 4). Time is rescaled to `ξ = τ/T ∈ [0,1]`, so the period `T`
becomes an unknown of the boundary value problem `y(0) = y(1)`.

Time integration (`scipy.integrate.solve_ivp`, Radau — an implicit Runge–Kutta
method) enters in two distinct roles:

1. **Seeding (once).** `seed_cycle` integrates the plain ODE over *many* periods
   from an arbitrary interior point at `p1 = 0.74` (just past the Hopf point)
   until the trajectory settles on the attracting cycle. The period is estimated
   from successive `x1`-maxima. This produces only the *initial guess*
   `(y0, T0)` for the root finder — the cycle is approached asymptotically, not
   solved exactly.
2. **Shooting (every Newton evaluation).** `integrate_period` integrates the
   state together with the variational equation `Φ' = T·J·Φ`, `Φ(0)=I`, over a
   *single* period, returning `y(1)` and the monodromy matrix `M = Φ(1)`. This
   is the "shot": the periodicity residual `y(1) − y0` and `M` are read off from
   it.

The two are used in sequence — seed once, then shoot repeatedly:

- The periodicity system is closed with the Poincaré phase condition
  `(y0 − y)ᵀ F(y; p1) = 0` (Seydel, Eq. 7.3) and solved by Newton–Raphson
  (`refine`), using the analytic shooting Jacobian (`M − I`, `F(y1)`, the phase
  row, and a finite-difference `p1` column). Each Newton iteration calls
  `integrate_period`; the seed from step 1 provides its starting point.
- `floquet` returns the Floquet multipliers and identifies the trivial
  multiplier `ν0 = 1` by eigenvector alignment with the flow `F(y0)`;
  `delta = |1 − ν0|` is the accuracy diagnostic. `order_multipliers` tracks the
  multipliers by continuity along the branch.
- `continue_in_p1` performs pseudo-arclength continuation (explicit Euler
  predictor, Newton corrector, with the period weighted in the arclength metric)
  and locates the first period doubling from a sign change of the test function
  `det(M + I) = ∏(ν_k + 1)`, refined by secant iteration.
- `plot_floquet` draws the Floquet multipliers up to the first bifurcation.

Outputs: `LimitCycle_Data.npy` (`p1, T, delta, det(M+I)`), `LimitCycle_Mults.npy`
(tracked multipliers), `PD_Point.npy` (orbit and parameter at the first period
doubling), and `figures/floquet_period1.png`. The first period doubling is found
at `p1 ≈ 0.9232`, `T ≈ 31.55`, with a real multiplier crossing `−1`.

### PeriodDoubling.py

Branch switching and the period-doubling cascade (exercise 5). Building on
`LimitCycle.py`:

- `branch_switch` seeds the period-`2T` orbit by perturbing `y0 ± ε·v` along the
  eigenvector `v` of `M` at `ν = −1`, and rejects the trivial double cover with
  `is_period_two` (the half-period point must differ from the start while the
  full period closes).
- `find_cascade` repeats switch-then-continue, carrying the multiplier labels
  across each switch by seeding the continuation with the squared parent
  multipliers, and rejects spurious detections (bad `delta`, `ν ≠ −1`, or `p1`
  out of range) once single shooting loses accuracy on the long orbits.

Outputs: `PeriodDoubling_Results.npy` (one record per doubling: level, `p1`,
`T`, multipliers, `delta`) and `PD_Branch_{2,4,8}.npy` / `PD_Mults_{2,4,8}.npy`
(period and multiplier tables per branch). The cascade is found at
`p1 = 0.9232, 0.9444, 0.9486` (periods `1→2→4→8`); the first Feigenbaum ratio is
`≈ 5.06` and the accumulation (chaos onset) is estimated at `p∞ ≈ 0.95`. Beyond
the third doubling the period-8 branch reaches the single-shooting accuracy
limit, which coincides with the accumulation point; the Lyapunov spectrum
(discussed in the report) is the appropriate tool past this point.

## Figures

`figures/` holds the continuation, stability-test and frequency plots
(`figure1–3.png`) and the limit-cycle Floquet multipliers up to the first period
doubling (`floquet_period1.png`).

## Report

The report can be found here:

https://www.overleaf.com/project/69b819d63f1835a9b1a8b646
