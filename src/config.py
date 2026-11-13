"""Experiment parameters for the replication of

Convergence rates for truncated and discretised signature expansions.

All experiment parameters live here as module-level UPPER_CASE variables.
No classes or dicts: every variable is annotated with its meaning, unit and
the paper section/table it comes from ("not specified in the paper" when the
paper leaves it open).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: linear delay equation (Section 4.2, p.13)
# ---------------------------------------------------------------------------
EXP1_D = 1                       # state dimension of the driving Brownian motion, Sec. 4.2
EXP1_T = 1.0                      # time horizon T, Sec. 4.2
EXP1_X0 = 0.25                    # initial value x of the delay equation, Sec. 4.2
EXP1_A = np.array([0.25, 1.0])    # a_i, i=0,1 (a_0=drift-type, a_1=diffusion-type), Sec. 4.2
EXP1_B = np.array([0.0, 1.0])     # b_i, i=0,1, Sec. 4.2
EXP1_N_KERNEL = (1, 1)             # number of exponential terms n_i in K_i, i=0,1, Sec. 4.2
# alpha^1=(alpha_0^1, alpha_1^1) and beta^1=(beta_0^1, beta_1^1), indexed by the
# kernel i in {0,1}; since n_i=1 for both kernels there is a single k=1 term each,
# so K_i(t) = ALPHA[i] * exp(BETA[i] * t), Sec. 4.2.
EXP1_ALPHA = np.array([-4.0, -2.0])   # alpha_i^1, i=0,1, Sec. 4.2
EXP1_BETA = np.array([1.0, 1.5])      # beta_i^1, i=0,1, Sec. 4.2

EXP1_N_REF_PAPER = 10_000         # reference Euler/interpolation grid size, Sec. 4.2
EXP1_N_REF_RUN = 2_000            # grid size used in the executed run (time budget, see RESULTS.md)
EXP1_N_PATHS_PAPER = 20_000       # Monte Carlo sample size, Sec. 4.2
EXP1_N_PATHS_RUN = 4_000          # Monte Carlo sample size used in the executed run
EXP1_M_LEVELS = (2, 3, 4, 5, 6, 7)       # truncation levels M tested, Sec. 4.2
EXP1_N_LEVELS = (10, 50, 100, 500, 1000, 5000)  # discretisation levels N tested, Sec. 4.2
EXP1_TEST_TIMES = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 4))  # evaluation times t
# The paper lists t in {0.0,...,1.0} step 0.1 (Sec. 4.2); the extra midpoints
# (step 0.05) are off-grid for N=10 and expose the scalloped shape described
# in the expected outcomes without changing the specified marker times.
EXP1_STRONG_P = 3                 # p used for strong errors ||.||_p, implementation notes, Sec. 4.2
EXP1_FIT_TIMES = (0.5, 0.75)       # times at which CB^M/(M!)^0.5 is fitted, Sec. 4.2
EXP1_EPSILON = 0.5                 # optimal summability exponent for the delay equation, Remark 5.5
EXP1_DISC_M_FIXED = (3, 5)         # M held fixed when studying discretisation error, Sec. 4.2

# ---------------------------------------------------------------------------
# Experiment 2: shifted Riemann-Liouville fractional Brownian motion (Sec. 4.3, p.15)
# ---------------------------------------------------------------------------
EXP2_D = 1                        # dimension, Sec. 4.3
EXP2_H = 0.3                       # Hurst parameter, Sec. 4.3
EXP2_H_ALT = 0.7                   # alternative Hurst parameter quoted for comparison, Sec. 4.3
EXP2_MU = 1.0 / 52.0                # shift mu, Sec. 4.3
EXP2_T = 1.0                       # time horizon, Sec. 4.3
EXP2_N_REF_PAPER = 10_000
EXP2_N_REF_RUN = 2_000
EXP2_N_PATHS_PAPER = 20_000
EXP2_N_PATHS_RUN = 4_000
EXP2_M_LEVELS = (2, 3, 4, 5, 6, 7)
EXP2_TEST_TIMES = tuple(np.round(np.arange(0.0, 1.0001, 0.05), 4))
EXP2_STRONG_P = 3                  # not stated for this section; reuses the p=3 of Sec. 4.2 (uncertainty)

# ---------------------------------------------------------------------------
# Experiment 3: signature volatility model, put option pricing (Sec. 4.4 / 7, p.16, p.28)
# ---------------------------------------------------------------------------
EXP3_S0 = 1.0                      # initial price S_0; not specified in the paper, normalised to 1
EXP3_K = 1.0                       # strike K; not specified in the paper, at-the-money
EXP3_RHO = -0.5                    # correlation rho between W and the price Brownian motion; not specified in the paper
EXP3_T = 1.0                       # time horizon, Sec. 4.4
EXP3_ELL1 = 0.3                    # coefficient of letter '1' (W) in the volatility functional ell, not specified in the paper
EXP3_ELL0 = 0.2                    # coefficient of letter '0' (time) in ell (mean level of sigma_t), not specified in the paper
EXP3_ELL_EMPTY = 0.2               # coefficient of the empty word in ell (base volatility level), not specified in the paper
# ell is chosen in A_exp-type form (geometric coefficient decay) so that it lies
# in L^1 with the same type of summability exponent as the delay equation example.
EXP3_N_REF_PAPER = 10_000
EXP3_N_REF_RUN = 2_000
EXP3_N_PATHS_PAPER = 20_000
EXP3_N_PATHS_RUN = 4_000
EXP3_M_LEVELS = (2, 3, 4, 5, 6)
EXP3_N_LEVELS = (10, 50, 100, 500, 1000)

# ---------------------------------------------------------------------------
# Experiment 4: strong truncation error lemma and sharpness (Sec. 3.2 / 5.1, p.10, p.17, p.21)
# ---------------------------------------------------------------------------
EXP4_D = 1                         # dimension used for the word-counting bound, Sec. 5.1
EXP4_P = 2                          # moment order p in psi_{p,t}, Sec. 5.1
EXP4_T_GRID = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))  # t in (0,T], Prop. 3.8
EXP4_EPS_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)  # epsilon in (0,1/2], Prop. 3.8
EXP4_N_BAR_GRID = (50, 500, 5000, 50_000)  # N values (uniformity check), Prop. 3.8
EXP4_M_GRID = tuple(range(2, 41, 2))       # truncation levels for the sharpness check, Prop. 3.8

# ---------------------------------------------------------------------------
# Experiment 5: strong and weak discretisation error on words (Sec. 5.2 / 5.3, p.22-26)
# ---------------------------------------------------------------------------
EXP5_D = 1                          # alphabet {0,1}, single Brownian direction, Sec. 5.2
EXP5_T = 1.0                        # time horizon
EXP5_N_PATHS_PAPER = 20_000
EXP5_N_PATHS_RUN = 4_000
EXP5_N_REF_PAPER = 10_000
EXP5_N_REF_RUN = 2_000
EXP5_N_LEVELS = (10, 50, 100, 500, 1000, 5000)
EXP5_WORDS_STRONG = ((1,), (1, 1), (0, 1), (1, 0))   # single words checked for the strong rate
EXP5_WORDS_WEAK = ((1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0))  # words checked for the weak rate
EXP5_STRONG_P = 2                   # moment order p for the strong word bound, Prop. 5.14

# ---------------------------------------------------------------------------
# Experiment 6: extensions to probability, distribution functions, risk measures (Sec. 4.1/6, p.11-12, 26-27)
# ---------------------------------------------------------------------------
EXP6_P = 4                          # moment order p used in Lemma 4.1 / Theorem 4.3, not specified -> chosen finite p>1
EXP6_THETA = 0.5                    # Hölder exponent theta in (0,1], Lemma 4.2, not specified in the paper
EXP6_Q_RISK = 4.0                   # integrability q of dQ/dP for the coherent risk measure, Theorem 4.3
EXP6_C_RHO = 1.5                    # bound C_rho on ||dQ/dP||_q, not specified in the paper
EXP6_ALPHA_VAR = (0.95, 0.99)       # VaR confidence levels alpha, Theorem 4.4
EXP6_DENSITY_C = 0.4                # lower bound c on the density of g(X_t) on B, Theorem 4.4(ii), not specified in the paper
EXP6_GAMMA = 0.2                    # margin gamma in Lemma 6.3's quantile-stability argument, not specified in the paper
EXP6_RHO_GRID = np.geomspace(0.5, 1e-4, 25)  # synthetic values of rho = ||Z1-Z2||_p used to verify the rate in E_a(M,N)

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------
RNG_SEED = 20240521  # fixed seed for every Monte Carlo experiment, for reproducibility
RESULTS_DIR = "results"
