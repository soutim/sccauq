#!/usr/bin/env Rscript
###############################################################################
# scca_benchmark_rankr.R -- rank-r sparse-CCA inference benchmark
#
# Compares 9 interval methods: ECCAR split-sample and closed-form (ours), Laha
# (leading pair only), and bootstrap baselines (ECCAR-raw, Witten, Parkhomenko,
# Waaijenborg, Gao-Ma, Wilms-Croux). Outputs coverage, Type I error, power,
# size-adjusted power, CI width, bias/variance and P-P calibration, subspace
# distance and runtime, as CSVs plus figures.
#
# Two experiments, both run by default (edit RUN below):
#   A  sample size   vary n at fixed (p,q)   -> <n_outdir>/scca_rankr_*.png
#   B  dimension     vary (p,q) at fixed n   -> <dim_outdir>/scca_dim_*.png
#
# Run:  source in RStudio, or `Rscript scca_benchmark_rankr.R`.
# Cluster entry points (flags override RUN):
#   --n=,--p=,--q=,--mc=,--B=,--K=,--cores=,--outdir=,--checkpoint
#   --aggregate                                  re-plot from checkpoints
#   --sweep=dim --sweep_n= --dims= --dirs=       combine dimension-sweep points
#   --precision=scio|nodewise                    precision estimator (default scio)
#   --label_style=abbrev|legend|direct           figure labelling (default abbrev)
#   --oracle_lam=off|gap|all                     ORACLE diagnostic, never for results
#
# Debiasing uses a symmetrized SCIO precision estimate (Liu & Luo 2015):
# ||Sigma_hat Theta_hat - I||_max = O_P(sqrt(log p / n)) by the KKT bound, with a
# shared Hessian across columns; symmetrizing gives the two-sided control the
# bilinear debiasing cancellation requires.
#
# --checkpoint writes resumable .rds per n. Replicate m uses an L'Ecuyer stream
# derived from (seed_base + n) independently of chunking, so resuming an
# interrupted run is bit-identical to running it straight through.
###############################################################################

## ------------------------- EXPERIMENT KNOBS --------------------------------
## Regime condition s^2 log^2(p+q) / n < 1 is evaluated and printed for every
## point before the run starts.
RUN <- list(
  ## shared
  r            = 3,                    # canonical pairs
  lambda_star  = c(0.9, 0.7, 0.5),     # true canonical correlations (length r)
  s_u          = 3,                    # nonzeros per column of U*
  s_v          = 3,                    # nonzeros per column of V*
  ar_rho       = 0.3,                  # AR(1) marginal correlation
  mc           = 500,                  # replicates per point
  B            = 200,                  # bootstrap resamples (competitors only)
  K            = 10,                   # split-sample folds
  alpha        = 0.05,
  seed_base    = 20260520,
  cores        = 1,                    # MC workers (forking: not on Windows)

  ## A. sample-size sweep
  do_n_sweep   = TRUE,
  n_grid       = c(100, 300, 500, 700, 900),
  n_p          = 800,
  n_q          = 850,
  n_outdir     = "results_nsweep",

  ## B. dimension sweep
  do_dim_sweep = TRUE,
  dim_n        = 500,
  dim_p        = c(600, 800, 1050, 1350, 1750),
  dim_q        = c(650, 850, 1100, 1400, 1800),   # same length as dim_p
  dim_outdir   = "results_dim_combined",
  dim_point_dir= "results_dim_p"       # per-point dirs: results_dim_p600, ...
)

## ---- configuration -- override via --name=value on the CLI ----
CONFIG <- list(
  ## ---- data generation ----
  n_grid      = RUN$n_grid,    # sample sizes (--n=100,200,400)
  p           = RUN$n_p,             # dimension of X            (--p)
  q           = RUN$n_q,             # dimension of Y            (--q)
  r           = RUN$r,                   # number of canonical pairs (--r)
  lambda_star = RUN$lambda_star,         # true canonical correlations, length r (--lambda)
  s_u         = RUN$s_u,                   # row-sparsity of U*        (--s_u)
  s_v         = RUN$s_v,                   # row-sparsity of V*        (--s_v)
  ar_rho      = RUN$ar_rho,                 # AR(1) marginal correlation (--ar_rho)
  ## ---- simulation ----
  M_grid      = RUN$mc,      # MC replicates per n   (--mc=5 or --mc=5,5,5)
  B           = RUN$B,                 # bootstrap resamples       (--B)
  K           = RUN$K,                  # split-sample folds        (--K)
  alpha       = RUN$alpha,                # CI level                  (--alpha)
  seed_truth_u = 7, seed_truth_v = 17, seed_base = RUN$seed_base,   # (--seed_base etc.)
  ## ---- ECCAR estimation / regularization ----
  admm_rho    = 1,                   # ADMM step size            (--admm_rho)
  lambda_mult = 0.7,                 # ADMM L1 penalty = lambda_mult * sqrt(log(p+q)/n)  (--lambda_mult)
  node_mult   = 1,                   # SCIO penalty = node_mult * sqrt(log(p)/n)         (--node_mult)
  admm_iter   = 5000, admm_tol = 1e-4,       # ADMM iteration cap / RELATIVE tolerance (matches ccar3 eps=1e-4)
  node_thresh = 1e-3, node_dfmax_mult = 2,   # SCIO coordinate-descent tol; dfmax = mult*sqrt(p)
  scio_iter   = 200,                         # SCIO coordinate-descent max sweeps per column
  split_shrink_k = 0,                        # non-paper finite-sample deflation on the split bump (0 => paper Def 9.4)
  split_cap = 0,                             # non-paper robust fold aggregation: Winsorize at split_cap x median (0 => paper mean)
  ## ---- ECCAR solve: inline ADMM by default (ccar3::ecca lacks B-hat/Z-hat needed for debias) ----
  use_ccar3   = FALSE,
  ccar3_path  = Sys.getenv("CCAR3_PATH", ""),                  # ccar3 package source (--ccar3_path; env CCAR3_PATH)
  methods_file= Sys.getenv("CCA_UQ_METHODS", ""),              # cca_uq_methods.R (--methods; env CCA_UQ_METHODS)
  prefer_source = TRUE,                        # passed to get_ccar3_api()/fit_ecca_cv(), as in cca_uq_simulation.R
  node_cores  = NULL,                          # workers for the SCIO column loop; NULL => auto (--node_cores)
  inner_cores = NULL,                          # within-rep workers for split folds + bootstrap resamples; NULL => auto (--inner_cores)
  ## ---- competitor methods ----
  comp_iter   = 100, comp_tol = 1e-5,        # Witten/Parkhomenko/Waaijenborg iteration cap / tol
  comp_lambda_mult = 1,              # soft-threshold penalty mult for Waaijenborg/Parkhomenko
  park_ridge  = 0.1,                 # Parkhomenko ridge
  gm_ridge    = 0.05,                # Gao-Ma whitening ridge
  wilms_alpha = 0.7,                 # Wilms-Croux elastic-net mixing (1=lasso, 0=ridge)
  wilms_lambda_mult = 0.5,           # Wilms-Croux penalty mult
  wilms_iter  = 50,                  # Wilms-Croux alternating cap (--fast sets 15)
  ## ---- parallel / output ----
  n_cores     = RUN$cores,                   # MC-loop workers; 1 = exact-reproducible serial (--cores)
  blas_threads = NULL,               # BLAS threads PER worker; NULL => auto      (--blas)
  outdir      = "scca_benchmark_results"
)

## ------------------------------ CLI parsing --------------------------------
args <- commandArgs(trailingOnly = TRUE)
AGG_ONLY  <- any(args == "--aggregate")
# Dimension-sweep aggregation: fixed n, x-axis = p+q, reads several point dirs.
# Triggered by --sweep=dim; combines their checkpoints and plots against p+q.
SWEEP_DIM <- { sv <- grep("^--sweep=", args, value = TRUE)
               length(sv) && sub("^--sweep=", "", sv[1]) == "dim" }
CONFIG$fast <- any(args == "--fast")
CONFIG$checkpoint <- any(args == "--checkpoint")   # opt-in: write resumable .rds files
CONFIG$save_figs <- any(args == "--save")          # deprecated no-op: CSVs + figures always go to --outdir
flag_raw <- function(name) { h <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(h)) sub(paste0("^--", name, "="), "", h[1]) else NULL }
num   <- function(name, d) { v <- flag_raw(name); if (is.null(v)) d else as.numeric(v) }
vecnum<- function(name, d) { v <- flag_raw(name); if (is.null(v)) d else as.numeric(strsplit(v, ",")[[1]]) }
`%||%` <- function(a, b) if (is.null(a)) b else a
# data generation
CONFIG$n_grid <- vecnum("n", CONFIG$n_grid)
CONFIG$p <- num("p", CONFIG$p);   CONFIG$q <- num("q", CONFIG$q);   CONFIG$r <- num("r", CONFIG$r)
CONFIG$lambda_star <- vecnum("lambda", CONFIG$lambda_star)
CONFIG$s_u <- num("s_u", CONFIG$s_u); CONFIG$s_v <- num("s_v", CONFIG$s_v); CONFIG$ar_rho <- num("ar_rho", CONFIG$ar_rho)
# simulation
CONFIG$B <- num("B", CONFIG$B); CONFIG$K <- num("K", CONFIG$K); CONFIG$alpha <- num("alpha", CONFIG$alpha)
CONFIG$seed_base <- num("seed_base", CONFIG$seed_base)
# estimation knobs
CONFIG$admm_rho <- num("admm_rho", CONFIG$admm_rho)
CONFIG$lambda_mult <- num("lambda_mult", CONFIG$lambda_mult)
CONFIG$node_mult <- num("node_mult", CONFIG$node_mult)
CONFIG$admm_iter <- num("admm_iter", CONFIG$admm_iter); CONFIG$admm_tol <- num("admm_tol", CONFIG$admm_tol)
CONFIG$scio_iter <- as.integer(num("scio_iter", CONFIG$scio_iter))
# Precision estimator for the debiasing: "scio" (symmetrized SCIO, default) or
# "nodewise" (the original nodewise LASSO). Both return a symmetric p x p estimate
# and are interchangeable at the call site, so this is a clean A/B switch.
CONFIG$precision <- flag_raw("precision") %||% "scio"
if (!CONFIG$precision %in% c("scio", "nodewise"))
  stop("--precision must be one of: scio, nodewise")
# How the 9-estimator line panels are labelled.
#   legend (default) : shared legend under each figure. CANNOT crop -- the legend is
#                      laid out by the graphics device, not positioned in data units.
#   direct           : end-of-line text inside a padded x gutter. Prettier when it
#                      works, but the text width is not known until render time, so
#                      long names (Waaijenborg, Parkhomenko) can overrun the panel.
CONFIG$label_style <- flag_raw("label_style") %||% "abbrev"
if (!CONFIG$label_style %in% c("abbrev", "legend", "direct"))
  stop("--label_style must be one of: abbrev, legend, direct")
# ORACLE diagnostic, never for reported results: substitutes the TRUE lambda*
# into the debiasing to test whether lambda-hat's bias corrupts the one-step
# correction via the gap denominators. off = paper estimator; gap = denominators
# only; all = everywhere lhat enters. Reported lambda-hat stays genuine.
CONFIG$oracle_lam <- flag_raw("oracle_lam") %||% "off"
if (!CONFIG$oracle_lam %in% c("off", "gap", "all"))
  stop("--oracle_lam must be one of: off, gap, all")
CONFIG$split_shrink_k <- num("split_shrink_k", CONFIG$split_shrink_k)
CONFIG$split_cap <- num("split_cap", CONFIG$split_cap)
CONFIG$park_ridge <- num("park_ridge", CONFIG$park_ridge); CONFIG$gm_ridge <- num("gm_ridge", CONFIG$gm_ridge)
CONFIG$wilms_alpha <- num("wilms_alpha", CONFIG$wilms_alpha)
# parallel
CONFIG$n_cores <- num("cores", CONFIG$n_cores)
bt <- flag_raw("blas"); CONFIG$blas_threads <- if (is.null(bt)) CONFIG$blas_threads else as.integer(bt)
# output folder + ccar3 solve + SCIO parallelism
CONFIG$outdir       <- flag_raw("outdir")      %||% CONFIG$outdir
CONFIG$use_ccar3    <- if (any(args == "--no_ccar3")) FALSE else
                       if (any(args == "--use_ccar3")) TRUE else isTRUE(CONFIG$use_ccar3)
CONFIG$ccar3_path   <- flag_raw("ccar3_path")  %||% CONFIG$ccar3_path
CONFIG$methods_file <- flag_raw("methods")     %||% CONFIG$methods_file
nc_in <- flag_raw("node_cores"); CONFIG$node_cores <- if (is.null(nc_in)) CONFIG$node_cores else as.integer(nc_in)
ic_in <- flag_raw("inner_cores"); CONFIG$inner_cores <- if (is.null(ic_in)) CONFIG$inner_cores else as.integer(ic_in)
# --fast preset: cheaper Wilms-Croux + fewer folds (unless --K given explicitly)
if (CONFIG$fast) { CONFIG$wilms_iter <- 15L; if (is.null(flag_raw("K"))) CONFIG$K <- 5L }
# MC replicates: --mc=<scalar|vec>, else positional ints, else default; recycle to length(n_grid)
pos <- suppressWarnings(as.integer(args[!grepl("^--", args)])); pos <- pos[!is.na(pos)]
mc_in <- if (!is.null(flag_raw("mc"))) vecnum("mc", NULL) else if (length(pos)) pos else CONFIG$M_grid
CONFIG$M_grid <- if (length(mc_in) == 1) rep(as.integer(mc_in), length(CONFIG$n_grid)) else as.integer(mc_in)
# integer-ize counts
for (k in c("p","q","r","s_u","s_v","B","K","n_cores","admm_iter")) CONFIG[[k]] <- as.integer(CONFIG[[k]])
CONFIG$n_grid <- as.integer(CONFIG$n_grid)
stopifnot(length(CONFIG$lambda_star) == CONFIG$r, length(CONFIG$M_grid) == length(CONFIG$n_grid))
M_grid <- CONFIG$M_grid

suppressPackageStartupMessages({
  library(glmnet); library(dplyr); library(tibble)
  library(ggplot2); library(tidyr); library(patchwork)
  library(parallel)
})
HAS_BLASCTL <- requireNamespace("RhpcBLASctl", quietly = TRUE)

# Cores partition: mc_workers x blas_threads_per_worker <= total cores.
TOTAL_CORES <- tryCatch(parallel::detectCores(), error = function(e) 1L)
if (is.na(TOTAL_CORES) || TOTAL_CORES < 1) TOTAL_CORES <- 1L
# Within-rep parallelism dominates when MC is serial; with parallel MC, inner_cores=1.
if (is.null(CONFIG$inner_cores))
  CONFIG$inner_cores <- if (CONFIG$n_cores <= 1) TOTAL_CORES else 1L
CONFIG$inner_cores <- max(1L, as.integer(CONFIG$inner_cores))
# Any forking implies 1 BLAS thread per worker.
forking <- CONFIG$n_cores > 1L || CONFIG$inner_cores > 1L
if (is.null(CONFIG$blas_threads)) CONFIG$blas_threads <- if (forking) 1L else TOTAL_CORES
# SCIO stays serial (parallelism is taken at fold/bootstrap level).
if (is.null(CONFIG$node_cores)) CONFIG$node_cores <- 1L
if (CONFIG$inner_cores > 1L) CONFIG$node_cores <- 1L
CONFIG$node_cores <- max(1L, as.integer(CONFIG$node_cores))
# Output destination: a results FOLDER (CSV tables + figure PNGs), created up front.
dir.create(CONFIG$outdir, showWarnings = FALSE, recursive = TRUE)
set_blas <- function(nthreads) {
  nthreads <- max(1L, as.integer(nthreads))
  if (HAS_BLASCTL) try(RhpcBLASctl::blas_set_num_threads(nthreads), silent = TRUE)
  Sys.setenv(OMP_NUM_THREADS = nthreads, OPENBLAS_NUM_THREADS = nthreads, MKL_NUM_THREADS = nthreads)
}
par_lapply <- function(X, FUN, cores) {
  if (cores > 1L) parallel::mclapply(X, FUN, mc.cores = cores, mc.preschedule = TRUE) else lapply(X, FUN)
}
if (max(CONFIG$n_cores, CONFIG$inner_cores) * CONFIG$blas_threads > TOTAL_CORES)
  cat(sprintf("NOTE: workers(%d) x blas(%d) > %d detected cores -- oversubscription likely.\n",
              max(CONFIG$n_cores, CONFIG$inner_cores), CONFIG$blas_threads, TOTAL_CORES))
cat(sprintf("Parallel config: MC workers=%d, within-rep workers=%d, BLAS threads/worker=%d (detected cores=%d).\n",
            CONFIG$n_cores, CONFIG$inner_cores, CONFIG$blas_threads, TOTAL_CORES))

## ---- echo the fully resolved configuration ----
if (!AGG_ONLY) {
  cat("\n================ RESOLVED CONFIGURATION ================\n")
  cat(sprintf("  data:    n = %s | p = %d | q = %d | r = %d | lambda* = %s\n",
              paste(CONFIG$n_grid, collapse=","), CONFIG$p, CONFIG$q, CONFIG$r, paste(CONFIG$lambda_star, collapse=",")))
  cat(sprintf("           s_u = %d | s_v = %d | ar_rho = %g\n", CONFIG$s_u, CONFIG$s_v, CONFIG$ar_rho))
  cat(sprintf("  sim:     mc = %s | B = %d | K = %d | alpha = %g | seed_base = %d\n",
              paste(CONFIG$M_grid, collapse=","), CONFIG$B, CONFIG$K, CONFIG$alpha, CONFIG$seed_base))
  cat(sprintf("  eccar:   admm_rho = %g | lambda_mult = %g | node_mult = %g | admm_iter = %d | admm_tol = %g | precision = %s\n",
              CONFIG$admm_rho, CONFIG$lambda_mult, CONFIG$node_mult, CONFIG$admm_iter, CONFIG$admm_tol, ifelse(identical(CONFIG$precision,"nodewise"), "nodewise LASSO", "symmetrized SCIO")))
  cat(sprintf("  comp:    comp_iter = %d | comp_lambda_mult = %g | park_ridge = %g | gm_ridge = %g\n",
              CONFIG$comp_iter, CONFIG$comp_lambda_mult, CONFIG$park_ridge, CONFIG$gm_ridge))
  cat(sprintf("           wilms: alpha = %g, lambda_mult = %g, iter = %d%s\n",
              CONFIG$wilms_alpha, CONFIG$wilms_lambda_mult, CONFIG$wilms_iter, if (CONFIG$fast) "   [--fast]" else ""))
  cat("=======================================================\n\n")
}

p <- CONFIG$p; q <- CONFIG$q; r <- CONFIG$r
s_u <- CONFIG$s_u; s_v <- CONFIG$s_v
lambda_star <- CONFIG$lambda_star; alpha <- CONFIG$alpha

## ------------------------------- helpers -----------------------------------
soft <- function(x, t) sign(x) * pmax(abs(x) - t, 0)
sym_sqrt <- function(S, fl = 1e-8) {
  ee <- eigen((S + t(S))/2, symmetric = TRUE)
  ee$vectors %*% diag(sqrt(pmax(ee$values, fl))) %*% t(ee$vectors)
}
# Exact ridged inverse / inverse-sqrt of a covariance from its (thin or full)
# eigendecomposition: (S + ridge I)^{-1} and (S + ridge I)^{-1/2}. With S = Q D Q'
# (range part) and an implicit zero-eigenvalue null space, the ridged operator is
#   f(S+ridge I) = f(ridge) I + Q [ f(d+ridge) - f(ridge) ] Q'.
# When the decomposition is THIN (rank k < p) this costs O(p^2 k) and avoids the
# O(p^3) solve()/eigen() the competitor whitening uses at p > n; for a full decomp
# it equals solve(S+ridge I) / sym_sqrt(solve(.)) exactly.
ridge_inv <- function(cxx, ridge) { Qr <- cxx$Q; p <- nrow(Qr)
  diag(p)/ridge + Qr %*% ((1/(cxx$d + ridge) - 1/ridge) * t(Qr)) }
ridge_invsqrt <- function(cxx, ridge) { Qr <- cxx$Q; p <- nrow(Qr)
  diag(p)/sqrt(ridge) + Qr %*% ((1/sqrt(cxx$d + ridge) - 1/sqrt(ridge)) * t(Qr)) }
# Single symmetric eigendecomposition of a covariance, reused for the ADMM
# denominator (Q, d) AND the canonical-direction square root (half). Floor and
# symmetrization match sym_sqrt / fit_admm exactly, so derived quantities are
# bit-identical to computing them separately. half = Q diag(sqrt(d)) Q' equals
# sym_sqrt(S): diagonal scaling is exact row scaling (no added rounding).
cov_decomp <- function(X, S = NULL, fl = 1e-8) {
  n <- nrow(X); p <- ncol(X)
  if (is.null(S)) S <- crossprod(X)/n
  if (p > n + 1L) {
    # THIN: rank(S) <= n, so recover the nonzero eigenpairs of S = X'X/n from the
    # n x n Gram matrix G = X X'/n (same nonzero spectrum). Eigenvector of S for
    # eigenvalue d_i is q_i = X' u_i / sqrt(n d_i) where G u_i = d_i u_i. This is
    # O(n^2 p + n^3) instead of the O(p^3) full eigendecomposition; the null space
    # (eigenvalue 0) is dropped and handled implicitly in the ADMM B-update.
    G  <- tcrossprod(X)/n
    eg <- eigen((G + t(G))/2, symmetric = TRUE)
    keep <- eg$values > fl; dr <- eg$values[keep]
    Qr <- crossprod(X, eg$vectors[, keep, drop = FALSE])
    Qr <- sweep(Qr, 2, sqrt(n * dr), "/")
    return(list(S = S, Q = Qr, d = dr, half = Qr %*% (sqrt(dr) * t(Qr)), thin = TRUE))
  }
  ee <- eigen((S + t(S))/2, symmetric = TRUE); d <- pmax(ee$values, fl)
  list(S = S, Q = ee$vectors, d = d, half = ee$vectors %*% (sqrt(d) * t(ee$vectors)), thin = FALSE)
}
# all permutations of 1:n (small r), used for component matching
permn <- function(n) { if (n == 1) return(list(1)); res <- list(); for (s in permn(n - 1)) for (i in 0:(n - 1)) res[[length(res) + 1]] <- append(s, n, after = i); res }
PERMS <- do.call(rbind, permn(r))
# match estimated directions to a reference: best column permutation + per-col sign,
# scored by |<U_est[,pi], U_ref>|_Sig
match_components <- function(U_est, U_ref, Sig) {
  best <- 1:r; bestscore <- -Inf; bestsign <- rep(1, r)
  for (i in seq_len(nrow(PERMS))) {
    pi <- PERMS[i, ]; Up <- U_est[, pi, drop = FALSE]
    sg <- numeric(r); score <- 0
    for (l in 1:r) { ip <- as.numeric(t(Up[, l]) %*% Sig %*% U_ref[, l]); sg[l] <- if (ip >= 0) 1 else -1; score <- score + abs(ip) }
    if (score > bestscore) { bestscore <- score; best <- pi; bestsign <- sg }
  }
  list(perm = best, sign = bestsign)
}
apply_match <- function(U, m) { Um <- U[, m$perm, drop = FALSE]; for (l in 1:r) Um[, l] <- m$sign[l] * Um[, l]; Um }
# Sigma-orthonormalize columns (Gram-Schmidt in Sig inner product) for subspace metric
sig_orthonormalize <- function(U, Sig) {
  Uon <- U
  for (l in 1:ncol(U)) {
    if (l > 1) for (k in 1:(l - 1)) Uon[, l] <- Uon[, l] - as.numeric(t(Uon[, k]) %*% Sig %*% Uon[, l]) * Uon[, k]
    nrm <- sqrt(as.numeric(t(Uon[, l]) %*% Sig %*% Uon[, l])); if (nrm < 1e-10) nrm <- 1
    Uon[, l] <- Uon[, l]/nrm
  }
  Uon
}
# Frobenius distance between Sig-orthogonal projections onto col-spaces of U_est, U_true
subspace_frob <- function(U_est, U_true, Sig) {
  Ue <- sig_orthonormalize(U_est, Sig); Ut <- sig_orthonormalize(U_true, Sig)
  Pe <- Ue %*% t(Ue) %*% Sig; Pt <- Ut %*% t(Ut) %*% Sig
  sqrt(sum((Pe - Pt)^2))
}
# Subspace distance via principal angles in the Sigma inner product:
# d = ||sin Theta||_F = sqrt( r - ||Ue' Sig Ut||_F^2 ), where the singular values of
# Ue' Sig Ut are the cosines of the principal angles between the two col-spaces.
# Invariant to basis/sign/permutation within each subspace; range [0, sqrt(r)].
subspace_dist <- function(U_est, U_true, Sig) {
  Ue <- sig_orthonormalize(U_est, Sig); Ut <- sig_orthonormalize(U_true, Sig)
  cosT <- pmin(pmax(svd(t(Ue) %*% Sig %*% Ut)$d, 0), 1)
  sqrt(max(ncol(U_true) - sum(cosT^2), 0))
}

## --------------------------- DGP / ground truth ----------------------------
ar1_cov <- function(d, rho) rho^abs(outer(seq_len(d), seq_len(d), "-"))
# disjoint, well-separated blocks => sparse + (numerically) Sigma-orthonormal
build_truth <- function(d, r, s, Sig, seed) {
  set.seed(seed)
  blk <- split(1:d, cut(1:d, r, labels = FALSE))
  U <- matrix(0, d, r)
  for (l in 1:r) { supp <- sort(sample(blk[[l]], s)); U[supp, l] <- (1/sqrt(s)) * sample(c(-1, 1), s, replace = TRUE) }
  for (l in 1:r) {
    if (l > 1) for (k in 1:(l - 1)) U[, l] <- U[, l] - as.numeric(t(U[, k]) %*% Sig %*% U[, l]) * U[, k]
    U[, l] <- U[, l]/sqrt(as.numeric(t(U[, l]) %*% Sig %*% U[, l]))
  }
  U
}
# (Re)build the DGP for a given (p, q). Estimators read p, q, Ustar, Vstar,
# SigmaX, SigmaY, chol_J as globals, so the dimension sweep reassigns them
# between points.
build_dgp <- function(p_new, q_new) {
  G <- .GlobalEnv
  p_new <- as.integer(p_new); q_new <- as.integer(q_new)
  assign("p", p_new, envir = G); assign("q", q_new, envir = G)
  CONFIG$p <- p_new; CONFIG$q <- q_new; assign("CONFIG", CONFIG, envir = G)
  SX <- ar1_cov(p_new, CONFIG$ar_rho); SY <- ar1_cov(q_new, CONFIG$ar_rho)
  Us <- build_truth(p_new, r, s_u, SX, CONFIG$seed_truth_u)
  Vs <- build_truth(q_new, r, s_v, SY, CONFIG$seed_truth_v)
  Bs <- Us %*% diag(lambda_star, r) %*% t(Vs)
  SXY <- SX %*% Bs %*% SY
  SJ <- rbind(cbind(SX, SXY), cbind(t(SXY), SY))
  ee <- min(eigen(SJ, symmetric = TRUE, only.values = TRUE)$values)
  if (ee < 1e-6) SJ <- SJ + (1e-6 - ee) * diag(p_new + q_new)
  for (nm in c("SigmaX","SigmaY","Ustar","Vstar","Bstar","SigmaXY","Sigma_joint","chol_J"))
    assign(nm, switch(nm, SigmaX=SX, SigmaY=SY, Ustar=Us, Vstar=Vs, Bstar=Bs,
                          SigmaXY=SXY, Sigma_joint=SJ, chol_J=chol(SJ)), envir = G)
  invisible(NULL)
}
build_dgp(CONFIG$p, CONFIG$q)

gen_data <- function(n) {
  Z <- matrix(rnorm(n * (p + q)), n, p + q) %*% chol_J
  list(X = scale(Z[, 1:p], scale = FALSE), Y = scale(Z[, p + (1:q)], scale = FALSE))
}

## ---- ESTIMATORS ----
# Solve B via ccar3 (through cca_uq_methods.R wrapper). Prefers fixed-lambda entry
# if present; else fit_ecca_cv() with a single-lambda grid.
.ccar3_env <- new.env(parent = globalenv())
ccar3_eccar_B <- function(X, Y, lambda) {
  env <- .ccar3_env
  if (!exists(".loaded", envir = env, inherits = FALSE)) {
    mf <- CONFIG$methods_file
    if (is.null(mf) || !nzchar(mf) || !file.exists(mf)) stop("methods_file not found: ", mf)
    sys.source(normalizePath(mf), envir = env, chdir = TRUE)   # defines get_ccar3_api, fit_ecca_cv, ...
    assign(".loaded", TRUE, envir = env)
  }
  if (!exists(".api", envir = env, inherits = FALSE)) {        # built once, then cached
    api <- if (exists("get_ccar3_api", envir = env, inherits = FALSE))
             get("get_ccar3_api", envir = env)(ccar3_path = CONFIG$ccar3_path,
                                               prefer_source = CONFIG$prefer_source, quiet = TRUE)
           else NULL
    assign(".api", api, envir = env)
  }
  api <- get(".api", envir = env)
  if (exists("fit_ecca_fixed_lambda", envir = env, inherits = FALSE)) {
    fit <- get("fit_ecca_fixed_lambda", envir = env)(
      X = X, Y = Y, r = CONFIG$r, lambda = lambda, preprocess_mode = "center",
      ccar3_api = api, ccar3_path = CONFIG$ccar3_path,
      prefer_source = CONFIG$prefer_source, verbose = FALSE)
  } else {                                                     # verified fit_ecca_cv signature, single lambda
    fit <- get("fit_ecca_cv", envir = env)(
      X = X, Y = Y, r = CONFIG$r, lambdas = lambda, kfolds = 2L, preprocess_mode = "center",
      ccar3_api = api, ccar3_path = CONFIG$ccar3_path,
      prefer_source = CONFIG$prefer_source, parallelize = FALSE, verbose = FALSE)
  }
  # Extract the p x q cross-covariance matrix B. The cca_uq wrapper exposes it as
  # top-level $B when ecca names it B/B_opt/Bhat; otherwise pull it from the raw
  # ecca output ($fit) -- by known aliases, then as the unique p x q matrix, so
  # this is robust to the field name used by whichever ccar3 version is loaded.
  p <- ncol(X); q <- ncol(Y)
  raw <- fit$fit
  B <- fit$B %||% fit$Bhat %||% fit$B_opt
  if (is.null(B) && !is.null(raw))
    B <- raw$Bhat %||% raw$B %||% raw$B_opt %||% raw$Bopt %||% raw$B_est %||%
         raw$Z %||% raw$Theta %||% raw$Pi %||% raw$Chat %||% raw$coef
  if (is.null(B) && !is.null(raw)) {
    mats <- Filter(function(z) is.matrix(z) && is.numeric(z) && nrow(z) == p && ncol(z) == q, raw)
    if (length(mats) >= 1L) B <- mats[[1L]]
  }
  if (is.null(B)) {
    raw_desc <- if (is.null(raw)) "<none>" else paste(sprintf("%s[%s]", names(raw),
                  vapply(raw, function(z) if (is.matrix(z)) paste(dim(z), collapse="x") else
                         paste0("len", length(z)), character(1))), collapse = ", ")
    stop("could not find the ", p, "x", q, " B matrix in the ccar3 fit. Raw ecca fields: ", raw_desc)
  }
  as.matrix(B)
}

# ECCAR penalized-LS solve. Both branches target the SAME convex program as the
# reference; by convexity its minimizer is unique, so identical (Sx,Sy,Sxy,lambda)
# give the same B once solved to convergence.
fit_admm <- function(X, Y, lambda, admm_rho = CONFIG$admm_rho, iter = CONFIG$admm_iter,
                     tol = CONFIG$admm_tol, cxx = NULL, cyy = NULL,
                     Zinit = NULL, Hinit = NULL) {
  n <- nrow(X); p <- ncol(X); q <- ncol(Y)
  if (is.null(cxx)) cxx <- cov_decomp(X)
  if (is.null(cyy)) cyy <- cov_decomp(Y)
  Sx <- cxx$S; Sy <- cyy$S; Sxy <- crossprod(X, Y)/n

  B <- NULL; Hout <- NULL
  if (isTRUE(CONFIG$use_ccar3)) {
    B <- tryCatch(ccar3_eccar_B(X, Y, lambda), error = function(e) {
      if (!exists(".ccar3_warned", envir = .ccar3_env, inherits = FALSE)) {
        message("NOTE: ccar3 solve unavailable (", conditionMessage(e),
                ").\n      Falling back to the inline ADMM (same convex program). ",
                "Pass --methods=PATH and --ccar3_path=PATH, or --no_ccar3 to silence.")
        assign(".ccar3_warned", TRUE, envir = .ccar3_env)
      }
      NULL })
  }

  if (is.null(B)) {
    # ADMM for min_B (1/2)<B, Sx B Sy> - <Sxy,B> + lambda||B||_1.
    # B-update in joint eigenbasis (low-rank "active-set" form, valid for thin or full Q):
    #   B = R/rho + Qx[(Qx' R Qy) o (1/(dx dy' + rho) - 1/rho)] Qy',  R = Sxy + rho(Z-H).
    # Costs O(k p q) at thin rank k <= n. Warm start (Zinit, Hinit) speeds refits.
    Qx <- cxx$Q; Qy <- cyy$Q; dx <- cxx$d; dy <- cyy$d
    M0 <- crossprod(Qx, Sxy) %*% Qy                       # kx x ky (constant)
    coef <- 1/(outer(dx, dy) + admm_rho) - 1/admm_rho
    Sxy_r <- Sxy / admm_rho
    Z <- if (is.null(Zinit)) matrix(0, p, q) else Zinit
    H <- if (is.null(Hinit)) matrix(0, p, q) else Hinit
    for (i in seq_len(iter)) {
      Zold <- Z; W <- Z - H
      Mrr <- M0 + admm_rho * (crossprod(Qx, W) %*% Qy)
      Bm  <- Sxy_r + W + Qx %*% (Mrr * coef) %*% t(Qy)
      Z <- soft(Bm + H, lambda/admm_rho); H <- H + Bm - Z
      # RELATIVE stopping rule (absolute test never fires at large p,q since the
      # Frobenius norms scale like sqrt(pq)); scaling by ||Z||_F fixes that.
      if (sqrt(sum((Bm - Z)^2)) + sqrt(sum((Z - Zold)^2)) <= tol * max(sqrt(sum(Z^2)), 1)) break
    }
    B <- Z; Hout <- H
  }
  Zsub <- pmax(pmin(-(Sx %*% B %*% Sy - Sxy)/lambda, 1), -1)
  list(B = B, Zsub = Zsub, Sx = Sx, Sy = Sy, Sxy = Sxy, H = Hout)
}
canonical_dirs_r <- function(B, X, Y, r, cxx = NULL, cyy = NULL) {
  if (is.null(cxx)) cxx <- cov_decomp(X)
  if (is.null(cyy)) cyy <- cov_decomp(Y)
  Sxh <- cxx$half; Syh <- cyy$half; ss <- svd(Sxh %*% B %*% Syh)
  U <- matrix(0, ncol(X), r); V <- matrix(0, ncol(Y), r); lam <- numeric(r)
  for (l in 1:r) {
    ls <- max(ss$d[l], 1e-8)
    U[, l] <- as.numeric(B %*% Syh %*% ss$v[, l])/ls
    V[, l] <- as.numeric(t(B) %*% Sxh %*% ss$u[, l])/ls
    lam[l] <- ss$d[l]
  }
  list(U = U, V = V, lambda = lam)
}
# ---------------------------------------------------------------------------
# Symmetrized SCIO precision estimator (Liu & Luo 2015), replacing nodewise LASSO.
#
# Per column j, SCIO solves the penalized-quadratic M-estimator
#     theta_j = argmin_theta  1/2 theta' Shat theta - e_j' theta + lam ||theta||_1,
# whose KKT condition  Shat theta_j - e_j = -lam z_j  (||z_j||_inf <= 1) yields the
# approximate-inversion bound ||Shat theta_j - e_j||_inf <= lam ~ sqrt(log p / n)
# BY OPTIMALITY -- the same rate, via the same mechanism, as nodewise LASSO. All p
# columns share the SAME Hessian Shat (only the linear term e_j changes), so the
# active set / warm-start structure is reusable, giving the constant-factor speedup.
# We symmetrize at the end (Theta <- (Theta+Theta')/2): nodewise controlled the
# row-wise residual Theta Shat - I (left inverse); SCIO column j controls the
# column-wise residual Shat theta_j - e_j (right inverse). Symmetrizing makes the
# control two-sided, which the bilinear debiasing cancellation in Theta_X S_n Theta_Y'
# needs (Theta_X acting as a left inverse, Theta_Y' as a right one).
#
# Coordinate descent for one column (active-set, soft-thresholding):
#   fixing all but coord i, the 1-D solution of the penalized quadratic is
#     theta_i <- soft( e_j[i] - sum_{k != i} S[i,k] theta_k , lam ) / S[i,i].
# Shat is the (centered) sample covariance; its diagonal is bounded away from 0.
scio_column <- function(S, dS, j, lam, max_iter = CONFIG$scio_iter, tol = CONFIG$node_thresh) {
  p <- nrow(S)
  theta <- numeric(p)
  # warm start at the diagonal (rescaled e_j): cheap and on-support for sparse Theta
  theta[j] <- 1 / dS[j]
  Sth <- S[, j] * theta[j]               # S %*% theta, maintained incrementally
  active <- j
  for (sweep in seq_len(max_iter)) {
    max_change <- 0
    # cycle over the active set first (cheap), periodically over all coords to admit entries
    cyc <- if (sweep %% 3L == 1L) seq_len(p) else active
    for (i in cyc) {
      old <- theta[i]
      # partial residual r_i = e_j[i] - (S theta)_i + S[i,i] theta_i
      ri <- (if (i == j) 1 else 0) - (Sth[i] - dS[i] * old)
      new <- soft(ri, lam) / dS[i]
      if (new != old) {
        delta <- new - old
        Sth <- Sth + S[, i] * delta      # rank-1 update of S %*% theta
        theta[i] <- new
        if (abs(delta) > max_change) max_change <- abs(delta)
      }
    }
    active <- which(theta != 0)
    if (max_change < tol) break
  }
  theta
}
scio_fast <- function(X, lam, dfmax = NULL) {
  n <- nrow(X); p <- ncol(X)
  Xc <- scale(X, scale = FALSE)
  S  <- crossprod(Xc) / n                # sample covariance (Hessian shared across columns)
  dS <- pmax(diag(S), 1e-8)
  # One SCIO solve per column -> column j of the precision estimate. Coordinate
  # descent is deterministic (no RNG), so computing the p columns in parallel is
  # bit-identical to the serial loop -- only the order of independent solves changes.
  one_col <- function(j) scio_column(S, dS, j, lam)
  nc <- CONFIG$node_cores
  cols <- if (!is.null(nc) && nc > 1L)
            parallel::mclapply(seq_len(p), one_col, mc.cores = nc)
          else lapply(seq_len(p), one_col)
  Theta <- do.call(cbind, cols)          # column j is theta_j
  (Theta + t(Theta)) / 2                  # symmetrized SCIO
}
# ---- Nodewise LASSO precision (van de Geer / Zhang-Zhang), --precision=nodewise --
# Row j from the LASSO of X_j on X_{-j}: with tau_j^2 = ||resid||^2/n + lam*||gamma||_1,
# row j is (-gamma_j, 1 at j) / tau_j^2. Symmetrized; drop-in for scio_fast.
nodewise_fast <- function(X, lam, dfmax = NULL) {
  n <- nrow(X); p <- ncol(X)
  Xc <- scale(X, scale = FALSE)
  one_row <- function(j) {
    fit <- tryCatch(glmnet::glmnet(Xc[, -j, drop = FALSE], Xc[, j], alpha = 1,
                                   lambda = lam, intercept = FALSE,
                                   standardize = FALSE, thresh = 1e-4),
                    error = function(e) NULL)
    row <- numeric(p)
    if (is.null(fit)) { row[j] <- 1; return(row) }        # degenerate fallback
    gam <- as.numeric(stats::coef(fit))[-1]
    res <- Xc[, j] - as.numeric(Xc[, -j, drop = FALSE] %*% gam)
    tau2 <- sum(res^2) / n + lam * sum(abs(gam))
    if (!is.finite(tau2) || tau2 < 1e-10) tau2 <- 1e-10
    row[-j] <- -gam / tau2
    row[j]  <- 1 / tau2
    row
  }
  nc <- CONFIG$node_cores
  rows <- if (!is.null(nc) && nc > 1L)
            parallel::mclapply(seq_len(p), one_row, mc.cores = nc)
          else lapply(seq_len(p), one_row)
  Theta <- do.call(rbind, rows)
  (Theta + t(Theta)) / 2
}
precision_fast <- function(X, lam, dfmax = NULL) {
  if (identical(CONFIG$precision, "nodewise")) nodewise_fast(X, lam, dfmax)
  else scio_fast(X, lam, dfmax)
}
# Rank-r ECCAR: per-component bias-corrected loadings. `light=TRUE` skips storing Theta.
eccar_fit_r <- function(X, Y, r, light = FALSE, admm_init = NULL) {
  n <- nrow(X)
  lambda_admm <- CONFIG$lambda_mult * sqrt(log(p + q)/n)
  lam_node    <- CONFIG$node_mult   * sqrt(log(p)/n)
  cxx <- cov_decomp(X); cyy <- cov_decomp(Y)              # one eigendecomp each, shared below
  fit <- fit_admm(X, Y, lambda = lambda_admm, cxx = cxx, cyy = cyy,
                  Zinit = admm_init$Z, Hinit = admm_init$H)
  can <- canonical_dirs_r(fit$B, X, Y, r, cxx = cxx, cyy = cyy)
  Tx <- precision_fast(X, lam_node); Ty <- precision_fast(Y, lam_node)
  Ctil <- fit$B %*% fit$Sy + lambda_admm * Tx %*% fit$Zsub
  Dtil <- t(fit$B) %*% fit$Sx + lambda_admm * Ty %*% t(fit$Zsub)
  Ubc <- matrix(0, p, r); Vbc <- matrix(0, q, r); lam <- numeric(r)
  # Bias-corrected loadings Utilde^bc = Utilde - Uhat^bias (Def 7.7 / eqs 47-50).
  # Residual has diagonal (self) + off-diagonal piece, gap-weighted by 1/(lambda_l^2 - lambda_l'^2).
  # Keeping only the diagonal piece left an O(1) bias at r >= 2 (worse for the weaker component).
  lhat <- pmax(can$lambda, 1e-8)
  # Diagnostic-only oracle-lambda ablation (CONFIG$oracle_lam; default "off" = no-op).
  # lhat_gap is what the gap denominators see; lhat is what the scalings/multipliers see.
  # Component ORDER assumption: both can$lambda and lambda_star are in decreasing order,
  # so element l of one corresponds to element l of the other. If a fit ever returns
  # components out of order this ablation would misassign them -- it is a diagnostic,
  # not a shipped estimator, so we accept that and flag it.
  lhat_gap <- lhat
  if (!identical(CONFIG$oracle_lam, "off") && length(CONFIG$lambda_star) >= r) {
    lstar <- pmax(as.numeric(CONFIG$lambda_star)[seq_len(r)], 1e-8)
    lhat_gap <- lstar
    if (identical(CONFIG$oracle_lam, "all")) lhat <- lstar
  }
  ZB   <- t(can$U) %*% fit$Zsub %*% can$V          # r x r: ZB[a,b] = Uhat_.a' Zsub Vhat_.b
  for (l in 1:r) {
    u <- can$U[, l]; v <- can$V[, l]; ll <- lhat[l]
    Up <- as.numeric(Ctil %*% v)/ll; Vp <- as.numeric(Dtil %*% u)/ll
    biasU <- ZB[l, l] * u                          # diagonal (self) term
    biasV <- ZB[l, l] * v
    if (r > 1) for (lp in setdiff(seq_len(r), l)) {  # off-diagonal coupling, eqs (48),(50)
      gap <- lhat_gap[l]^2 - lhat_gap[lp]^2; if (abs(gap) < 1e-8) next
      biasU <- biasU - (ll * ZB[l, lp] + lhat[lp] * ZB[lp, l]) / gap * lhat[lp] * can$U[, lp]
      biasV <- biasV - (ll * ZB[lp, l] + lhat[lp] * ZB[l, lp]) / gap * lhat[lp] * can$V[, lp]
    }
    Ubc[, l] <- Up - (lambda_admm/ll) * biasU
    Vbc[, l] <- Vp - (lambda_admm/ll) * biasV
    lam[l] <- min(max(as.numeric(t(u) %*% fit$Sxy %*% v), 1e-6), 0.999)
  }
  out <- list(U = Ubc, V = Vbc, lambda = lam, Uhat = can$U, Vhat = can$V,
              admm = list(Z = fit$B, H = fit$H))
  if (!light) { out$Tx <- Tx; out$Ty <- Ty }
  out
}
# Raw ECCAR fit: canonical directions ONLY (no SCIO precision, no debiasing).
# Used by the ECCAR-raw bootstrap, which consumes only Uhat/Vhat. Removing the
# discarded SCIO + debiasing here is bit-identical (coordinate descent does not
# touch the RNG stream) and eliminates the per-bootstrap precision-solve work.
eccar_raw_fit <- function(X, Y, r, admm_init = NULL) {
  n <- nrow(X); lambda_admm <- CONFIG$lambda_mult * sqrt(log(p + q)/n)
  cxx <- cov_decomp(X); cyy <- cov_decomp(Y)
  fit <- fit_admm(X, Y, lambda = lambda_admm, cxx = cxx, cyy = cyy,
                  Zinit = admm_init$Z, Hinit = admm_init$H)
  can <- canonical_dirs_r(fit$B, X, Y, r, cxx = cxx, cyy = cyy)
  lam <- numeric(r)
  for (l in 1:r) lam[l] <- min(max(as.numeric(t(can$U[, l]) %*% fit$Sxy %*% can$V[, l]), 1e-6), 0.999)
  list(U = can$U, V = can$V, lambda = lam)
}

## --- rank-1 building blocks for competitor methods (operate on Sxy) ---
witten_rank1 <- function(Sxy, c_u, c_v, iter = CONFIG$comp_iter, tol = CONFIG$comp_tol) {
  bisect <- function(z, ct, mi = 30) { zn <- sqrt(sum(z^2)); if (zn < 1e-10) return(z); u <- z/zn
    if (sum(abs(u)) <= ct) return(z); lo <- 0; hi <- max(abs(z))
    for (i in 1:mi) { t <- (lo + hi)/2; zt <- soft(z, t); nt <- sqrt(sum(zt^2)); if (nt < 1e-10) { hi <- t; next }
      if (sum(abs(zt))/nt > ct) lo <- t else hi <- t }; soft(z, (lo + hi)/2) }
  ss <- svd(Sxy, nu = 1, nv = 1); u <- as.numeric(ss$u); v <- as.numeric(ss$v)
  for (it in 1:iter) { uo <- u; vo <- v
    u <- as.numeric(bisect(Sxy %*% v, c_u)); nu <- sqrt(sum(u^2)); if (nu < 1e-10) u <- uo else u <- u/nu
    v <- as.numeric(bisect(crossprod(Sxy, u), c_v)); nv <- sqrt(sum(v^2)); if (nv < 1e-10) v <- vo else v <- v/nv
    if (sqrt(sum((u - uo)^2)) + sqrt(sum((v - vo)^2)) < tol) break }
  list(u = u, v = v)
}
waaijenborg_rank1 <- function(Sxy, lam_u, lam_v, iter = CONFIG$comp_iter, tol = CONFIG$comp_tol) {
  ss <- svd(Sxy, nu = 1, nv = 1); u <- as.numeric(ss$u); v <- as.numeric(ss$v)
  for (it in 1:iter) { uo <- u; vo <- v
    u <- soft(as.numeric(Sxy %*% v), lam_u); nu <- sqrt(sum(u^2)); if (nu < 1e-10) u <- uo else u <- u/nu
    v <- soft(as.numeric(crossprod(Sxy, u)), lam_v); nv <- sqrt(sum(v^2)); if (nv < 1e-10) v <- vo else v <- v/nv
    if (sqrt(sum((u - uo)^2)) + sqrt(sum((v - vo)^2)) < tol) break }
  list(u = u, v = v)
}
parkhomenko_rank1 <- function(Sxy, Sxinv, Syinv, Sx, Sy, lam_u, lam_v, iter = CONFIG$comp_iter, tol = CONFIG$comp_tol) {
  M <- Sxinv %*% Sxy %*% Syinv; ss <- svd(M, nu = 1, nv = 1); u <- as.numeric(ss$u); v <- as.numeric(ss$v)
  nu <- sqrt(as.numeric(t(u) %*% Sx %*% u)); u <- u/max(nu, 1e-10)
  nv <- sqrt(as.numeric(t(v) %*% Sy %*% v)); v <- v/max(nv, 1e-10)
  for (it in 1:iter) { uo <- u; vo <- v
    u <- soft(as.numeric(Sxinv %*% Sxy %*% v), lam_u); nu <- sqrt(as.numeric(t(u) %*% Sx %*% u)); if (nu < 1e-10) u <- uo else u <- u/nu
    v <- soft(as.numeric(Syinv %*% t(Sxy) %*% u), lam_v); nv <- sqrt(as.numeric(t(v) %*% Sy %*% v)); if (nv < 1e-10) v <- vo else v <- v/nv
    if (sqrt(sum((u - uo)^2)) + sqrt(sum((v - vo)^2)) < tol) break }
  list(u = u, v = v)
}
# Generic deflation: estimate r components by repeatedly fitting rank-1 on the
# deflated cross-covariance Sxy <- Sxy - lam * Sx u v' Sy, then Sx/Sy-normalize.
deflate_fit <- function(X, Y, r, rank1_fn) {
  n <- nrow(X); Sx <- crossprod(X)/n; Sy <- crossprod(Y)/n; Sxy <- crossprod(X, Y)/n
  U <- matrix(0, ncol(X), r); V <- matrix(0, ncol(Y), r); lam <- numeric(r); Sxy_d <- Sxy
  for (l in 1:r) {
    f <- rank1_fn(Sxy_d, Sx, Sy)
    u <- f$u; v <- f$v
    nu <- sqrt(as.numeric(t(u) %*% Sx %*% u)); u <- u/max(nu, 1e-10)
    nv <- sqrt(as.numeric(t(v) %*% Sy %*% v)); v <- v/max(nv, 1e-10)
    ll <- as.numeric(t(u) %*% Sxy %*% v)
    U[, l] <- u; V[, l] <- v; lam[l] <- ll
    Sxy_d <- Sxy_d - ll * (Sx %*% u) %*% (t(v) %*% Sy)   # deflate
  }
  list(U = U, V = V, lambda = lam)
}
witten_pmd_r   <- function(X, Y, r) deflate_fit(X, Y, r, function(Sxy, Sx, Sy) witten_rank1(Sxy, sqrt(s_u), sqrt(s_v)))
waaijenborg_r  <- function(X, Y, r) { n <- nrow(X); lu <- CONFIG$comp_lambda_mult*sqrt(log(p)/n); lv <- CONFIG$comp_lambda_mult*sqrt(log(q)/n)
  deflate_fit(X, Y, r, function(Sxy, Sx, Sy) waaijenborg_rank1(Sxy, lu, lv)) }
parkhomenko_r  <- function(X, Y, r, ridge = CONFIG$park_ridge) { n <- nrow(X); lu <- CONFIG$comp_lambda_mult*sqrt(log(p)/n); lv <- CONFIG$comp_lambda_mult*sqrt(log(q)/n)
  cxx <- cov_decomp(X); cyy <- cov_decomp(Y)
  Sx <- cxx$S + ridge * diag(p); Sy <- cyy$S + ridge * diag(q)
  Sxinv <- if (isTRUE(cxx$thin)) ridge_inv(cxx, ridge) else solve(Sx)
  Syinv <- if (isTRUE(cyy$thin)) ridge_inv(cyy, ridge) else solve(Sy)
  deflate_fit(X, Y, r, function(Sxy, Sx2, Sy2) parkhomenko_rank1(Sxy, Sxinv, Syinv, Sx2, Sy2, lu, lv)) }

# Gao-Ma: native top-r whitened thresholded SVD
gao_ma_r <- function(X, Y, r, iter = CONFIG$comp_iter, tol = CONFIG$comp_tol, ridge = CONFIG$gm_ridge) {
  n <- nrow(X); cxx <- cov_decomp(X); cyy <- cov_decomp(Y)
  Sx <- cxx$S + ridge * diag(p); Sy <- cyy$S + ridge * diag(q); Sxy <- crossprod(X, Y)/n
  Sxh_inv <- if (isTRUE(cxx$thin)) ridge_invsqrt(cxx, ridge) else sym_sqrt(solve(Sx))
  Syh_inv <- if (isTRUE(cyy$thin)) ridge_invsqrt(cyy, ridge) else sym_sqrt(solve(Sy))
  M <- Sxh_inv %*% Sxy %*% Syh_inv; ss <- svd(M, nu = r, nv = r)
  A <- ss$u[, 1:r, drop = FALSE]; Bb <- ss$v[, 1:r, drop = FALSE]
  for (it in 1:iter) {
    Ao <- A; Za <- M %*% Bb
    for (l in 1:r) { idx <- order(abs(Za[, l]), decreasing = TRUE)[1:min(s_u, p)]; col <- numeric(p); col[idx] <- Za[idx, l]; A[, l] <- col }
    A <- qr.Q(qr(A))
    Zb <- crossprod(M, A)
    for (l in 1:r) { idx <- order(abs(Zb[, l]), decreasing = TRUE)[1:min(s_v, q)]; col <- numeric(q); col[idx] <- Zb[idx, l]; Bb[, l] <- col }
    Bb <- qr.Q(qr(Bb))
    if (sqrt(sum((A - Ao)^2)) < tol) break
  }
  U <- matrix(0, p, r); V <- matrix(0, q, r); lam <- numeric(r)
  for (l in 1:r) {
    u <- as.numeric(Sxh_inv %*% A[, l]); v <- as.numeric(Syh_inv %*% Bb[, l])
    nu <- sqrt(as.numeric(t(u) %*% Sx %*% u)); nv <- sqrt(as.numeric(t(v) %*% Sy %*% v))
    u <- u/max(nu, 1e-10); v <- v/max(nv, 1e-10)
    U[, l] <- u; V[, l] <- v; lam[l] <- as.numeric(t(u) %*% Sxy %*% v)
  }
  list(U = U, V = V, lambda = lam)
}
# Wilms-Croux: alternating elastic-net, deflate responses by removing fitted comps
wilms_croux_r <- function(X, Y, r, alpha_en = CONFIG$wilms_alpha, iter = CONFIG$wilms_iter, tol = CONFIG$comp_tol) {
  n <- nrow(X); lu <- CONFIG$wilms_lambda_mult * sqrt(log(p)/n); lv <- CONFIG$wilms_lambda_mult * sqrt(log(q)/n)
  Sx <- crossprod(X)/n; Sy <- crossprod(Y)/n; Sxy0 <- crossprod(X, Y)/n
  U <- matrix(0, p, r); V <- matrix(0, q, r); lam <- numeric(r); Sxy <- Sxy0
  for (l in 1:r) {
    ss <- svd(Sxy, nu = 1, nv = 1); u <- as.numeric(ss$u); v <- as.numeric(ss$v)
    for (it in 1:iter) { uo <- u; vo <- v
      fu <- tryCatch(glmnet::glmnet(X, as.numeric(Y %*% v), alpha = alpha_en, intercept = FALSE, standardize = FALSE, lambda = lu, thresh = 1e-4), error = function(e) NULL)
      if (is.null(fu)) break; u <- as.numeric(coef(fu))[-1]; nu <- sqrt(sum(u^2)); if (nu < 1e-10) u <- uo else u <- u/nu
      fv <- tryCatch(glmnet::glmnet(Y, as.numeric(X %*% u), alpha = alpha_en, intercept = FALSE, standardize = FALSE, lambda = lv, thresh = 1e-4), error = function(e) NULL)
      if (is.null(fv)) break; v <- as.numeric(coef(fv))[-1]; nv <- sqrt(sum(v^2)); if (nv < 1e-10) v <- vo else v <- v/nv
      if (sqrt(sum((u - uo)^2)) + sqrt(sum((v - vo)^2)) < tol) break }
    nu <- sqrt(as.numeric(t(u) %*% Sx %*% u)); nv <- sqrt(as.numeric(t(v) %*% Sy %*% v))
    u <- u/max(nu, 1e-10); v <- v/max(nv, 1e-10); ll <- as.numeric(t(u) %*% Sxy0 %*% v)
    U[, l] <- u; V[, l] <- v; lam[l] <- ll
    Sxy <- Sxy - ll * (Sx %*% u) %*% (t(v) %*% Sy)
  }
  list(U = U, V = V, lambda = lam)
}

## ============================ INFERENCE ====================================
# Bootstrap CI for rank-r: each replicate's components matched to the reference.
bootstrap_ci_r <- function(X, Y, est_fn, U_ref, V_ref, Sig_x, Sig_y, alpha = 0.05, B = 30) {
  n <- nrow(X)
  idx_list <- lapply(seq_len(B), function(b) sample.int(n, replace = TRUE))  # pre-drawn (serial RNG => reproducible)
  one_boot <- function(idx) tryCatch({
    Xb <- scale(X[idx, , drop = FALSE], scale = FALSE); Yb <- scale(Y[idx, , drop = FALSE], scale = FALSE)
    fb <- est_fn(Xb, Yb); if (is.null(fb)) return(NULL)
    m <- match_components(fb$U, U_ref, Sig_x)
    list(U = apply_match(fb$U, m), V = apply_match(fb$V, m), lam = fb$lambda[m$perm])
  }, error = function(e) NULL)
  res <- par_lapply(idx_list, one_boot, cores = CONFIG$inner_cores)   # est_fn is deterministic => parallel-safe
  Us <- array(NA, c(B, p, r)); Vs <- array(NA, c(B, q, r)); lams <- matrix(NA, B, r)
  for (b in seq_len(B)) { rb <- res[[b]]
    if (is.null(rb) || inherits(rb, "try-error")) next
    Us[b, , ] <- rb$U; Vs[b, , ] <- rb$V; lams[b, ] <- rb$lam }
  qlo <- function(A) apply(A, c(2, 3), quantile, probs = alpha/2, na.rm = TRUE)      # -> [coord, comp]
  qhi <- function(A) apply(A, c(2, 3), quantile, probs = 1 - alpha/2, na.rm = TRUE)
  list(U_lo = qlo(Us), U_hi = qhi(Us),
       V_lo = qlo(Vs), V_hi = qhi(Vs),
       lam_lo = apply(lams, 2, quantile, probs = alpha/2, na.rm = TRUE),
       lam_hi = apply(lams, 2, quantile, probs = 1 - alpha/2, na.rm = TRUE))
}
# plug-in asymptotic variance (per component), shared by ECCAR+CF and ECCAR+split
plugin_var <- function(Ubc, Vbc, lam, Tx, Ty) {
  # Closed-form plug-in variance, Def 9.1 eq (67)-(68). lam = lambda-tilde (Anderson, eq 39);
  # U/V = bias-corrected loadings. Tx, Ty are the symmetrized SCIO precision estimates
  # (diag(Tx) = (Theta_X)_jj enters the leading term, exactly as nodewise did).
  #   sigma^2_{j,l} = (Theta_X)_jj (1-lam_l^2)/lam_l^2  +  U_{j,l}^2 (3/2 - 1/lam_l^2)
  #                 + sum_{m != l} U_{j,m}^2 nu_{l,m},
  # with nu_{l,m} = lam_m^2 (1-lam_l^2) [lam_l^2 (3 - 2 lam_l^2) - lam_m^2] / (lam_l^2 (lam_l^2 - lam_m^2)^2).
  vU <- matrix(0, p, r); vV <- matrix(0, q, r); l2 <- lam^2
  dTx <- diag(Tx); dTy <- diag(Ty)
  nu <- function(ll2, lm2) { g <- ll2 - lm2; if (abs(g) < 1e-8) return(0)
    lm2 * (1 - ll2) * (ll2 * (3 - 2 * ll2) - lm2) / (ll2 * g^2) }
  for (l in 1:r) {
    ll2 <- l2[l]; selfc <- 1.5 - 1/ll2
    crossU <- numeric(p); crossV <- numeric(q)
    if (r > 1) for (m in setdiff(seq_len(r), l)) {
      cf <- nu(ll2, l2[m]); crossU <- crossU + Ubc[, m]^2 * cf; crossV <- crossV + Vbc[, m]^2 * cf
    }
    vU[, l] <- pmax(dTx * (1 - ll2)/ll2 + Ubc[, l]^2 * selfc + crossU, 1e-6)
    vV[, l] <- pmax(dTy * (1 - ll2)/ll2 + Vbc[, l]^2 * selfc + crossV, 1e-6)
  }
  list(vU = vU, vV = vV, vlam = (1 - lam^2)^2)
}
make_ci <- function(Ubc, Vbc, lam, vU, vV, vlam, n, alpha) {
  z <- qnorm(1 - alpha/2)
  list(U_lo = Ubc - z * sqrt(vU/n), U_hi = Ubc + z * sqrt(vU/n),
       V_lo = Vbc - z * sqrt(vV/n), V_hi = Vbc + z * sqrt(vV/n),
       lam_lo = lam - z * sqrt(vlam/n), lam_hi = lam + z * sqrt(vlam/n))
}
# ECCAR closed-form (plug-in only) and split-sample (plug-in + bump), both rank-r
eccar_cis_r <- function(X, Y, full_fit, K = 8, alpha = 0.05) {
  n <- nrow(X); half <- n %/% 2
  Ubc <- full_fit$U; Vbc <- full_fit$V; lam <- full_fit$lambda; Tx <- full_fit$Tx; Ty <- full_fit$Ty
  pv <- plugin_var(Ubc, Vbc, lam, Tx, Ty)
  ci_cf <- make_ci(Ubc, Vbc, lam, pv$vU, pv$vV, pv$vlam, n, alpha)     # closed-form
  # split-sample finite-sample bump (per coord/component). Folds are independent and,
  # with pre-drawn permutations, deterministic -> run them in parallel within the rep.
  perms <- lapply(seq_len(K), function(k) sample.int(n))
  one_fold <- function(perm) tryCatch({
    A <- perm[1:half]; Bb <- perm[(half + 1):(2 * half)]
    fA <- eccar_fit_r(X[A, ], Y[A, ], r, light = TRUE); fB <- eccar_fit_r(X[Bb, ], Y[Bb, ], r, light = TRUE)
    mA <- match_components(fA$U, Ubc, SigmaX); mB <- match_components(fB$U, Ubc, SigmaX)
    UA <- apply_match(fA$U, mA); UB <- apply_match(fB$U, mB)
    VA <- apply_match(fA$V, mA); VB <- apply_match(fB$V, mB)
    list(SU = sqrt(n/4) * (UA - UB), SV = sqrt(n/4) * (VA - VB),
         SL = sqrt(n/4) * (fA$lambda[mA$perm] - fB$lambda[mB$perm]))
  }, error = function(e) NULL)
  folds <- par_lapply(perms, one_fold, cores = CONFIG$inner_cores)
  folds <- Filter(function(f) !is.null(f) && !inherits(f, "try-error"), folds)
  Kok <- length(folds)
  if (Kok == 0L) { return(list(cf = ci_cf, split = ci_cf)) }   # all folds failed -> fall back to closed-form
  SU <- array(0, c(Kok, p, r)); SV <- array(0, c(Kok, q, r)); SL <- matrix(0, Kok, r)
  for (k in seq_len(Kok)) { SU[k, , ] <- folds[[k]]$SU; SV[k, , ] <- folds[[k]]$SV; SL[k, ] <- folds[[k]]$SL }
  # Split-sample, Def 9.4 / eq (77)-(78):
  #   sigma^2_full = sigma^2_plugin + [(1/K) sum_k D_k^2 - sigma^2_plugin]_+
  #   D_k = sqrt(n/4) (Ubc^(Ak) - Ubc^(Bk)).
  # split_cap>0 (non-paper) Winsorizes folds at split_cap x median.
  cap <- CONFIG$split_cap
  agg <- function(x) { if (cap > 0 && length(x) >= 3) { md <- median(x); if (md > 0) x <- pmin(x, cap * md) }; mean(x) }
  cU <- apply(SU^2, c(2, 3), agg); cV <- apply(SV^2, c(2, 3), agg); cL <- apply(SL^2, 2, agg)
  # OPTIONAL (non-paper) finite-sample deflation; delta_n = n/(n+k), k=0 => paper Def 9.4.
  delta_n <- n / (n + CONFIG$split_shrink_k)
  vU <- pv$vU + delta_n * pmax(cU - pv$vU, 0); vV <- pv$vV + delta_n * pmax(cV - pv$vV, 0)
  vlam <- pv$vlam + delta_n * pmax(cL - pv$vlam, 0)
  ci_split <- make_ci(Ubc, Vbc, lam, vU, vV, vlam, n, alpha)
  list(cf = ci_cf, split = ci_split)
}
`%||%` <- function(a, b) if (is.null(a)) b else a

## ----------------------- one Monte Carlo replicate -------------------------
one_rep <- function(n) {
  d <- gen_data(n); X <- d$X; Y <- d$Y
  tm <- numeric(0)
  .tic <- function() proc.time()["elapsed"]

  t0 <- .tic()
  fe <- eccar_fit_r(X, Y, r)
  mE <- match_components(fe$U, Ustar, SigmaX)
  fe$U <- apply_match(fe$U, mE); fe$V <- apply_match(fe$V, mE); fe$lambda <- fe$lambda[mE$perm]
  fe$Uhat <- apply_match(fe$Uhat, mE); fe$Vhat <- apply_match(fe$Vhat, mE)
  eccar_cis <- eccar_cis_r(X, Y, fe, K = CONFIG$K, alpha = alpha)
  tm["ECCAR + split (ours)"] <- tm["ECCAR + closed-form"] <- .tic() - t0

  # ECCAR-raw bootstrap
  t0 <- .tic()
  eccar_raw_fn <- function(Xb, Yb) eccar_raw_fit(Xb, Yb, r)
  ci_eccar_boot <- bootstrap_ci_r(X, Y, eccar_raw_fn, fe$Uhat, fe$Vhat, SigmaX, SigmaY, alpha, CONFIG$B)
  tm["ECCAR-raw + bootstrap"] <- .tic() - t0

  # Witten: fit once, reuse for its own bootstrap reference, Laha init, subspace
  t0 <- .tic()
  fw <- witten_pmd_r(X, Y, r); mw <- match_components(fw$U, Ustar, SigmaX)
  fw$U <- apply_match(fw$U, mw); fw$V <- apply_match(fw$V, mw)
  ci_w_boot <- bootstrap_ci_r(X, Y, function(Xb, Yb) witten_pmd_r(Xb, Yb, r),
                              fw$U, fw$V, SigmaX, SigmaY, alpha, CONFIG$B)
  tm["Witten PMD + bootstrap"] <- .tic() - t0

  t0 <- .tic()
  # Laha: rank-1 method. Component 1 only, one-step debias of leading Witten pair,
  # rank-1 closed-form variance (leading + self; no cross term). Comps >= 2 stay NA.
  # Tx, Ty here are the symmetrized SCIO precision estimates from the ECCAR fit.
  Sx <- crossprod(X)/n; Sy <- crossprod(Y)/n; Sxy <- crossprod(X, Y)/n
  U0 <- fw$U; V0 <- fw$V; Tx <- fe$Tx; Ty <- fe$Ty
  UL <- matrix(NA_real_, p, r); VL <- matrix(NA_real_, q, r); lamL <- rep(NA_real_, r)
  vUL <- matrix(NA_real_, p, r); vVL <- matrix(NA_real_, q, r); vlamL <- rep(NA_real_, r)
  lam0 <- min(max(as.numeric(t(U0[, 1]) %*% Sxy %*% V0[, 1]), 1e-6), 0.999)
  UL[, 1] <- as.numeric(U0[, 1] + Tx %*% (Sxy %*% V0[, 1] - lam0 * Sx %*% U0[, 1])/lam0)
  VL[, 1] <- as.numeric(V0[, 1] + Ty %*% (t(Sxy) %*% U0[, 1] - lam0 * Sy %*% V0[, 1])/lam0)
  lamL[1] <- min(max(as.numeric(t(UL[, 1]) %*% Sxy %*% VL[, 1]), 1e-6), 0.999)
  l2L <- lamL[1]^2; scL <- 1.5 - 1/l2L
  vUL[, 1] <- pmax(diag(Tx) * (1 - l2L)/l2L + UL[, 1]^2 * scL, 1e-6)
  vVL[, 1] <- pmax(diag(Ty) * (1 - l2L)/l2L + VL[, 1]^2 * scL, 1e-6)
  vlamL[1] <- (1 - l2L)^2
  ci_laha <- make_ci(UL, VL, lamL, vUL, vVL, vlamL, n, alpha)
  tm["Laha + closed-form"] <- .tic() - t0

  # remaining bootstrap methods: fit once on full data, reuse fit for reference
  # AND subspace error; only the bootstrap loop re-fits (inherently).
  t0 <- .tic()
  fit_p <- parkhomenko_r(X, Y, r); mp <- match_components(fit_p$U, Ustar, SigmaX)
  ci_p_boot  <- bootstrap_ci_r(X, Y, function(a, b) parkhomenko_r(a, b, r), apply_match(fit_p$U, mp), apply_match(fit_p$V, mp), SigmaX, SigmaY, alpha, CONFIG$B)
  tm["Parkhomenko + bootstrap"] <- .tic() - t0
  t0 <- .tic()
  fit_wa <- waaijenborg_r(X, Y, r); mwa <- match_components(fit_wa$U, Ustar, SigmaX)
  ci_wa_boot <- bootstrap_ci_r(X, Y, function(a, b) waaijenborg_r(a, b, r), apply_match(fit_wa$U, mwa), apply_match(fit_wa$V, mwa), SigmaX, SigmaY, alpha, CONFIG$B)
  tm["Waaijenborg + bootstrap"] <- .tic() - t0
  t0 <- .tic()
  fit_gm <- gao_ma_r(X, Y, r); mgm <- match_components(fit_gm$U, Ustar, SigmaX)
  ci_gm_boot <- bootstrap_ci_r(X, Y, function(a, b) gao_ma_r(a, b, r), apply_match(fit_gm$U, mgm), apply_match(fit_gm$V, mgm), SigmaX, SigmaY, alpha, CONFIG$B)
  tm["Gao-Ma + bootstrap"] <- .tic() - t0
  t0 <- .tic()
  fit_wc <- wilms_croux_r(X, Y, r); mwc <- match_components(fit_wc$U, Ustar, SigmaX)
  ci_wc_boot <- bootstrap_ci_r(X, Y, function(a, b) wilms_croux_r(a, b, r), apply_match(fit_wc$U, mwc), apply_match(fit_wc$V, mwc), SigmaX, SigmaY, alpha, CONFIG$B)
  tm["Wilms-Croux + bootstrap"] <- .tic() - t0

  # subspace distance (sin-Theta, Sigma inner product) of the Sigma-orthonormalized
  # canonical directions. ECCAR is reported BOTH raw (Uhat) and debiased (Ubc).
  subU <- c(
    eccar_raw = subspace_dist(fe$Uhat, Ustar, SigmaX),
    eccar_bc  = subspace_dist(fe$U,    Ustar, SigmaX),
    witten    = subspace_dist(fw$U,    Ustar, SigmaX),
    parkho    = subspace_dist(fit_p$U, Ustar, SigmaX),
    waaij     = subspace_dist(fit_wa$U, Ustar, SigmaX),
    gaoma     = subspace_dist(fit_gm$U, Ustar, SigmaX),
    wilms     = subspace_dist(fit_wc$U, Ustar, SigmaX))

  list(ci_eccar_split = eccar_cis$split, ci_eccar_cf = eccar_cis$cf,
       ci_laha = ci_laha, ci_eccar_boot = ci_eccar_boot, ci_w_boot = ci_w_boot,
       ci_p_boot = ci_p_boot, ci_wa_boot = ci_wa_boot, ci_gm_boot = ci_gm_boot,
       ci_wc_boot = ci_wc_boot, subU = subU, timing = tm)
}

## ---- runner ----
# n_cores=1: serial loop, BLAS gets all cores. n_cores>1: mclapply with fixed
# L'Ecuyer substreams (reproducible given seed + n_cores), 1 BLAS thread per worker.
run_one_n <- function(n, M) {
  if (M <= 0) return(NULL)
  fn <- file.path(CONFIG$outdir, sprintf("scca_rankr_n%d.rds", n))
  # Self-describing checkpoint, only written when --checkpoint is set.
  save_ckpt <- function(reps, n) if (CONFIG$checkpoint) saveRDS(list(
    reps = reps, n = n, p = p, q = q, r = r,
    Ustar = Ustar, Vstar = Vstar, lambda_star = lambda_star), fn)
  # Complete checkpoint (>= M reps) is reused; a partial one is RESUMED -- only the
  # missing indices are run. Safe because rep m's stream does not depend on chunking.
  prev <- NULL; start_at <- 1L
  if (CONFIG$checkpoint && file.exists(fn)) {
    ck <- tryCatch(readRDS(fn), error = function(e) NULL)
    # Reusable only at the SAME (p, q, r, lambda*): resuming across a dimension
    # change would splice CI matrices of different sizes into one rep list.
    if (!is.null(ck) && !is.null(ck$p) &&
        !(identical(as.integer(ck$p), as.integer(p)) &&
          identical(as.integer(ck$q), as.integer(q)) &&
          identical(as.integer(ck$r), as.integer(r)) &&
          isTRUE(all.equal(as.numeric(ck$lambda_star), as.numeric(lambda_star))))) {
      cat(sprintf(paste0("n=%d: checkpoint in %s was built at p=%s q=%s r=%s lambda*=%s\n",
                         "       but this run is p=%d q=%d r=%d lambda*=%s -- IGNORING it ",
                         "and recomputing.\n"),
                  n, basename(fn), ck$p, ck$q, ck$r, paste(ck$lambda_star, collapse = ","),
                  p, q, r, paste(lambda_star, collapse = ",")))
      flush.console()
      ck <- NULL
    }
    prev <- if (is.null(ck)) NULL else ck$reps
    if (!is.null(prev) && length(prev) >= M) {
      cat(sprintf("n=%d has >= %d reps in checkpoint; reusing.\n", n, M)); return(prev)
    }
    if (!is.null(prev) && length(prev) > 0L) {
      # count only reps that actually succeeded; a failed slot is redone
      ok <- vapply(prev, function(rp) is.list(rp) && !inherits(rp, "try-error"), logical(1))
      start_at <- length(prev) + 1L
      cat(sprintf("n=%d: RESUMING from checkpoint with %d rep(s) (%d usable); running %d..%d\n",
                  n, length(prev), sum(ok), start_at, M)); flush.console()
    }
  }
  t0 <- Sys.time()

  if (CONFIG$n_cores <= 1) {
    set_blas(CONFIG$blas_threads)            # all cores to BLAS in serial mode
    set.seed(CONFIG$seed_base + n)
    cat(sprintf("Running n=%d, M=%d [serial MC, %d within-rep workers, BLAS=%d] ...\n",
                n, M, CONFIG$inner_cores, CONFIG$blas_threads)); flush.console()
    reps <- vector("list", M)
    if (!is.null(prev) && length(prev)) reps[seq_along(prev)] <- prev
    if (start_at > M) return(reps)
    for (m in seq(start_at, M)) {
      reps[[m]] <- one_rep(n); save_ckpt(reps[1:m], n)
      cat(sprintf("  rep %d/%d (%.0fs)\n", m, M, as.numeric(difftime(Sys.time(), t0, units = "secs")))); flush.console()
    }
    return(reps)
  }

  # ---- parallel MC path ----
  # Pre-derive M independent, FIXED L'Ecuyer streams from the master seed so the
  # stream for rep m does not depend on how reps are chunked across workers.
  RNGkind("L'Ecuyer-CMRG"); set.seed(CONFIG$seed_base + n)
  streams <- vector("list", M); s <- .Random.seed
  for (m in seq_len(M)) { streams[[m]] <- s; s <- parallel::nextRNGStream(s) }
  run_rep <- function(m) {
    set_blas(CONFIG$blas_threads)            # pin BLAS inside each forked worker
    assign(".Random.seed", streams[[m]], envir = .GlobalEnv)
    one_rep(n)
  }
  cat(sprintf("Running n=%d, M=%d [parallel: %d workers x %d BLAS threads] ...\n",
              n, M, CONFIG$n_cores, CONFIG$blas_threads)); flush.console()
  reps <- vector("list", M); done <- 0L
  if (!is.null(prev) && length(prev)) { reps[seq_along(prev)] <- prev; done <- length(prev) }
  if (start_at > M) return(reps)
  todo <- seq(start_at, M)                                          # only the missing reps
  chunks <- split(todo, ceiling(seq_along(todo)/CONFIG$n_cores))    # checkpoint per chunk
  for (ch in chunks) {
    res <- parallel::mclapply(ch, run_rep, mc.cores = CONFIG$n_cores, mc.preschedule = FALSE)
    for (i in seq_along(ch)) reps[[ch[i]]] <- res[[i]]
    done <- max(ch); save_ckpt(reps[seq_len(done)], n)
    cat(sprintf("  %d/%d reps done (%.0fs)\n", done, M, as.numeric(difftime(Sys.time(), t0, units = "secs")))); flush.console()
  }
  reps
}

## ---------------------------- aggregator -----------------------------------
# Per-component coverage / Type I / power. CI fields are matrices [coord, comp]
# (loadings) or length-r vectors (lambda).
metrics_component <- function(ci, truthU, truthV, truthLam) {
  out <- list(); rr <- ncol(truthU)
  for (l in 1:rr) {
    covU <- (truthU[, l] >= ci$U_lo[, l]) & (truthU[, l] <= ci$U_hi[, l])
    covV <- (truthV[, l] >= ci$V_lo[, l]) & (truthV[, l] <= ci$V_hi[, l])
    rejU <- (0 < ci$U_lo[, l]) | (0 > ci$U_hi[, l])
    rejV <- (0 < ci$V_lo[, l]) | (0 > ci$V_hi[, l])
    out[[l]] <- list(covU = covU, covV = covV, rejU = rejU, rejV = rejV,
                     covLam = (truthLam[l] >= ci$lam_lo[l]) & (truthLam[l] <= ci$lam_hi[l]),
                     rejLam = (0 < ci$lam_lo[l]) | (0 > ci$lam_hi[l]))
  }
  out
}
# Component is "defined" by an estimator iff some replicate's CI is non-NA.
# Rank-1 methods (Laha) define only the leading pair.
comp_defined <- function(reps, field, l) {
  for (m in seq_along(reps)) {
    rp <- reps[[m]]; if (is.null(rp)) next
    ci <- rp[[field]]
    if (is.null(ci) || is.null(ci$U_lo) || ncol(ci$U_lo) < l) next
    if (any(!is.na(ci$U_lo[, l]))) return(TRUE)
  }
  FALSE
}
extract_est <- function(reps, n, field, label, TR) {
  M <- length(reps); rows <- list()
  Ustar <- TR$Ustar; Vstar <- TR$Vstar; lambda_star <- TR$lambda_star
  p <- TR$p; q <- TR$q; r <- TR$r
  for (l in 1:r) {
    if (!comp_defined(reps, field, l)) next   # skip components an estimator does not define (e.g. Laha rank-1: only l=1)
    cU <- matrix(NA, M, p); cV <- matrix(NA, M, q); rU <- matrix(NA, M, p); rV <- matrix(NA, M, q)
    cL <- numeric(M); rL <- numeric(M)
    for (m in 1:M) { ci <- reps[[m]][[field]]; if (is.null(ci)) next
      mm <- metrics_component(ci, Ustar, Vstar, lambda_star)[[l]]
      cU[m, ] <- mm$covU; cV[m, ] <- mm$covV; rU[m, ] <- mm$rejU; rV[m, ] <- mm$rejV; cL[m] <- mm$covLam; rL[m] <- mm$rejLam }
    rows[[length(rows) + 1]] <- bind_rows(
      tibble(side = "U", comp = l, coord = seq_len(p), signal_magnitude = abs(Ustar[, l]),
             status = ifelse(Ustar[, l] == 0, "null", "signal"),
             coverage = colMeans(cU, na.rm = TRUE), rejection = colMeans(rU, na.rm = TRUE)),
      tibble(side = "V", comp = l, coord = seq_len(q), signal_magnitude = abs(Vstar[, l]),
             status = ifelse(Vstar[, l] == 0, "null", "signal"),
             coverage = colMeans(cV, na.rm = TRUE), rejection = colMeans(rV, na.rm = TRUE)),
      tibble(side = "lam", comp = l, coord = NA_integer_, signal_magnitude = abs(lambda_star[l]),
             status = "signal", coverage = mean(cL, na.rm = TRUE), rejection = mean(rL, na.rm = TRUE)))
  }
  bind_rows(rows) %>% mutate(estimator = label, n = n)
}
ESTIMATORS <- c(ci_eccar_split = "ECCAR + split (ours)", ci_eccar_cf = "ECCAR + closed-form",
                ci_laha = "Laha + closed-form", ci_eccar_boot = "ECCAR-raw + bootstrap",
                ci_w_boot = "Witten PMD + bootstrap", ci_p_boot = "Parkhomenko + bootstrap",
                ci_wa_boot = "Waaijenborg + bootstrap", ci_gm_boot = "Gao-Ma + bootstrap",
                ci_wc_boot = "Wilms-Croux + bootstrap")

# Size-adjusted power + CI width per component. T = |center|/(half-width/z);
# threshold = 95th percentile of T over null coords (size-calibrated to 0.05).
extract_stats <- function(reps, n, field, label, TR) {
  z975 <- qnorm(0.975); M <- length(reps); rows <- list()
  Ustar <- TR$Ustar; Vstar <- TR$Vstar; r <- TR$r
  side_row <- function(l, side_lab, lo_field, hi_field, Star) {
    Tnull <- c(); Tsig <- c(); Wnull <- c(); Wsig <- c()
    for (m in 1:M) {
      ci <- reps[[m]][[field]]; if (is.null(ci)) next
      lo <- ci[[lo_field]][, l]; hi <- ci[[hi_field]][, l]
      if (all(is.na(lo))) next
      width <- hi - lo; center <- (lo + hi)/2
      halfw <- pmax(width/2, 1e-12)
      Tstat <- abs(center)/(halfw/z975)
      nm <- Star[, l] == 0
      Tnull <- c(Tnull, Tstat[nm]);  Tsig <- c(Tsig, Tstat[!nm])
      Wnull <- c(Wnull, width[nm]);  Wsig <- c(Wsig, width[!nm])
    }
    c_thr <- quantile(Tnull, 0.95, na.rm = TRUE)
    tibble(estimator = label, n = n, comp = l, side = side_lab,
           size_adj_power = mean(Tsig > c_thr, na.rm = TRUE),
           width_null = median(Wnull, na.rm = TRUE),
           width_signal = median(Wsig, na.rm = TRUE))
  }
  for (l in 1:r) {
    if (!comp_defined(reps, field, l)) next
    rows[[length(rows) + 1]] <- side_row(l, "U", "U_lo", "U_hi", Ustar)
    rows[[length(rows) + 1]] <- side_row(l, "V", "V_lo", "V_hi", Vstar)
    # lambda-tilde CI width: one value per rep per component, no null status
    Wlam <- numeric(0)
    for (m in 1:M) {
      ci <- reps[[m]][[field]]; if (is.null(ci)) next
      if (!is.null(ci$lam_lo) && !is.na(ci$lam_lo[l]))
        Wlam <- c(Wlam, ci$lam_hi[l] - ci$lam_lo[l])
    }
    if (length(Wlam))
      rows[[length(rows) + 1]] <- tibble(estimator = label, n = n, comp = l, side = "lam",
                        size_adj_power = NA_real_, width_null = NA_real_,
                        width_signal = median(Wlam, na.rm = TRUE))
  }
  bind_rows(rows)
}

# P-P calibration: raw Z = (center - truth)/SE per coord per rep for Wald CIs.
extract_pp <- function(reps, n, field, label, TR) {
  z <- qnorm(0.975); M <- length(reps)
  Ustar <- TR$Ustar; Vstar <- TR$Vstar; lambda_star <- TR$lambda_star; r <- TR$r; rows <- list()
  collect <- function(l, side_lab, lo_field, hi_field, truth_vec) {
    Znull <- numeric(0); Zsig <- numeric(0)
    for (m in 1:M) {
      ci <- reps[[m]][[field]]; if (is.null(ci)) next
      lo <- ci[[lo_field]][, l]; hi <- ci[[hi_field]][, l]
      if (all(is.na(lo))) next
      center <- (lo + hi)/2; se <- (hi - lo)/(2 * z)
      good <- !is.na(center) & se > 0
      if (!any(good)) next
      zst <- (center[good] - truth_vec[good])/se[good]
      nm <- truth_vec[good] == 0
      Znull <- c(Znull, zst[nm]); Zsig <- c(Zsig, zst[!nm])
    }
    bind_rows(
      if (length(Znull)) tibble(estimator = label, n = n, comp = l, side = side_lab, status = "null",   Z = Znull) else NULL,
      if (length(Zsig))  tibble(estimator = label, n = n, comp = l, side = side_lab, status = "signal", Z = Zsig)  else NULL)
  }
  for (l in 1:r) {
    if (!comp_defined(reps, field, l)) next
    rows[[length(rows) + 1]] <- collect(l, "U", "U_lo", "U_hi", Ustar[, l])
    rows[[length(rows) + 1]] <- collect(l, "V", "V_lo", "V_hi", Vstar[, l])
    Zl <- numeric(0)
    for (m in 1:M) { ci <- reps[[m]][[field]]; if (is.null(ci)) next
      if (!is.null(ci$lam_lo) && !is.na(ci$lam_lo[l])) {
        cen <- (ci$lam_lo[l] + ci$lam_hi[l])/2; sel <- (ci$lam_hi[l] - ci$lam_lo[l])/(2 * z)
        if (sel > 0) Zl <- c(Zl, (cen - lambda_star[l])/sel)
      }
    }
    if (length(Zl))
      rows[[length(rows) + 1]] <- tibble(estimator = label, n = n, comp = l,
                                         side = "lam", status = "signal", Z = Zl)
  }
  bind_rows(rows)
}
# Bias / variance diagnostic for Wald-type CIs (ECCAR+split, ECCAR+CF, Laha).
# center=(lo+hi)/2, SE=(hi-lo)/(2z). se_over_sd >1: over-inflated; <1: under-covers.
extract_diag <- function(reps, n, field, label, TR) {
  z <- qnorm(0.975); M <- length(reps)
  Ustar <- TR$Ustar; Vstar <- TR$Vstar; r <- TR$r; p <- TR$p; q <- TR$q
  lambda_star <- TR$lambda_star; rows <- list()
  side_rows <- function(l, side_lab, lo_field, hi_field, Star, dim_n) {
    C <- matrix(NA_real_, M, dim_n); S <- matrix(NA_real_, M, dim_n)
    for (m in 1:M) { ci <- reps[[m]][[field]]; if (is.null(ci)) next
      lo <- ci[[lo_field]][, l]; hi <- ci[[hi_field]][, l]
      if (all(is.na(lo))) next
      C[m, ] <- (lo + hi)/2; S[m, ] <- (hi - lo)/(2 * z)
    }
    if (all(is.na(C))) return(list())
    out <- list()
    for (st in c("signal", "null")) {
      idx <- if (st == "signal") which(Star[, l] != 0) else which(Star[, l] == 0)
      if (!length(idx)) next
      cmean  <- colMeans(C[, idx, drop = FALSE], na.rm = TRUE)
      esd    <- apply(C[, idx, drop = FALSE], 2, sd, na.rm = TRUE)
      smean  <- colMeans(S[, idx, drop = FALSE], na.rm = TRUE)
      out[[length(out) + 1]] <- tibble(
        estimator = label, n = n, comp = l, side = side_lab, status = st,
        signal     = round(mean(abs(Star[idx, l])), 3),
        est_abs    = round(mean(abs(cmean)), 3),
        abs_bias   = round(mean(abs(cmean - Star[idx, l])), 3),
        emp_sd     = round(mean(esd), 4),
        mean_se    = round(mean(smean), 4),
        se_over_sd = round(mean(smean) / mean(esd), 2),
        # se_over_sd above is a RATIO OF MEANS: mean(SE_j)/mean(SD_j). When a few
        # coordinates have detonated variance (eigen-gap blowup) it is dominated by
        # those outliers and reads >>1 even while MOST coordinates have SE_j < SD_j
        # (too-narrow intervals) -- which is what actually drives Type I error.
        # The per-coordinate ratio distribution below is the honest diagnostic.
        se_sd_med  = { rr <- smean / esd; rr <- rr[is.finite(rr)]
                       if (length(rr)) round(stats::median(rr), 2) else NA_real_ },
        se_sd_q25  = { rr <- smean / esd; rr <- rr[is.finite(rr)]
                       if (length(rr)) round(unname(stats::quantile(rr, 0.25)), 2) else NA_real_ },
        se_sd_q75  = { rr <- smean / esd; rr <- rr[is.finite(rr)]
                       if (length(rr)) round(unname(stats::quantile(rr, 0.75)), 2) else NA_real_ },
        # fraction of coordinates whose reported SE is below their own empirical SD
        frac_se_lt_sd = { rr <- smean / esd; rr <- rr[is.finite(rr)]
                          if (length(rr)) round(mean(rr < 1), 3) else NA_real_ },
        # heavy-tail indicator: how far the mean ratio sits above the median one
        se_sd_tail = { rr <- smean / esd; rr <- rr[is.finite(rr)]
                       if (length(rr) && stats::median(rr) > 0)
                         round(mean(rr) / stats::median(rr), 2) else NA_real_ })
    }
    out
  }
  for (l in 1:r) {
    if (!comp_defined(reps, field, l)) next
    for (rw in side_rows(l, "U", "U_lo", "U_hi", Ustar, p)) rows[[length(rows) + 1]] <- rw
    for (rw in side_rows(l, "V", "V_lo", "V_hi", Vstar, q)) rows[[length(rows) + 1]] <- rw
    # lambda-tilde diagnostic: one scalar per rep per component
    Lc <- numeric(0); Ls <- numeric(0)
    for (m in 1:M) { ci <- reps[[m]][[field]]; if (is.null(ci)) next
      if (!is.null(ci$lam_lo) && !is.na(ci$lam_lo[l])) {
        Lc <- c(Lc, (ci$lam_lo[l] + ci$lam_hi[l])/2)
        Ls <- c(Ls, (ci$lam_hi[l] - ci$lam_lo[l])/(2 * z))
      }
    }
    if (length(Lc) >= 2) {
      esd_l <- sd(Lc); mse_l <- mean(Ls); cm_l <- mean(Lc); ls_l <- lambda_star[l]
      rows[[length(rows) + 1]] <- tibble(
        estimator = label, n = n, comp = l, side = "lam", status = "signal",
        signal     = round(ls_l, 3),
        est_abs    = round(cm_l, 3),
        abs_bias   = round(abs(cm_l - ls_l), 3),
        emp_sd     = round(esd_l, 4),
        mean_se    = round(mse_l, 4),
        se_over_sd = round(mse_l / max(esd_l, 1e-12), 2))
    }
  }
  bind_rows(rows)
}
DIAG_ESTIMATORS <- c(ci_eccar_split = "ECCAR + split (ours)",
                     ci_eccar_cf = "ECCAR + closed-form", ci_laha = "Laha + closed-form")

## ---- journal styling: muted palette + serif theme + direct end-of-line labels ----
# Desaturated, print-friendly palette. ECCAR variants warm/salient; competitors muted.
COLOR_MAP <- c("ECCAR + split (ours)" = "#b2182b", "ECCAR + closed-form" = "#ef8a62",
               "Laha + closed-form" = "#762a83", "ECCAR-raw + bootstrap" = "#2166ac",
               "Witten PMD + bootstrap" = "#8c8c8c", "Parkhomenko + bootstrap" = "#5aae61",
               "Waaijenborg + bootstrap" = "#9970ab", "Gao-Ma + bootstrap" = "#c9a227",
               "Wilms-Croux + bootstrap" = "#bababa")
# Short labels for direct end-of-line annotation (legends are removed on line panels).
SHORT_LAB <- c("ECCAR + split (ours)" = "ECCAR-split", "ECCAR + closed-form" = "ECCAR-CF",
               "Laha + closed-form" = "Laha", "ECCAR-raw + bootstrap" = "ECCAR-raw",
               "Witten PMD + bootstrap" = "Witten", "Parkhomenko + bootstrap" = "Parkhomenko",
               "Waaijenborg + bootstrap" = "Waaijenborg", "Gao-Ma + bootstrap" = "Gao-Ma",
               "Wilms-Croux + bootstrap" = "Wilms-Croux")
# Two-letter codes drawn beside the line ends; short enough to fit the gutter at any
# font metric, which full names were not.
ABBREV <- c("ECCAR + split (ours)" = "ES", "ECCAR + closed-form" = "EC",
            "ECCAR-raw + bootstrap" = "ER", "Laha + closed-form" = "LH",
            "Witten PMD + bootstrap" = "WT", "Parkhomenko + bootstrap" = "PK",
            "Waaijenborg + bootstrap" = "WB", "Gao-Ma + bootstrap" = "GM",
            "Wilms-Croux + bootstrap" = "WC")
# Legend spells each code out: "ES = ECCAR + split (ours)".
LEGEND_LAB <- stats::setNames(paste0(ABBREV[names(ABBREV)], " = ", names(ABBREV)),
                              names(ABBREV))

make_plots <- function(summary_df, subspace_df, stats_df, diag_df = NULL,
                       metrics_df = NULL, pp_df = NULL, timing_df = NULL, xlab = "n") {
  BASEFONT <- "serif"
  th <- theme_bw(base_size = 11, base_family = BASEFONT) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(linewidth = 0.25, colour = "grey90"),
          panel.border      = element_rect(linewidth = 0.4, colour = "grey55"),
          plot.title        = element_text(face = "plain", size = 11, hjust = 0),
          plot.subtitle     = element_text(size = 9,  colour = "grey30"),
          axis.title        = element_text(size = 10),
          axis.text         = element_text(size = 9, colour = "grey25"),
          axis.ticks        = element_line(linewidth = 0.3, colour = "grey55"),
          legend.position   = if (!identical(CONFIG$label_style, "direct")) "bottom" else "none",
                  legend.justification = "left",
          legend.box.just      = "left",
          legend.margin        = margin(t = 2, r = 0, b = 0, l = 0),
          legend.title      = element_blank(),
          legend.text       = element_text(size = 8.5, family = BASEFONT),
          legend.key.width  = unit(1.3, "lines"),
          legend.key.height = unit(0.8, "lines"),
          plot.margin       = margin(5, 5, 5, 5))
  # abbrev = legend ("ES = ECCAR + split (ours)") PLUS 2-letter codes at the line ends
  # legend = legend only;  direct = full short names at the line ends, no legend
  USE_LEGEND <- !identical(CONFIG$label_style, "direct")
  USE_DIRECT <-  identical(CONFIG$label_style, "abbrev") ||
                 identical(CONFIG$label_style, "direct")
  ABBREV_MODE <- identical(CONFIG$label_style, "abbrev")
  # `lv` pins legend levels so every panel yields an IDENTICAL guide; patchwork
  # collects guides only when they match exactly, else two legends are drawn.
  cl <- function(lv = NULL) {
    if (ABBREV_MODE)
      scale_color_manual(values = COLOR_MAP, labels = LEGEND_LAB, limits = lv,
                         drop = FALSE, guide = guide_legend(nrow = 3, byrow = TRUE))
    else scale_color_manual(values = COLOR_MAP, limits = lv, drop = FALSE)
  }
  fl <- function() scale_fill_manual(values = COLOR_MAP)
  LWD <- 0.55; PTS <- 1.5

  # Padding slots on the right hold the end-of-line labels (2-letter codes: 1 slot).
  XPAD_SLOTS <- if (!USE_DIRECT) 0L else if (ABBREV_MODE) 1L else 2L
  xfac <- function(nvec, pad = XPAD_SLOTS) {
    real <- as.character(sort(unique(as.integer(as.character(nvec)))))
    lv <- if (pad > 0L) c(real, paste0("\u00a0", seq_len(pad))) else real
    factor(as.character(nvec), levels = lv)
  }
  xbreaks_real <- function(nvec) as.character(sort(unique(as.integer(as.character(nvec)))))

  # End-of-line labels: 2-letter codes (abbrev) or short names (direct); no-op otherwise.
  add_end_labels <- function(g, d, yv, gap = 0.045, size = if (ABBREV_MODE) 2.9 else 2.55) {
    if (!USE_DIRECT) return(g)
    nmax <- max(as.integer(as.character(d$n)))
    lab <- d[as.integer(as.character(d$n)) == nmax, , drop = FALSE]
    lab <- lab[is.finite(lab[[yv]]), , drop = FALSE]
    if (!nrow(lab)) return(g)
    lab <- lab[order(lab[[yv]]), , drop = FALSE]
    y <- lab[[yv]]
    if (is.na(gap)) { rng <- diff(range(y)); gap <- if (rng > 0) 0.045 * rng else 0 }
    if (gap > 0) for (i in seq_along(y)[-1]) if (y[i] - y[i-1] < gap) y[i] <- y[i-1] + gap
    lab$.ly <- y
    lab$.lab <- if (ABBREV_MODE) ABBREV[as.character(lab$estimator)] else SHORT_LAB[as.character(lab$estimator)]
    nreal <- length(unique(as.integer(as.character(d$n))))
    g + geom_text(data = lab, aes(x = nreal + 0.55, y = .ly, label = .lab, color = estimator),
                  hjust = 0, size = size, family = BASEFONT, show.legend = FALSE)
  }

  # one metric vs n, directly labelled (no legend).
  line_metric <- function(d, yv, ytitle, title, hy = NA, ylim = c(0, 1.02), gap = 0.045,
                          lv = NULL) {
    d <- d %>% mutate(.xf = xfac(n))
    g <- ggplot(d, aes(.xf, .data[[yv]], color = estimator, group = estimator))
    if (!is.na(hy)) g <- g + geom_hline(yintercept = hy, linetype = "22",
                                        linewidth = 0.3, color = "grey45")
    g <- g + geom_line(linewidth = LWD) + geom_point(size = PTS) +
      cl(lv) + labs(x = "n", y = ytitle, title = title) +
      scale_x_discrete(drop = FALSE, breaks = xbreaks_real(d$n),
                       expand = expansion(add = c(0.4, 0.2))) + th
    g <- add_end_labels(g, d, yv, gap = gap)
    g + coord_cartesian(ylim = ylim)
  }

  ## ---- per-component figures, one per (side, component) ----
  make_comp_fig <- function(side_lab, side_disp) {
    figs <- list()
    for (l in seq_len(r)) {
      dS  <- summary_df %>% filter(side == side_lab, comp == l)
      dst <- stats_df  %>% filter(side == side_lab, comp == l)
      if (nrow(dS) == 0) { figs[[l]] <- NULL; next }
      # one level set for all panels -> one collected legend
      lv <- intersect(names(COLOR_MAP), unique(c(dS$estimator, dst$estimator)))
      p_t1 <- line_metric(dS %>% filter(status == "null"),   "rejection",
                          "Type I error", "Type I error", 0.05, c(0, NA), lv = lv)
      p_pw <- line_metric(dS %>% filter(status == "signal"), "rejection",
                          "Power", "Power", 1.0, lv = lv)
      # coverage: null solid, signal dashed -- direct-label only the null series
      dScov <- dS %>% mutate(.xf = xfac(n))
      p_cov <- ggplot(dScov, aes(.xf, coverage, color = estimator,
                              group = interaction(estimator, status), linetype = status)) +
        geom_hline(yintercept = 0.95, linetype = "22", linewidth = 0.3, color = "grey45") +
        geom_line(linewidth = LWD) + geom_point(size = PTS - 0.2) +
        cl(lv) + scale_linetype_manual(values = c(null = "solid", signal = "22"), guide = "none") +
        labs(x = "n", y = "Coverage", title = "Coverage") +
        scale_x_discrete(drop = FALSE, breaks = xbreaks_real(dScov$n),
                         expand = expansion(add = c(0.4, 0.2))) + th
      p_cov <- add_end_labels(p_cov, dS %>% filter(status == "null"), "coverage")
      p_cov <- p_cov + coord_cartesian(ylim = c(0, 1.02))
      p_sap <- line_metric(dst, "size_adj_power", "Size-adjusted power",
                           "Size-adjusted power", NA, ylim = c(0, 1.02), lv = lv)
      figs[[l]] <- (p_t1 | p_pw) / (p_cov | p_sap) +
        plot_layout(guides = if (USE_LEGEND) "collect" else "keep") +
        plot_annotation(
          title = sprintf("Component %d (%s), lambda* = %.2f", l, side_disp, lambda_star[l]),
          theme = theme(plot.title = element_text(family = BASEFONT, size = 12, hjust = 0)))
      if (USE_LEGEND) figs[[l]] <- figs[[l]] &
        theme(legend.position = "bottom", legend.justification = "left",
              legend.box.just = "left")
    }
    figs
  }
  comp_figs   <- make_comp_fig("U", "U")
  comp_figs_V <- make_comp_fig("V", "V")

  ## ---- split-sample correction contrast: ECCAR+CF vs ECCAR+split ----
  # Only two series here, so a compact bottom legend is clearer than dodged-bar labels.
  leg2 <- theme(legend.position = "bottom", legend.justification = "left",
                legend.box.just = "left", legend.title = element_blank(),
                legend.text = element_text(size = 9, family = BASEFONT),
                legend.key.height = unit(0.8, "lines"))
  mk_contrast <- function(l) {
    d <- summary_df %>% filter(side == "U", status == "null", comp == l,
                               estimator %in% c("ECCAR + closed-form", "ECCAR + split (ours)"))
    b1 <- ggplot(d, aes(factor(n), rejection, fill = estimator)) +
      geom_col(position = position_dodge(0.7), width = 0.6) +
      geom_hline(yintercept = 0.05, linetype = "22", linewidth = 0.3, color = "grey35") +
      fl() + labs(x = "n", y = "Type I error",
                  title = sprintf("Component %d", l)) + th + leg2
    b2 <- ggplot(d, aes(factor(n), coverage, fill = estimator)) +
      geom_col(position = position_dodge(0.7), width = 0.6) +
      geom_hline(yintercept = 0.95, linetype = "22", linewidth = 0.3, color = "grey35") +
      fl() + labs(x = "n", y = "Coverage", title = sprintf("Component %d", l)) +
      coord_cartesian(ylim = c(0, 1.02)) + th + leg2
    (b1 | b2)
  }
  contrast_fig <- wrap_plots(lapply(seq_len(r), mk_contrast), ncol = 1) +
    plot_layout(guides = "collect") +
    plot_annotation(title = "Split-sample vs closed-form variance (null coordinates, U)",
                    theme = theme(plot.title = element_text(family = BASEFONT, size = 12))) &
    theme(legend.position = "bottom", legend.justification = "left",
          legend.box.just = "left")

  ## ---- subspace distance vs n (direct-labelled) ----
  sub_named <- subspace_df %>% mutate(method = recode(method,
    eccar_raw = "ECCAR (raw)", eccar_bc = "ECCAR (debiased)", witten = "Witten",
    parkho = "Parkhomenko", waaij = "Waaijenborg", gaoma = "Gao-Ma", wilms = "Wilms-Croux"))
  sub_pal <- c("ECCAR (raw)" = "#2166ac", "ECCAR (debiased)" = "#b2182b",
               "Witten" = "#8c8c8c", "Parkhomenko" = "#5aae61", "Waaijenborg" = "#9970ab",
               "Gao-Ma" = "#c9a227", "Wilms-Croux" = "#bababa")
  sub_leg <- theme(legend.position = "bottom", legend.justification = "left",
                legend.box.just = "left", legend.title = element_blank(),
                   legend.text = element_text(size = 8.5, family = BASEFONT),
                   legend.key.width = unit(1.4, "lines"))
  f_sub_lin <- ggplot(sub_named, aes(factor(n), subspace_dist, color = method, group = method)) +
    geom_line(linewidth = LWD) + geom_point(size = PTS) +
    scale_color_manual(values = sub_pal) +
    labs(x = "n", y = expression(paste("|| sin ", Theta, " ||"[F])), title = "Linear scale") +
    th
  f_sub_log <- ggplot(sub_named, aes(n, subspace_dist, color = method, group = method)) +
    geom_line(linewidth = LWD) + geom_point(size = PTS) +
    scale_color_manual(values = sub_pal) +
    scale_x_log10() + scale_y_log10() +
    labs(x = "n (log scale)", y = expression(paste("|| sin ", Theta, " ||"[F], "  (log)")),
         title = "Log-log (slope = rate)") + th
  f_sub <- (f_sub_lin | f_sub_log) +
    plot_layout(guides = "collect") +
    plot_annotation(title = "Subspace estimation error",
                    theme = theme(plot.title = element_text(family = BASEFONT, size = 12))) &
    sub_leg

  ## ---- variance calibration: reported SE vs empirical SD (closed-form vs split) ----
  cal_fig <- NULL
  if (!is.null(diag_df) && nrow(diag_df)) {
    cal <- diag_df %>% filter(estimator %in% c("ECCAR + split (ours)", "ECCAR + closed-form"),
                              side == "U")
    lvC <- intersect(names(COLOR_MAP), unique(cal$estimator))   # one guide across comps
    mk_cal <- function(l) {
      g <- ggplot(cal %>% filter(comp == l),
                  aes(factor(n), se_over_sd, color = estimator,
                      group = interaction(estimator, status), linetype = status)) +
        geom_hline(yintercept = 1, linetype = "22", linewidth = 0.3, color = "grey45") +
        geom_line(linewidth = LWD) + geom_point(size = PTS) + cl(lvC) +
        scale_linetype_manual(values = c(null = "22", signal = "solid"), guide = "none") +
        labs(x = "n", y = "SE / SD", title = sprintf("Component %d", l)) +
        coord_cartesian(ylim = c(0, NA)) + th
      g
    }
    cal_fig <- wrap_plots(lapply(seq_len(r), mk_cal), ncol = r) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = "Variance calibration, U (reported SE / empirical SD; dashed null, solid signal)",
        theme = theme(plot.title = element_text(family = BASEFONT, size = 11))) &
      theme(legend.position = "bottom", legend.justification = "left",
                legend.box.just = "left", legend.title = element_blank(),
            legend.text = element_text(size = 9, family = BASEFONT))
  }

  ## ---- lambda-tilde inference: ONE FIGURE PER COMPONENT (coverage / power / width) ----
  lambda_figs <- list()
  dL <- summary_df %>% filter(side == "lam")
  if (nrow(dL)) {
    stL <- stats_df %>% filter(side == "lam")
    for (l in seq_len(r)) {
      dLl  <- dL  %>% filter(comp == l)
      stLl <- stL %>% filter(comp == l)
      if (!nrow(dLl)) { lambda_figs[[l]] <- NULL; next }
      dLlx  <- dLl  %>% mutate(.xf = xfac(n))
      stLlx <- stLl %>% mutate(.xf = xfac(n))
      # shared level set across the three panels -> a single collected legend
      lvL <- intersect(names(COLOR_MAP), unique(c(dLl$estimator, stLl$estimator)))
      xsc <- function() scale_x_discrete(drop = FALSE, breaks = xbreaks_real(dLl$n),
                                         expand = expansion(add = c(0.4, 0.2)))
      p_lcov <- ggplot(dLlx, aes(.xf, coverage, color = estimator, group = estimator)) +
        geom_hline(yintercept = 0.95, linetype = "22", linewidth = 0.3, color = "grey45") +
        geom_line(linewidth = LWD) + geom_point(size = PTS) + cl(lvL) +
        labs(x = "n", y = "Coverage", title = "Coverage") + xsc() + th
      p_lcov <- add_end_labels(p_lcov, dLl, "coverage")
      p_lcov <- p_lcov + coord_cartesian(ylim = c(0, 1.02))
      p_lpw <- ggplot(dLlx, aes(.xf, rejection, color = estimator, group = estimator)) +
        geom_hline(yintercept = 1.0, linetype = "22", linewidth = 0.3, color = "grey45") +
        geom_line(linewidth = LWD) + geom_point(size = PTS) + cl(lvL) +
        labs(x = "n", y = "Rejection rate", title = "Power") + xsc() + th
      p_lpw <- add_end_labels(p_lpw, dLl, "rejection")
      p_lpw <- p_lpw + coord_cartesian(ylim = c(0, 1.02))
      p_lw <- ggplot(stLlx, aes(.xf, width_signal, color = estimator, group = estimator)) +
        geom_line(linewidth = LWD) + geom_point(size = PTS) + cl(lvL) +
        labs(x = "n", y = "Median CI width", title = "Interval width") +
        scale_x_discrete(drop = FALSE, breaks = xbreaks_real(stLl$n),
                         expand = expansion(add = c(0.4, 0.2))) + th
      p_lw <- add_end_labels(p_lw, stLl, "width_signal", gap = NA)
      p_lw <- p_lw + coord_cartesian()
      lambda_figs[[l]] <- (p_lcov | p_lpw | p_lw) +
        plot_layout(guides = if (USE_LEGEND) "collect" else "keep") +
        plot_annotation(
          title = sprintf("Canonical correlation inference: component %d, lambda* = %.2f",
                          l, lambda_star[l]),
          theme = theme(plot.title = element_text(family = BASEFONT, size = 12)))
      if (USE_LEGEND) lambda_figs[[l]] <- lambda_figs[[l]] &
        theme(legend.position = "bottom", legend.justification = "left",
              legend.box.just = "left")
    }
  }

  ## ---- variance calibration: lambda-tilde SE/SD, split vs closed-form, all components ----
  lambda_cal_fig <- NULL
  if (!is.null(diag_df) && nrow(diag_df) > 0) {
    diagL <- diag_df %>% filter(side == "lam",
                                estimator %in% c("ECCAR + split (ours)", "ECCAR + closed-form"))
    if (nrow(diagL))
      lambda_cal_fig <- ggplot(diagL, aes(factor(n), se_over_sd, color = estimator, group = estimator)) +
        geom_hline(yintercept = 1, linetype = "22", linewidth = 0.3, color = "grey45") +
        geom_line(linewidth = LWD) + geom_point(size = PTS) + cl() +
        facet_wrap(~ comp, nrow = 1,
                   labeller = labeller(comp = function(x)
                     sprintf("Component %s (lambda* = %.2f)", x, lambda_star[as.integer(x)]))) +
        labs(x = "n", y = "SE / SD",
             title = "Canonical correlation variance calibration") +
        coord_cartesian(ylim = c(0, NA)) + th +
        theme(legend.position = "bottom", legend.justification = "left",
                legend.box.just = "left", legend.title = element_blank(),
              legend.text = element_text(size = 9, family = BASEFONT),
              strip.text = element_text(family = BASEFONT, size = 9))
  }

  ## ---- P-P calibration: empirical vs nominal coverage (Wald CIs, U side) ----
  pp_fig <- NULL
  if (!is.null(pp_df) && nrow(pp_df) > 0) {
    alphas <- seq(0.01, 0.99, by = 0.01); zcrit <- qnorm(1 - alphas/2)
    pp_curve <- pp_df %>% filter(side == "U") %>%
      group_by(estimator, n, comp, status) %>%
      do(tibble(nominal = 1 - alphas,
                empirical = sapply(zcrit, function(z0) mean(abs(.$Z) <= z0)))) %>%
      ungroup()
    if (nrow(pp_curve) > 0)
      pp_fig <- ggplot(pp_curve, aes(nominal, empirical, color = estimator, linetype = status)) +
        geom_abline(slope = 1, intercept = 0, color = "grey55", linewidth = 0.3) +
        geom_line(linewidth = LWD) + cl() +
        scale_linetype_manual(values = c(null = "solid", signal = "22"), guide = "none") +
        facet_grid(comp ~ n, labeller = label_both) +
        coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
        labs(x = expression(paste("Nominal  1 - ", alpha)), y = "Empirical coverage",
             title = "P-P calibration, U (solid null, dashed signal)") + th +
        theme(legend.position = "bottom", legend.justification = "left",
                legend.box.just = "left", legend.title = element_blank(),
              legend.text = element_text(size = 9, family = BASEFONT),
              strip.text = element_text(family = BASEFONT, size = 8))
  }

  ## ---- runtime (direct-labelled) ----
  timing_fig <- NULL
  if (!is.null(timing_df) && nrow(timing_df) > 0) {
    tm <- timing_df %>% filter(estimator != "ECCAR + closed-form") %>% mutate(.xf = xfac(n))
    timing_fig <- ggplot(tm, aes(.xf, mean_seconds, color = estimator, group = estimator)) +
      geom_line(linewidth = LWD) + geom_point(size = PTS) + cl() + scale_y_log10() +
      labs(x = "n", y = "Seconds / replicate (log)", title = "Runtime per replicate") +
      scale_x_discrete(drop = FALSE, breaks = xbreaks_real(tm$n),
                       expand = expansion(add = c(0.4, 0.2))) + th
    timing_fig <- add_end_labels(timing_fig, tm, "mean_seconds", gap = NA)
  }

  if (interactive()) { for (f in comp_figs) print(f); for (f in comp_figs_V) if (!is.null(f)) print(f); print(contrast_fig); print(f_sub); if (!is.null(cal_fig)) print(cal_fig); for (f in lambda_figs) if (!is.null(f)) print(f); if (!is.null(lambda_cal_fig)) print(lambda_cal_fig); if (!is.null(pp_fig)) print(pp_fig); if (!is.null(timing_fig)) print(timing_fig) }

  # When the x-variable is not n (e.g. dimension sweep: xlab = "p + q"), rewrite the
  # x-axis title on every figure. `& labs()` applies across all panels of a patchwork
  # composite; `+ labs()` on a plain ggplot. Filenames get a prefix so a dim sweep in
  # the same folder never overwrites an n-sweep's PNGs.
  relab <- function(g) { if (is.null(g)) return(NULL)
    if (inherits(g, "patchwork")) g & labs(x = xlab) else g + labs(x = xlab) }
  PFX <- if (identical(xlab, "n")) "scca_rankr_" else "scca_dim_"
  if (!identical(xlab, "n")) {
    comp_figs   <- lapply(comp_figs, relab)
    comp_figs_V <- lapply(comp_figs_V, relab)
    contrast_fig <- relab(contrast_fig); f_sub <- relab(f_sub); cal_fig <- relab(cal_fig)
    lambda_figs <- lapply(lambda_figs, relab); lambda_cal_fig <- relab(lambda_cal_fig)
    pp_fig <- relab(pp_fig); timing_fig <- relab(timing_fig)
  }

  ## ---- write one readable PNG per component (U and V), plus the rest -----
  saved <- character(0)
  for (l in seq_len(r)) if (!is.null(comp_figs[[l]])) {
    f <- file.path(CONFIG$outdir, sprintf("%scomponent%d.png", PFX, l))
    ggsave(f, comp_figs[[l]], width = 11.5, height = 8.0, dpi = 200); saved <- c(saved, basename(f))
  }
  for (l in seq_len(r)) if (!is.null(comp_figs_V[[l]])) {
    f <- file.path(CONFIG$outdir, sprintf("%scomponent%d_V.png", PFX, l))
    ggsave(f, comp_figs_V[[l]], width = 11.5, height = 8.0, dpi = 200); saved <- c(saved, basename(f))
  }
  ggsave(file.path(CONFIG$outdir, paste0(PFX, "split_contrast.png")), contrast_fig,
         width = 8, height = 3.0 * r, dpi = 200); saved <- c(saved, paste0(PFX, "split_contrast.png"))
  ggsave(file.path(CONFIG$outdir, paste0(PFX, "subspace.png")), f_sub,
         width = 11, height = 4.6, dpi = 200); saved <- c(saved, paste0(PFX, "subspace.png"))
  if (!is.null(cal_fig)) { ggsave(file.path(CONFIG$outdir, paste0(PFX, "variance_calibration.png")), cal_fig,
         width = 4.8 * r, height = 4.6, dpi = 200); saved <- c(saved, paste0(PFX, "variance_calibration.png")) }
  for (l in seq_len(r)) if (!is.null(lambda_figs[[l]])) {
    f <- file.path(CONFIG$outdir, sprintf("%slambda_comp%d.png", PFX, l))
    ggsave(f, lambda_figs[[l]], width = 13.5, height = 5.2, dpi = 200); saved <- c(saved, basename(f))
  }
  if (!is.null(lambda_cal_fig)) { ggsave(file.path(CONFIG$outdir, paste0(PFX, "lambda_calibration.png")), lambda_cal_fig,
         width = 4.2 * r, height = 4.2, dpi = 200); saved <- c(saved, paste0(PFX, "lambda_calibration.png")) }
  if (!is.null(pp_fig)) { ggsave(file.path(CONFIG$outdir, paste0(PFX, "pp_calibration.png")), pp_fig,
         width = 4.2 * length(unique(pp_df$n)), height = 4.2 * r, dpi = 200, limitsize = FALSE); saved <- c(saved, paste0(PFX, "pp_calibration.png")) }
  if (!is.null(timing_fig)) { ggsave(file.path(CONFIG$outdir, paste0(PFX, "timing.png")), timing_fig,
         width = 7, height = 4.6, dpi = 200); saved <- c(saved, paste0(PFX, "timing.png")) }
  cat(sprintf("Saved figures: %s\n", paste(saved, collapse = ", ")))

  invisible(list(components = comp_figs, components_V = comp_figs_V,
                 split_contrast = contrast_fig, subspace = f_sub,
                 calibration = cal_fig, lambda = lambda_figs,
                 lambda_calibration = lambda_cal_fig,
                 pp = pp_fig, timing = timing_fig))
}

aggregate_all <- function(mem = NULL) {
  # Assemble per-n datasets, each = list(n, reps, TR). Two sources:
  #  * mem != NULL : in-memory results from the current run. The ground truth is
  #    taken straight from memory (it generated the data), so no reconstruction
  #    and no dimension mismatch is possible. This is the normal path.
  #  * mem == NULL : read stored checkpoints (--aggregate). New files are self-
  #    describing; legacy files fall back to session globals with a clear guard.
  datasets <- list()
  if (!is.null(mem)) {
    for (nm in names(mem$reps_by_n)) {
      reps <- mem$reps_by_n[[nm]]; if (is.null(reps) || !length(reps)) next
      datasets[[length(datasets) + 1]] <- list(n = as.integer(nm), reps = reps, TR = mem$TR)
    }
  } else {
    for (n in CONFIG$n_grid) {
      fn <- file.path(CONFIG$outdir, sprintf("scca_rankr_n%d.rds", n)); if (!file.exists(fn)) { cat("missing:", fn, "\n"); next }
      obj <- readRDS(fn); reps <- obj$reps
      if (!is.null(obj$Ustar)) {
        TR <- list(Ustar = obj$Ustar, Vstar = obj$Vstar, lambda_star = obj$lambda_star, p = obj$p, q = obj$q, r = obj$r)
      } else {
        TR <- list(Ustar = Ustar, Vstar = Vstar, lambda_star = lambda_star, p = p, q = q, r = r)
        ci1 <- reps[[1]]$ci_eccar_split
        if (!is.null(ci1) && nrow(ci1$U_lo) != p) stop(sprintf(
          "Checkpoint %s has loadings of dimension p=%d but the current session has p=%d.\n  This file predates self-describing checkpoints. Re-run aggregation with the matching\n  dimensions, e.g.  Rscript scca_benchmark_rankr.R --p=%d --q=%d --aggregate",
          fn, nrow(ci1$U_lo), p, nrow(ci1$U_lo), nrow(ci1$V_lo) ))
      }
      datasets[[length(datasets) + 1]] <- list(n = n, reps = reps, TR = TR)
    }
  }
  ml <- list(); subl <- list(); stl <- list(); dgl <- list(); ppl <- list(); tml <- list(); TR <- NULL
  for (ds in datasets) {
    reps <- ds$reps; TR <- ds$TR; n <- ds$n
    # Gram-Schmidt leaves ~1e-38 fill-in in later components; `== 0` would count it
    # as undetectable SIGNAL and cap power. Real loadings are 1/sqrt(s) ~ 0.58.
    TR$Ustar[abs(TR$Ustar) < 1e-10] <- 0
    TR$Vstar[abs(TR$Vstar) < 1e-10] <- 0
    # A failed mclapply worker leaves a try-error object (not NULL) in its slot.
    n_before <- length(reps)
    reps <- Filter(function(rp) is.list(rp) && !inherits(rp, "try-error"), reps)
    if (length(reps) < n_before)
      cat(sprintf("  n=%d: dropped %d failed rep(s); aggregating over %d.\n",
                  n, n_before - length(reps), length(reps)))
    if (!length(reps)) next
    for (field in names(ESTIMATORS)) {
      ml[[length(ml) + 1]] <- extract_est(reps, n, field, ESTIMATORS[[field]], TR)
      stl[[length(stl) + 1]] <- extract_stats(reps, n, field, ESTIMATORS[[field]], TR)
    }
    for (field in names(DIAG_ESTIMATORS)) {
      dgl[[length(dgl) + 1]] <- extract_diag(reps, n, field, DIAG_ESTIMATORS[[field]], TR)
      ppl[[length(ppl) + 1]] <- extract_pp(reps, n, field, DIAG_ESTIMATORS[[field]], TR)
    }
    sm <- sapply(reps, function(rp) rp$subU); subl[[length(subl) + 1]] <- tibble(n = n, method = rownames(sm), subspace_dist = rowMeans(sm, na.rm = TRUE))
    # Per-method timing: aggregate elapsed-seconds across reps. Stored in each
    # rep as rp$timing (named numeric, one entry per estimator-as-CI-unit).
    tm_mat <- do.call(rbind, lapply(reps, function(rp) rp$timing))
    if (!is.null(tm_mat) && ncol(tm_mat))
      tml[[length(tml) + 1]] <- tibble(estimator = colnames(tm_mat), n = n,
                                       mean_seconds = colMeans(tm_mat, na.rm = TRUE),
                                       median_seconds = apply(tm_mat, 2, median, na.rm = TRUE))
  }
  metrics_df <- bind_rows(ml)
  if (nrow(metrics_df) == 0) { cat("Nothing to aggregate.\n"); return(invisible()) }
  summary_df <- metrics_df %>% group_by(estimator, n, comp, side, status) %>%
    summarise(coverage = round(mean(coverage), 3), rejection = round(mean(rejection), 3), .groups = "drop")
  subspace_df <- bind_rows(subl); stats_df <- bind_rows(stl); diag_df <- bind_rows(dgl)
  pp_df <- bind_rows(ppl); timing_df <- bind_rows(tml)
  # use the truth actually loaded (r, lambda) for labelling, not the session globals
  r_agg <- max(summary_df$comp); lam_agg <- TR$lambda_star

  for (cc in 1:r_agg) {
    cat(sprintf("\n===== COMPONENT %d (lambda*=%.2f): coverage =====\n", cc, lam_agg[cc]))
    print(summary_df %>% filter(comp == cc) %>% select(-rejection, -comp) %>%
            pivot_wider(names_from = c(side, status), values_from = coverage), n = 40)
    cat(sprintf("\n===== COMPONENT %d: rejection (null=TypeI, signal=Power) =====\n", cc))
    print(summary_df %>% filter(comp == cc) %>% select(-coverage, -comp) %>%
            pivot_wider(names_from = c(side, status), values_from = rejection), n = 40)
  }
  cat("\n===== SIZE-ADJUSTED POWER (U, threshold calibrated to 5% size) =====\n")
  print(stats_df %>% select(estimator, n, comp, size_adj_power) %>%
          pivot_wider(names_from = c(n, comp), values_from = size_adj_power), n = 20)
  cat("\n===== MEDIAN CI WIDTH at null coords (U) =====\n")
  print(stats_df %>% select(estimator, n, comp, width_null) %>%
          pivot_wider(names_from = c(n, comp), values_from = width_null), n = 20)
  cat("\n===== SUBSPACE distance: sin-Theta (mean over reps; ECCAR raw vs debiased) =====\n")
  print(subspace_df %>% pivot_wider(names_from = n, values_from = subspace_dist), n = 20)

  cat("\n===== POINT-ESTIMATE BIAS / VARIANCE (U side; Wald CIs) =====\n")
  cat("signal=mean|U*|; est_abs=mean|estimate|; abs_bias=|mean(est)-truth|;\n",
      "emp_sd=sampling SD of estimate; mean_se=SE the method uses;\n",
      "se_over_sd >1 => CI over-inflated (low power, over-coverage), <1 => under-coverage.\n", sep = "")
  print(diag_df %>% arrange(estimator, comp, desc(status), n), n = 60)

  out <- list(metrics_df = metrics_df, summary_df = summary_df,
              subspace_df = subspace_df, stats_df = stats_df, diag_df = diag_df,
              pp_df = pp_df, timing_df = timing_df)
  # ---- write results as a FOLDER of CSV tables (the default deliverable) ----
  od <- CONFIG$outdir
  utils::write.csv(summary_df,  file.path(od, "summary_metrics.csv"),               row.names = FALSE)
  utils::write.csv(metrics_df,  file.path(od, "metrics_by_coordinate.csv"),         row.names = FALSE)
  utils::write.csv(stats_df,    file.path(od, "size_adjusted_power_and_width.csv"), row.names = FALSE)
  utils::write.csv(subspace_df, file.path(od, "subspace_distance.csv"),                row.names = FALSE)
  utils::write.csv(diag_df,     file.path(od, "point_estimate_diagnostics.csv"),    row.names = FALSE)
  if (nrow(pp_df))     utils::write.csv(pp_df,     file.path(od, "pp_calibration_data.csv"), row.names = FALSE)
  if (nrow(timing_df)) utils::write.csv(timing_df, file.path(od, "timing.csv"),              row.names = FALSE)
  cfg_scalars <- c("p","q","r","lambda_star","s_u","s_v","ar_rho","n_grid","M_grid",
                   "B","K","alpha","admm_rho","admm_iter","admm_tol","lambda_mult",
                   "node_mult","scio_iter","use_ccar3","n_cores","inner_cores","node_cores","blas_threads","seed_base",
                   "oracle_lam","precision","label_style")
  run_config <- data.frame(
    setting = cfg_scalars,
    value   = vapply(cfg_scalars, function(k) paste(CONFIG[[k]], collapse = ","), character(1)),
    stringsAsFactors = FALSE)
  utils::write.csv(run_config, file.path(od, "run_config.csv"), row.names = FALSE)
  cat(sprintf("\nWrote CSV results to folder: %s\n", normalizePath(od, mustWork = FALSE)))
  out$plots <- make_plots(summary_df, subspace_df, stats_df, diag_df,
                          metrics_df = metrics_df, pp_df = pp_df, timing_df = timing_df)
  invisible(out)
}

## ---- DIMENSION sweep: fixed n, x-axis = p+q, combine several point dirs ----
# Reuses every extractor and make_plots. Each point's checkpoint carries its own
# ground truth (p,q differ across points), so per-point aggregation is correct.
# The extractors key rows on a scalar they call "n"; we feed p+q there, then
# make_plots(xlab = "p + q") retitles the axis. Reads:
#   --dims=<p+q,...>  --dirs=<dir,...>  (same order)   [--sweep_n=<fixed n>]
# If --dirs omitted, auto-discovers via --glob (default results_dim_*), sorted.
aggregate_dim <- function(fixed_n = NULL, dims = NULL, dirs = NULL) {
  # Args come from the driver, or from --sweep_n / --dims / --dirs when NULL.
  if (is.null(fixed_n)) fixed_n <- as.integer(flag_raw("sweep_n") %||% "500")
  if (is.null(dims)) dims <- as.integer(strsplit(flag_raw("dims") %||% "", ",")[[1]])
  if (is.null(dirs)) dirs <- strsplit(flag_raw("dirs") %||% "", ",")[[1]]
  if (!length(dirs)) {
    dirs <- sort(Sys.glob(flag_raw("glob") %||% "results_dim_*"))
    cat("Auto-discovered dirs:\n  ", paste(dirs, collapse = "\n  "), "\n")
  }
  if (!length(dirs) || length(dims) != length(dirs))
    stop("dimension sweep needs --dims and --dirs of equal length (or --dims with --glob).")
  dir.create(CONFIG$outdir, showWarnings = FALSE, recursive = TRUE)

  datasets <- list()
  for (i in seq_along(dirs)) {
    d <- dirs[i]; pq <- dims[i]
    fn <- Sys.glob(file.path(d, sprintf("scca_rankr_n%d.rds", fixed_n)))
    if (!length(fn)) fn <- Sys.glob(file.path(d, "scca_rankr_n*.rds"))
    if (!length(fn)) { cat("SKIP (no checkpoint):", d, "\n"); next }
    obj <- readRDS(fn[1]); reps <- obj$reps
    nb <- length(reps)
    reps <- Filter(function(rp) is.list(rp) && !inherits(rp, "try-error"), reps)
    if (length(reps) < nb) cat(sprintf("  %s: dropped %d failed rep(s)\n", d, nb - length(reps)))
    if (!length(reps)) { cat("SKIP (no valid reps):", d, "\n"); next }
    TR <- list(Ustar = obj$Ustar, Vstar = obj$Vstar, lambda_star = obj$lambda_star,
               p = obj$p, q = obj$q, r = obj$r)
    TR$Ustar[abs(TR$Ustar) < 1e-10] <- 0        # drop Gram-Schmidt phantom signal
    TR$Vstar[abs(TR$Vstar) < 1e-10] <- 0
    cat(sprintf("  loaded %s: n=%d p=%d q=%d -> x(p+q)=%d, %d reps\n",
                d, obj$n, obj$p, obj$q, pq, length(reps)))
    datasets[[length(datasets) + 1]] <- list(n = pq, reps = reps, TR = TR)   # 'n' := p+q
  }
  if (!length(datasets)) { cat("Nothing to aggregate.\n"); return(invisible()) }

  ml <- list(); subl <- list(); stl <- list(); dgl <- list(); ppl <- list(); tml <- list(); TR <- NULL
  for (ds in datasets) {
    reps <- ds$reps; TR <- ds$TR; n <- ds$n
    for (field in names(ESTIMATORS)) {
      ml[[length(ml)+1]]  <- extract_est(reps, n, field, ESTIMATORS[[field]], TR)
      stl[[length(stl)+1]] <- extract_stats(reps, n, field, ESTIMATORS[[field]], TR)
    }
    for (field in names(DIAG_ESTIMATORS)) {
      dgl[[length(dgl)+1]] <- extract_diag(reps, n, field, DIAG_ESTIMATORS[[field]], TR)
      ppl[[length(ppl)+1]] <- extract_pp(reps, n, field, DIAG_ESTIMATORS[[field]], TR)
    }
    sm <- sapply(reps, function(rp) rp$subU)
    subl[[length(subl)+1]] <- tibble(n = n, method = rownames(sm),
                                     subspace_dist = rowMeans(sm, na.rm = TRUE))
    tm_mat <- do.call(rbind, lapply(reps, function(rp) rp$timing))
    if (!is.null(tm_mat) && ncol(tm_mat))
      tml[[length(tml)+1]] <- tibble(estimator = colnames(tm_mat), n = n,
                                     mean_seconds = colMeans(tm_mat, na.rm = TRUE),
                                     median_seconds = apply(tm_mat, 2, median, na.rm = TRUE))
  }
  metrics_df <- bind_rows(ml)
  summary_df <- metrics_df %>% group_by(estimator, n, comp, side, status) %>%
    summarise(coverage = round(mean(coverage),3), rejection = round(mean(rejection),3), .groups="drop")
  subspace_df <- bind_rows(subl); stats_df <- bind_rows(stl); diag_df <- bind_rows(dgl)
  pp_df <- bind_rows(ppl); timing_df <- bind_rows(tml)

  od <- CONFIG$outdir
  ren <- function(df) if ("n" %in% names(df)) dplyr::rename(df, p_plus_q = n) else df
  utils::write.csv(ren(summary_df),  file.path(od, "dim_summary_metrics.csv"),  row.names = FALSE)
  utils::write.csv(ren(stats_df),    file.path(od, "dim_size_power_width.csv"), row.names = FALSE)
  utils::write.csv(ren(subspace_df), file.path(od, "dim_subspace_distance.csv"),row.names = FALSE)
  utils::write.csv(ren(diag_df),     file.path(od, "dim_point_diagnostics.csv"),row.names = FALSE)
  cat(sprintf("\nWrote dimension-sweep CSVs to %s\n", normalizePath(od, mustWork = FALSE)))

  make_plots(summary_df, subspace_df, stats_df, diag_df,
             metrics_df = metrics_df, pp_df = pp_df, timing_df = timing_df, xlab = "p + q")
  cat(sprintf("Dimension sweep complete (fixed n=%d, x = p + q). Figures in %s\n",
              fixed_n, normalizePath(od)))
  invisible(NULL)
}

## ------------------------------- main --------------------------------------
# Normal run: generate + run + aggregate in ONE pass, using the in-memory ground
# truth (no file round-trip, no reconstruction, so no dimension mismatch is
# possible). `--aggregate` re-reads stored checkpoints instead (self-describing).
## ---- driver: both experiments in one pass (sourced, or no CLI flags) ----
regime_table <- function() {
  cat("\n================== REGIME CHECK  s^2 log^2(p+q) / n ==================\n")
  cat("   (< 1 = inside the theoretical regime; > 1 = outside)\n")
  s2 <- max(RUN$s_u, RUN$s_v)^2
  if (isTRUE(RUN$do_n_sweep)) {
    cat(sprintf("\n  A. SAMPLE-SIZE sweep   p = %d, q = %d, s = %d\n",
                RUN$n_p, RUN$n_q, max(RUN$s_u, RUN$s_v)))
    for (n in RUN$n_grid) {
      ratio <- s2 * log(RUN$n_p + RUN$n_q)^2 / n
      cat(sprintf("     n = %5d   ratio = %5.2f  %-8s  p,q > n: %s\n", n, ratio,
                  if (ratio < 1) "IN" else "OUT",
                  if (RUN$n_p > n && RUN$n_q > n) "yes" else "NO"))
    }
  }
  if (isTRUE(RUN$do_dim_sweep)) {
    cat(sprintf("\n  B. DIMENSION sweep     n = %d, s = %d\n",
                RUN$dim_n, max(RUN$s_u, RUN$s_v)))
    for (i in seq_along(RUN$dim_p)) {
      P <- RUN$dim_p[i]; Q <- RUN$dim_q[i]
      ratio <- s2 * log(P + Q)^2 / RUN$dim_n
      cat(sprintf("     p = %5d q = %5d (p+q = %5d)  ratio = %5.2f  %-8s\n",
                  P, Q, P + Q, ratio, if (ratio < 1) "IN" else "OUT"))
    }
  }
  cat("=====================================================================\n\n")
}

run_experiments <- function() {
  stopifnot(length(RUN$dim_p) == length(RUN$dim_q))
  regime_table()
  # Required: the dimension sweep aggregates by re-reading each point's .rds.
  CONFIG$checkpoint <- TRUE; assign("CONFIG", CONFIG, envir = .GlobalEnv)

  ## ---- A. sample-size sweep: vary n at fixed (p, q) ----
  if (isTRUE(RUN$do_n_sweep)) {
    cat("\n########## A. SAMPLE-SIZE SWEEP ##########\n")
    build_dgp(RUN$n_p, RUN$n_q)
    CONFIG$outdir <- RUN$n_outdir; assign("CONFIG", CONFIG, envir = .GlobalEnv)
    dir.create(CONFIG$outdir, showWarnings = FALSE, recursive = TRUE)
    reps_by_n <- list()
    for (n in RUN$n_grid)
      reps_by_n[[as.character(n)]] <- run_one_n(as.integer(n), as.integer(RUN$mc))
    aggregate_all(mem = list(reps_by_n = reps_by_n,
                             TR = list(Ustar = Ustar, Vstar = Vstar,
                                       lambda_star = lambda_star, p = p, q = q, r = r)))
  }

  ## ---- B. dimension sweep: vary (p, q) at fixed n ----
  if (isTRUE(RUN$do_dim_sweep)) {
    cat("\n########## B. DIMENSION SWEEP ##########\n")
    dirs <- character(0); dims <- integer(0)
    for (i in seq_along(RUN$dim_p)) {
      P <- RUN$dim_p[i]; Q <- RUN$dim_q[i]
      cat(sprintf("\n---- dimension point %d/%d: p = %d, q = %d (p+q = %d) ----\n",
                  i, length(RUN$dim_p), P, Q, P + Q))
      build_dgp(P, Q)                       # fresh truth at this (p, q)
      od <- paste0(RUN$dim_point_dir, P)
      CONFIG$outdir <- od; assign("CONFIG", CONFIG, envir = .GlobalEnv)
      dir.create(od, showWarnings = FALSE, recursive = TRUE)
      run_one_n(as.integer(RUN$dim_n), as.integer(RUN$mc))
      dirs <- c(dirs, od); dims <- c(dims, P + Q)
    }
    CONFIG$outdir <- RUN$dim_outdir; assign("CONFIG", CONFIG, envir = .GlobalEnv)
    dir.create(CONFIG$outdir, showWarnings = FALSE, recursive = TRUE)
    aggregate_dim(fixed_n = as.integer(RUN$dim_n), dims = dims, dirs = dirs)
  }

  cat("\n########## DONE ##########\n")
  if (isTRUE(RUN$do_n_sweep))
    cat(sprintf("  sample-size figures: %s/scca_rankr_*.png\n", RUN$n_outdir))
  if (isTRUE(RUN$do_dim_sweep))
    cat(sprintf("  dimension  figures: %s/scca_dim_*.png\n", RUN$dim_outdir))
  invisible(NULL)
}

if (SWEEP_DIM) {
  aggregate_dim()
} else if (length(args) == 0L || any(args == "--all")) {
  run_experiments()          # RStudio / no-flag path: BOTH experiments
} else if (AGG_ONLY) {
  aggregate_all()
} else {
  TR <- list(Ustar = Ustar, Vstar = Vstar, lambda_star = lambda_star, p = p, q = q, r = r)
  reps_by_n <- list()
  for (i in seq_along(CONFIG$n_grid)) {
    n <- CONFIG$n_grid[i]
    reps_by_n[[as.character(n)]] <- run_one_n(n, M_grid[i])
  }
  aggregate_all(mem = list(reps_by_n = reps_by_n, TR = TR))
}
