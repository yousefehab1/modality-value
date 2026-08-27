# ==============================================================================
# scripts/export_tcga_molecular.R
#
# Phase 4 (molecular arm) read-only export from the existing thesis R
# pipeline (~/Master's Work/Thesis). This script does NOT modify or commit
# anything in the thesis repo -- it only source()s its core/*.R files by
# absolute path and reuses their functions to write one flat CSV here:
#
#   data/tcga/molecular_scores.csv
#   columns: sample_id, <signature>_ssGSEA (one per gene set in Data/ML.csv),
#            age, sex, stage, os_time, os_event
#
# Design decisions, and why:
#
# - Cohort: TCGA-COAD only (the "TCGA half" of modules/crc_survival.R),
#   not GSE39582. GSE39582 is a microarray cohort with no obvious analogue
#   to a "second held-out cohort" for this exercise, and only one cohort is
#   needed to prove fusion/value.py is metric-agnostic across a second
#   modality-agnostic ModalityModel. Pulling in GSE39582 would also require
#   downloading it from GEO, which this script avoids (see below).
#
# - No fresh downloads: TCGA-COAD's STAR TPM matrix and colData are already
#   cached on disk from a previous thesis run
#   (Thesis/cache/TCGA_COAD_tpm_ensembl.rds, TCGA_COAD_coldata.rds). This
#   script reads them via the thesis's own cache_rds() helper, which returns
#   the cached object and *never* re-executes the GDCquery/GDCdownload
#   fallback closure when the cache file is already present. If those cache
#   files are ever missing, this script will refuse to fetch fresh data and
#   will stop with a clear error instead (see the guard below) --
#   TCGA-COAD RNA-seq is several GB and this machine has very limited free
#   disk space right now.
#
# - Clinical covariates (age, sex, stage, os_time, os_event) come from the
#   TCGA Clinical Data Resource (Liu et al. 2018), already present locally
#   at Thesis/Data/TCGA-CDR.csv, via the thesis's own load_tcga_cdr(). This
#   script does NOT call TCGAbiolinks::GDCquery_clinic() (used in
#   modules/crc_survival.R only for CRC-specific treatment status, which is
#   not part of the required CSV schema here) -- avoiding that network call
#   entirely.
#
# - Endpoint: overall survival (OS), not the module's PFI-based recurrence
#   endpoint -- os_time/os_event is the schema io/tcga.py expects, and OS is
#   the least-processed, most standard endpoint (event = death, no landmark
#   truncation). Time is reported in MONTHS (OS.time days / DAYS_PER_MONTH),
#   matching the thesis's own convention in core/clinical.R. This is a
#   monotonic rescaling of days and does not change concordance (C-index).
#
# - Signature scores: ssGSEA over the whole signature panel in Data/ML.csv
#   (mirroring main.R's default CRC_SIGNATURES = NULL, i.e. every column),
#   via the thesis's own run_ssgsea() + add_composite(), so the score-naming
#   convention ("<Signature>_ssGSEA") matches the rest of the thesis exactly.
#   The signature file itself (Data/ML.csv) is read only, never edited.
#
# Run from anywhere:
#   Rscript scripts/export_tcga_molecular.R
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

THESIS_ROOT <- "/Users/elabd/Master's Work/Thesis"
# Absolute path to this agent's modality-value worktree (this script is run
# via `Rscript scripts/export_tcga_molecular.R` from there, or via `make`).
MODALITY_VALUE_ROOT <- "/Users/elabd/Master's Work/modality-value/.claude/worktrees/agent-a0362a5d109a6575b"
OUT_CSV <- file.path(MODALITY_VALUE_ROOT, "data", "tcga", "molecular_scores.csv")

stopifnot(dir.exists(THESIS_ROOT))

# We need cache_rds()'s relative "cache/..." lookups and CDR_FILE = "Data/..."
# to resolve inside the thesis repo, so operate with it as the working
# directory for the duration of this script only.
old_wd <- getwd()
setwd(THESIS_ROOT)
on.exit(setwd(old_wd), add = TRUE)

# ---- Source thesis core (read-only; order matches thesis main.R) -----------
source(file.path(THESIS_ROOT, "core", "config.R"))
source(file.path(THESIS_ROOT, "core", "io.R"))
source(file.path(THESIS_ROOT, "core", "id_conversion.R"))
source(file.path(THESIS_ROOT, "core", "signatures.R"))
source(file.path(THESIS_ROOT, "core", "expression.R"))
source(file.path(THESIS_ROOT, "core", "scoring.R"))
source(file.path(THESIS_ROOT, "core", "clinical.R"))

message("== Phase 4 export: TCGA-COAD molecular scores ==")

# ---- 1. Load the already-cached TCGA-COAD TPM matrix + colData -------------
tpm_cache_key <- paste0("TCGA_COAD_tpm_", ID_TYPE)
tpm_cache_file <- file.path(CACHE_DIR, paste0(tpm_cache_key, ".rds"))
coldata_cache_file <- file.path(CACHE_DIR, "TCGA_COAD_coldata.rds")

if (!file.exists(tpm_cache_file) || !file.exists(coldata_cache_file)) {
  stop(
    "Required TCGA-COAD cache files are missing:\n",
    "  ", tpm_cache_file, " (exists: ", file.exists(tpm_cache_file), ")\n",
    "  ", coldata_cache_file, " (exists: ", file.exists(coldata_cache_file), ")\n",
    "Refusing to trigger a fresh GDCdownload() of TCGA-COAD RNA-seq -- it is ",
    "several GB and this machine has very limited free disk space right now. ",
    "Stop and report this rather than downloading."
  )
}

mat_tcga <- cache_rds(tpm_cache_key, function() {
  stop("Unreachable: tpm cache file exists but cache_rds() tried to rebuild it.")
})
coad_coldata <- cache_rds("TCGA_COAD_coldata", function() {
  stop("Unreachable: coldata cache file exists but cache_rds() tried to rebuild it.")
})

message(sprintf("Loaded cached TCGA-COAD TPM matrix: %d genes x %d samples.",
                nrow(mat_tcga), ncol(mat_tcga)))

# ---- 2. Signature panel -> ssGSEA scores -----------------------------------
panel <- unify_panel_ids(load_signature_panel(SIG_FILE))   # read-only; never edited
gs    <- build_gene_sets(panel)                            # whole panel (main.R default)

message("Running ssGSEA over the full signature panel (this may take a minute)...")
scores_tcga <- run_ssgsea(mat_tcga, gs) %>%
  add_composite() %>%
  tibble::rownames_to_column("Full_Barcode") %>%
  dplyr::arrange(Full_Barcode) %>%                          # deterministic vial selection
  dplyr::mutate(sample_id = tcga_patient_id(Full_Barcode)) %>%
  dplyr::distinct(sample_id, .keep_all = TRUE) %>%
  dplyr::select(-Full_Barcode)

score_cols <- setdiff(colnames(scores_tcga), "sample_id")
message(sprintf("Scored %d samples across %d signature columns: %s",
                nrow(scores_tcga), length(score_cols), paste(score_cols, collapse = ", ")))

# ---- 3. Clinical covariates from the TCGA-CDR (age, sex, stage, OS) -------
cdr <- load_tcga_cdr() %>%
  dplyr::filter(Project_ID == "TCGA-COAD") %>%
  dplyr::transmute(
    sample_id = Patient_ID,
    age       = as.numeric(age_at_initial_pathologic_diagnosis),
    sex       = tolower(trimws(gender)),
    stage     = Stage,
    os_time   = OS_months,     # months; see header comment
    os_event  = as.integer(OS_event)
  )

# ---- 4. Join scores + clinical, write CSV ----------------------------------
molecular_df <- scores_tcga %>%
  dplyr::inner_join(cdr, by = "sample_id") %>%
  dplyr::filter(!is.na(os_time), !is.na(os_event)) %>%
  dplyr::select(sample_id, dplyr::all_of(score_cols), age, sex, stage, os_time, os_event)

message(sprintf("Final export: %d patients with scores + clinical + OS endpoint.",
                nrow(molecular_df)))
stopifnot(nrow(molecular_df) > 0)

dir.create(dirname(OUT_CSV), recursive = TRUE, showWarnings = FALSE)
write.csv(molecular_df, OUT_CSV, row.names = FALSE)
message("Wrote: ", OUT_CSV)
