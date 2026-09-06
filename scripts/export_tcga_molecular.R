# ==============================================================================
# scripts/export_tcga_molecular.R
#
# Read-only export from the existing thesis R pipeline (~/Master's
# Work/Thesis): sources its core/*.R files by absolute path and reuses their
# functions to write one flat CSV here:
#
#   data/tcga/molecular_scores.csv
#   columns: sample_id, <signature>_ssGSEA (one per gene set in Data/ML.csv),
#            age, sex, stage, os_time, os_event
#
# Cohort is TCGA-COAD only, using clinical covariates from the TCGA Clinical
# Data Resource (Liu et al. 2018) and ssGSEA scores over the full signature
# panel in Data/ML.csv -- both read-only from the thesis repo, never edited.
# The TPM matrix and colData must already be cached on disk from a previous
# thesis run; this script refuses to trigger a fresh GDCdownload() (TCGA-COAD
# RNA-seq is several GB) and stops with a clear error if the cache is missing.
#
# Run from the modality-value repo root:
#   Rscript scripts/export_tcga_molecular.R
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

THESIS_ROOT <- "/Users/elabd/Master's Work/Thesis"
MODALITY_VALUE_ROOT <- getwd()
OUT_CSV <- file.path(MODALITY_VALUE_ROOT, "data", "tcga", "molecular_scores.csv")

stopifnot(dir.exists(THESIS_ROOT))
stopifnot(file.exists(file.path(MODALITY_VALUE_ROOT, "pyproject.toml")))

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

message("== Exporting TCGA-COAD molecular scores ==")

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
