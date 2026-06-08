1 - run find_proteins_metaldb.py specify the metal type
2 - extract_binding_labels_sequence_metaldb_peptides.py
3 - sort_training_data_CU.py
4 - BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_metaltype_updated.py



The surrogate models are fitted independently for each objective, but the acquisition function is calculated jointly across all four objectives.

  ### How It Works

  1. Independent GP per objective

  In BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_CU_updated.py:924, the code fits four separate SingleTaskGP models:

  for m in range(M):
      y_m = Y[:, m:m+1]
      gp_m = SingleTaskGP(Z, y_m, ...)

  These independent models are combined into one ModelListGP:

  return ModelListGP(*models)

  Thus, the code assumes that the objectives are conditionally independent given the latent peptide representation.

  2. One joint multi-objective acquisition function

  The code constructs a single qExpectedHypervolumeImprovement acquisition function:

  acq = qExpectedHypervolumeImprovement(
      model=mo_model,
      ref_point=ref_point,
      partitioning=partitioning,
      sampler=sampler,
  )

  See BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_CU_updated.py:1250.

  qEHVI combines predictions from all four independent GPs and scores each latent candidate according to its expected improvement in the four-dimensional Pareto-front hypervolume.
  Therefore, there is not a separate acquisition optimum for each objective.

  ### How The Optimum Candidate Is Found

  The acquisition function is numerically optimized in the normalized latent space:

  U_cand, _ = optimize_acqf(
      acq,
      bounds=bounds,
      q=Q_BATCH,
      num_restarts=N_RESTARTS,
      raw_samples=RAW_SAMPLES,
      sequential=True,
  )

  See BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_CU_updated.py:1265.

  The process is:

  - Define a trust region around the labeled peptide having the highest mean objective value.
  - Generate RAW_SAMPLES = 64 initial latent points inside that trust region.
  - Use these samples to initialize acquisition-function optimization.
  - Run local optimization using only N_RESTARTS = 1.
  - Return the latent point maximizing joint expected hypervolume improvement.

  Because Q_BATCH = 1, only one optimal latent vector is returned per BO iteration.

  ### From Latent Optimum To Peptides

  The selected latent vector is decoded into up to eight peptide sequences:

  cand_peps_raw = decode_multiple_candidates(
      model, Z_cand, n_samples_per_z=8, temperature=0.9
  )

  See BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_CU_updated.py:1286.

  For the single optimized latent vector, decoding produces:

  - One argmax peptide.
  - Seven stochastic peptide samples.

  After removing previously seen or training peptides, all remaining decoded peptides are evaluated by the black-box objectives and added to the BO dataset. The code does not use
  the acquisition function to rank or select among these eight decoded peptides.

  So, strictly speaking:

  - The acquisition optimizer finds one optimum latent vector.
  - Multiple peptides are sampled around that latent vector.
  - Every novel sampled peptide is evaluated, rather than selecting one optimal peptide.
