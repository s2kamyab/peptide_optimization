
4 - BO_latent_flow_VAE_lolBO_with_monitoring_clean_wandb_metaltype_updated.py
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

corrected data:

1- python create_fixed_len10_metalpdb_chain_mapped_corrected.py 
  --raw-csv metalpdb_Cu.csv 
  --target-metal Cu 
  --output-csv metalpdb_CU_chain_mapped_len10_high_confidence.csv 
  --sequence-source uniprot 
  --window-len 10 
  --negative-fraction 0.0 
  --minimum-alignment-identity 0.90 
  --dedup-level peptide_metal_labels

If the old extraction produced wrong or ambiguous peptide windows, then the diffusion model learned from those peptide windows. It did not learn the wrong labels directly, but it may have learned from a noisier set of MetalPDB-derived peptide sequences.

2- python create_fixed_len10_metalpdb_chain_mapped_corrected_chunked.py 
  --raw-csv metalpdb_all_metals_unique/metalpdb_all_metals_all_records.csv 
  --target-metal ALL 
  --output-dir metalpdb_all_metals_chain_mapped_len10_high_confidence_parts 
  --sequence-source pdb-chain 
  --window-len 10 
  --negative-fraction 0.0 
  --dedup-level peptide_metal_labels 
  --save-every-rows 5000 
  --raw-batch-rows 5000



pretrain commands:
1- python pretrain_gru_vae_all_metalpdb.py   --hidden-size 128  --latent-dim 64   --n-layers 2   --lr 3e-4   --kl-beta 0.0001   --kl-warmup-epochs 30   --epochs 50
2- python pretrain_gru_vae_all_metalpdb.py   --hidden-size 128  --latent-dim 64   --n-layers 2   --lr 3e-4   --kl-beta 0.0001   --kl-warmup-epochs 30   --epochs 100
3- python pretrain_gru_vae_all_metalpdb.py   --hidden-size 128  --latent-dim 64   --n-layers 2   --lr 3e-4   --kl-beta 0.0001   --kl-warmup-epochs 30   --epochs 150        

Fine Tuning:
1- python finetune_gru_vae_cu_with_normalizing_flow.py   --init-checkpoint transfer_gru_vae_checkpoints/pretrained_gru_vae_no_flow_250.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 128   --latent-dim 64   --n-layers 2   --vae-lr 1e-5   --flow-lr 1e-4   --epochs 200

2- python finetune_gru_vae_cu_with_normalizing_flow_step_2_diagnostics_suite.py   --init-checkpoint transfer_gru_vae_flow_finetune_step1_checkpoints\finetuned_cu_gru_vae_with_flow.pt  --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 128   --latent-dim 64   --n-layers 2   --epochs 100   --run-suite   


######################################################################################################################
Second Round
######################################################################################################################
1- python pretrain_gru_vae_all_metalpdb_h32_z32.py   --hidden-size 32   --latent-dim 32   --n-layers 2   --lr 3e-4   --kl-beta 0.0001   --kl-warmup-epochs 30   --epochs 250   --no-resume   

2- python finetune_gru_vae_cu_h32_z32_with_roundtrip_loss.py   --init-checkpoint transfer_gru_vae_checkpoints_h32_z32/pretrained_gru_vae_no_flow_h32_z32.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 32   --latent-dim 32   --n-layers 2   --n-flows 2   --vae-lr 1e-5   --flow-lr 1e-4   --kl-beta 0.01   --roundtrip-loss-weight 0.01   --roundtrip-cosine-weight 0.1   --epochs 200     bad result

2- python finetune_gru_vae_cu_h32_z32_roundtrip_multicheckpoint.py   --init-checkpoint transfer_gru_vae_checkpoints_h32_z32/pretrained_gru_vae_no_flow_h32_z32.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 32   --latent-dim 32   --n-layers 2   --n-flows 2   --vae-lr 1e-5   --flow-lr 1e-4   --kl-beta 0.01   --roundtrip-loss-weight 0.05   --roundtrip-cosine-weight 0.1   --epochs 200   

 3- python finetune_gru_vae_cu_h32_z32_no_roundtrip_control.py   --init-checkpoint transfer_gru_vae_checkpoints_h32_z32/pretrained_gru_vae_no_flow_h32_z32.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 32   --latent-dim 32   --n-layers 2   --n-flows 2   --vae-lr 1e-5   --flow-lr 1e-4   --kl-beta 0.01   --roundtrip-loss-weight 0.0   --roundtrip-cosine-weight 0.1   --epochs 200 

 4- python finetune_gru_vae_cu_h32_z32_roundtrip_multicheckpoint.py   --init-checkpoint transfer_gru_vae_checkpoints_h32_z32/pretrained_gru_vae_no_flow_h32_z32.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --hidden-size 32   --latent-dim 32   --n-layers 2   --n-flows 2   --vae-lr 1e-5   --flow-lr 1e-4   --kl-beta 0.01   --roundtrip-loss-weight 0.05   --roundtrip-cosine-weight 0.1   --epochs 500           

##########################################################Diffusion ####################################################
1- python pretrain_gru_vae_all_metalpdb_h32_z32_latent_diffusion.py ^
  --parts-dir metalpdb_all_metals_unique/binding_windows_len10/parts ^
  --hidden-size 32 ^
  --latent-dim 32 ^
  --n-layers 2 ^
  --vae-epochs 250 ^
  --vae-lr 3e-4 ^
  --kl-beta 0.0001 ^
  --kl-warmup-epochs 30 ^
  --teacher-forcing-start 1.0 ^
  --teacher-forcing-end 0.5 ^
  --diffusion-epochs 100 ^
  --diffusion-lr 1e-4 ^
  --diffusion-hidden-dim 128 ^
  --diffusion-time-dim 32 ^
  --diffusion-blocks 4 ^
  --diffusion-train-steps 100 ^
  --ddim-steps 20 ^
  --sanity-samples 512 ^
  --no-resume


2- python finetune_cu_latent_diffusion_h32_z32.py   --init-checkpoint transfer_gru_vae_latent_diffusion_checkpoints_h32_z32/pretrained_gru_vae_latent_diffusion_h32_z32.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --epochs 100   --batch-size 64   --diffusion-lr 3e-5   --score-head-lr 1e-4   --score-loss-weight 0.1   --diffusion-loss-weight 1.0   --recon-loss-weight 0.0   --ddim-steps 20   --shell-samples 512   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

3- python finetune_cu_latent_diffusion_h32_z32_preimage_refined.py   --init-checkpoint transfer_gru_vae_latent_diffusion_checkpoints_h32_z32/pretrained_gru_vae_latent_diffusion_h32_z32_old.pt   --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv   --out-dir cu_latent_diffusion_finetune_h32_z32_preimage_refined_v2   --epochs 50   --finetune-vae   --vae-lr 1e-6   --recon-loss-weight 0.1   --diffusion-loss-weight 1.0   --score-loss-weight 0.1   --preimage-opt-steps 300   --preimage-opt-lr 1e-2   --preimage-ce-weight 2.0   --preimage-h-mse-weight 0.02   --preimage-anchor-weight 0.001   --preimage-sphere-weight 0.0001

4 - python finetune_cu_latent_diffusion_h32_z32_preimage_refined_fixed.py 
  --init-checkpoint cu_latent_diffusion_finetune_h32_z32_preimage_refined_v2/last_epoch_cu_latent_diffusion.pt 
  --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv 
  --out-dir cu_latent_diffusion_finetune_h32_z32_preimage_refined_v2_preimage_export 
  --epochs 0 
  --export-inversion-coordinates 
  --export-optimized-preimage-coordinates 
  --preimage-opt-steps 300 
  --preimage-opt-lr 1e-2 
  --preimage-ce-weight 2.0 
  --preimage-h-mse-weight 0.02 
  --preimage-anchor-weight 0.001 
  --preimage-sphere-weight 0.0001


  5- python finetune_cu_latent_diffusion_h32_z32_preimage_refined_fixed.py 
  --init-checkpoint transfer_gru_vae_latent_diffusion_checkpoints_h32_z32/pretrained_gru_vae_latent_diffusion_h32_z32_old.pt 
  --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv 
  --out-dir cu_latent_diffusion_finetune_h32_z32_preimage_refined_v3 
  --epochs 100 
  --finetune-vae 
  --vae-lr 3e-6 
  --recon-loss-weight 0.3 
  --diffusion-loss-weight 0.5 
  --score-loss-weight 0.05 
  --preimage-opt-steps 500 
  --preimage-opt-lr 5e-3 
  --preimage-ce-weight 3.0 
  --preimage-h-mse-weight 0.01 
  --preimage-anchor-weight 0.0005 
  --preimage-sphere-weight 0.00005

  6 - python finetune_cu_latent_diffusion_h32_z32_adapter_refined.py 
  --train-latent-adapter 
  --init-checkpoint cu_latent_diffusion_finetune_h32_z32_preimage_refined_v3/last_epoch_cu_latent_diffusion.pt 
  --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv 
  --out-dir cu_latent_adapter_refined_v1 
  --epochs 100 
  --adapter-hidden-dim 64 
  --adapter-scale 0.1 
  --adapter-lr 1e-4 
  --adapter-recon-weight 1.0 
  --adapter-latent-anchor-weight 0.05 
  --adapter-smoothness-weight 0.01 
  --adapter-smooth-noise-std 0.05 
  --preimage-opt-steps 500 
  --preimage-opt-lr 5e-3 
  --preimage-ce-weight 3.0 
  --preimage-h-mse-weight 0.01 
  --preimage-anchor-weight 0.0005 
  --preimage-sphere-weight 0.00005

  7- python finetune_cu_latent_diffusion_h32_z32_adapter_refined.py 
  --train-latent-adapter 
  --init-checkpoint cu_latent_diffusion_finetune_h32_z32_preimage_refined_v3/last_epoch_cu_latent_diffusion.pt 
  --cu-csv metalpdb_binding_windows_len10_CU_scored_ranked.csv 
  --out-dir cu_latent_adapter_refined_v2 
  --epochs 100 
  --adapter-hidden-dim 128 
  --adapter-scale 0.2 
  --adapter-lr 3e-4 
  --adapter-recon-weight 2.0 
  --adapter-latent-anchor-weight 0.01 
  --adapter-smoothness-weight 0.001 
  --adapter-smooth-noise-std 0.05 
  --preimage-opt-steps 500 
  --preimage-opt-lr 5e-3 
  --preimage-ce-weight 3.0 
  --preimage-h-mse-weight 0.01 
  --preimage-anchor-weight 0.0005 
  --preimage-sphere-weight 0.00005



  #############################################################################################
  3rd round: diffusion model instead of GRU-VAE

    peptide one-hot x0, shape [10, 20]
        ↓
    add Gaussian diffusion noise
            ↓
    noised peptide tensor xt
            ↓
    GRU-based diffusion denoiser
            ↓
    predict noise ε and reconstructed peptide logits
            ↓
    argmax over 20 amino acids per position
            ↓
    reconstructed peptide

  1- python pretrain_peptide_direct_sequence_diffusion_search_cost_sampling_h32.py 
  --hidden-size 32 
  --n-layers 2 
  --time-dim 32 
  --diffusion-epochs 250 
  --diffusion-lr 3e-4 
  --diffusion-train-steps 100 
  --out-dir transfer_peptide_direct_sequence_diffusion_search_cost_sampling_h32 
  --no-resume

#########################################################################################
Pretrain diffusion instead of GRU-VAE using the corrected data "all metal"

1 - python pretrain_peptide_direct_sequence_diffusion_chain_mapped_parts_h32.py 
  --parts-dir metalpdb_all_metals_chain_mapped_len10_high_confidence_parts\parts 
  --part-glob metalpdb_ALL_chain_mapped_len10_high_confidence_part_*.csv 
  --expected-parts 73 
  --peptide-col peptide_len10 
  --split-col split 
  --train-split train 
  --validation-split validation 
  --test-split test 
  --out-dir transfer_peptide_direct_sequence_diffusion_chain_mapped_h32 
  --hidden-size 32 
  --n-layers 2 
  --time-dim 32 
  --diffusion-epochs 250 
  --diffusion-lr 3e-4 
  --diffusion-train-steps 100 
  --recon-ce-weight 0.5 
  --x0-mse-weight 0.1 
  --chunksize 8192 
  --validation-batches 50 
  --test-batches 0 
  --no-resume
########################################
fine tune using realnvp normalizing flow:

 1- python finetune_cu_direct_sequence_diffusion_noise_flow_h32_cudnn_fix_v2.py 
  --init-checkpoint transfer_peptide_direct_sequence_diffusion_chain_mapped_h32\best_validation_direct_sequence_diffusion_search_cost_sampling_h32.pt 
  --cu-csv metalpdb_CU_chain_mapped_len10_high_confidence.csv 
  --peptide-col peptide_len10 
  --labels-col binding_site_labels_len10 
  --score-col final_score 
  --auto-score-if-missing 
  --split-col split 
  --train-split train 
  --validation-split validation 
  --test-split test 
  --out-dir cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_conservative 
  --epochs 50 
  --batch-size 64 
  --ddim-steps 20 
  --flow-layers 4 
  --flow-hidden-dim 128 
  --flow-lr 5e-5 
  --score-head-lr 1e-4 
  --flow-recon-ce-weight 1.0 
  --flow-x0-mse-weight 0.2 
  --score-loss-weight 0.2 
  --flow-anchor-weight 0.05 
  --sphere-loss-weight 0.005 
  --preimage-steps 80 
  --preimage-lr 5e-2 
  --export-bo-coordinates

####################################### Evaluating the peptides using blackbox function ################################
python sort_training_data_CU_blackbox_consistent.py 
  --input-csv metalpdb_CU_chain_mapped_len10_high_confidence.csv 
  --output-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --summary-json metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked_summary.json 
  --objective-module black_box_fcn_mo_CU_f.py 
  --ranges-json cu_objective_fixed_ranges_training_CU_updated_margined.json 
  --compute-missing-a3d


  ########################################
fine tune again using  evaluated data to affect score head training with  realnvp normalizing flow:

 1- python finetune_cu_direct_sequence_diffusion_noise_flow_h32_cudnn_fix_v2.py 
  --init-checkpoint transfer_peptide_direct_sequence_diffusion_chain_mapped_h32\best_validation_direct_sequence_diffusion_search_cost_sampling_h32.pt 
  --cu-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --peptide-col peptide_len10 
  --labels-col binding_site_labels_len10 
  --score-col final_score 
  --split-col split 
  --train-split train 
  --validation-split validation 
  --test-split test 
  --out-dir cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored 
  --epochs 50 
  --batch-size 64 
  --ddim-steps 20 
  --flow-layers 4 
  --flow-hidden-dim 128 
  --flow-lr 5e-5 
  --score-head-lr 1e-4 
  --flow-recon-ce-weight 1.0 
  --flow-x0-mse-weight 0.2 
  --score-loss-weight 0.2 
  --flow-anchor-weight 0.05 
  --sphere-loss-weight 0.005 
  --preimage-steps 80 
  --preimage-lr 5e-2 
  --export-bo-coordinates

######################################################################################
Bayesian optimization
###################################################################################
1- python BO_gp_after_flow_direct_diffusion_noise_flow_h32_fix_v2.py 
  --flow-checkpoint cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\best_val_score_mse_cu_direct_diffusion_noise_flow.pt 
  --data-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --coordinate-csv cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_noise_flow_coordinates_for_bo.csv 
  --preimage-cache cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_epsilon0_preimage_cache.pt 
  --peptide-col peptide_len10 
  --out-dir bo_results_CU_direct_diffusion_after_flow_gp 
  --decoder-dir bo_decoder_monitoring_CU_direct_diffusion_after_flow_gp 
  --pareto-dir pareto_front_CU_direct_diffusion_after_flow_gp 
  --bo-iters 20
  --initial-labeled 256 
  --q-batch 1 
  --num-restarts 4 
  --raw-samples 64 
  --mc-samples 64 
  --tr-radius-unit 0.15 
  --ddim-steps 20 
  --decoder-samples-per-z 8 
  --local-noise-std 0.05


  RESULT: Collapse to one peptide. The model is not numerically failing.
The BO acquisition is not enough to generate novelty.
The decoder basin around the best Cu peptide is too stable under local noise.

The local-noise test confirms this: even with local perturbations around each BO candidate, the decoded peptide stays exactly HSHEEREHAE.


  2- python BO_gp_after_flow_direct_diffusion_noise_flow_h32_fix_v2.py 
  --flow-checkpoint cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\best_val_score_mse_cu_direct_diffusion_noise_flow.pt 
  --data-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --coordinate-csv cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_noise_flow_coordinates_for_bo.csv 
  --preimage-cache cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_epsilon0_preimage_cache.pt 
  --peptide-col peptide_len10 
  --out-dir bo_results_CU_direct_diffusion_after_flow_gp_explore 
  --decoder-dir bo_decoder_monitoring_CU_direct_diffusion_after_flow_gp_explore 
  --pareto-dir pareto_front_CU_direct_diffusion_after_flow_gp_explore 
  --bo-iters 30 
  --initial-labeled 512 
  --q-batch 4 
  --num-restarts 12 
  --raw-samples 256 
  --mc-samples 128 
  --tr-radius-unit 0.35 
  --ddim-steps 20 
  --decoder-samples-per-z 32 
  --local-noise-std 0.75

  RESULT: CUDA out of memory

  next fix:
  1. Uses qLogExpectedHypervolumeImprovement by default when available
  2. Optimizes q-batch one candidate at a time
  3. Catches CUDA OOM during acquisition optimization
  4. Retries with smaller temporary mc/raw/restart settings
  5. Keeps BO in the same epsilonK after-flow space

  3-1 $env:PYTORCH_ALLOC_CONF="expandable_segments:True"
  3-2 python BO_gp_after_flow_direct_diffusion_noise_flow_h32_fix_v3_oom_safe.py 
  --flow-checkpoint cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\best_val_score_mse_cu_direct_diffusion_noise_flow.pt 
  --data-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --coordinate-csv cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_noise_flow_coordinates_for_bo.csv 
  --preimage-cache cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_epsilon0_preimage_cache.pt 
  --peptide-col peptide_len10 
  --out-dir bo_results_CU_direct_diffusion_after_flow_gp_explore_v3 
  --decoder-dir bo_decoder_monitoring_CU_direct_diffusion_after_flow_gp_explore_v3 
  --pareto-dir pareto_front_CU_direct_diffusion_after_flow_gp_explore_v3 
  --bo-iters 30 
  --initial-labeled 512 
  --q-batch 4 
  --optimize-q-one-at-a-time 
  --num-restarts 6 
  --raw-samples 128 
  --mc-samples 32 
  --max-partition-points 15 
  --tr-radius-unit 0.35 
  --ddim-steps 20 
  --decoder-samples-per-z 32 
  --local-noise-std 0.75 
  --acqf qlogehvi

##########################################################################################
  Train GRU-VAE - gp after flow using new data
  4-1- python pretrain_gru_vae_all_metalpdb_h32_z32_high_confidence_data.py   --hidden-size 32   --latent-dim 32   --n-layers 2   --lr 3e-4   --kl-beta 0.0001   --kl-warmup-epochs 30   --epochs 250   --no-resume

  4-2- python finetune_gru_vae_cu_h32_z32_roundtrip_multicheckpoint_high_confidence_data.py   --init-checkpoint transfer_gru_vae_checkpoints_h32_z32_high_confidence_dataset\pretrained_gru_vae_no_flow_h32_z32.pt   --cu-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv   --hidden-size 32   --latent-dim 32   --n-layers 2   --n-flows 2   --vae-lr 1e-5   --flow-lr 1e-4   --kl-beta 0.01   --roundtrip-loss-weight 0.05   --roundtrip-cosine-weight 0.1   --epochs 500 

  Bayesian Optimization
  4-3- python BO_gp_after_flow_h32_z32_best_roundtrip_high_confidence.py
  bad result
 ################################################################################################################## 
    increase the hidden size and change normalizing flow from planar to realnvp
    
    Train GRU-VAE - gp after flow using new data

    1- python pretrain_gru_vae_all_metalpdb_h64_z64_latent_conditioned_high_confidence_data.py 
  --parts-dir metalpdb_all_metals_chain_mapped_len10_high_confidence_parts\parts 
  --file-pattern auto 
  --hidden-size 64 
  --latent-dim 64 
  --n-layers 2 
  --lr 3e-4 
  --kl-beta 0.0001 
  --kl-warmup-epochs 30 
  --epochs 250 
  --no-resume
  
  Result: KL = 0! GPT said this is deterministic autoencode and is ok to proceed with finetuning

  2- python finetune_gru_vae_cu_h64_z64_latent_conditioned_realnvp_roundtrip_high_confidence_data.py 
  --init-checkpoint transfer_gru_vae_checkpoints_h64_z64_latent_conditioned_high_confidence_dataset\pretrained_gru_vae_latent_conditioned_h64_z64.pt
  --cu-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --peptide-col peptide_len10 
  --score-col final_score 
  --hidden-size 64 
  --latent-dim 64 
  --flow-layers 4 
  --flow-hidden-dim 128 
  --epochs 300
  
  3- python BO_gp_after_flow_gru_vae_h64_z64_latent_conditioned_realnvp_high_confidence.py 
  --flow-checkpoint transfer_gru_vae_realnvp_checkpoints_h64_z64_latent_conditioned_high_confidence_data\best_score_mse_h64_z64_latent_conditioned_realnvp_roundtrip.pt 
  --data-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --bo-iters 30 
  --initial-labeled 512 
  --q-batch 2 
  --optimize-q-one-at-a-time 
  --tr-radius-unit 0.35 
  --decoder-samples-per-z 24 
  --novelty-radii 0.10 0.25 0.50 0.75
  
 4- Geoff believes the diffusion should not worsen the resutls so we compute the correlation between gru latent space and after diffusion:
python analyze_gru_vae_mu_vs_diffusion_noise_space.py 
  --gru-checkpoint transfer_gru_vae_realnvp_checkpoints_h64_z64_latent_conditioned_high_confidence_data\best_score_mse_h64_z64_latent_conditioned_realnvp_roundtrip.pt 
  --data-csv metalpdb_CU_chain_mapped_len10_high_confidence_blackbox_scored_ranked.csv 
  --diffusion-coordinate-csv cu_direct_sequence_diffusion_noise_flow_chainmapped_h32_blackbox_scored\cu_direct_diffusion_noise_flow_coordinates_for_bo.csv 
  --peptide-col peptide_len10 
  --out-dir mu_vs_diffusion_space_analysis 
  --make-plots 
  
  ### How It Works

  The surrogate models are fitted independently for each objective, but the acquisition function is calculated jointly across all four objectives.


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