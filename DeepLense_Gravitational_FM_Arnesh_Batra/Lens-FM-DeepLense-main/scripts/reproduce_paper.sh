#!/usr/bin/env bash
# Reproduce every Lens-LeJEPA number in the paper, in order.
#   1. pretrain Lens-LeJEPA and the LeJEPA control on Model I (~6 GPU-h each)
#   2. classification: Model II at 100/500/1000/5000/full labels per class + Model III full
#   3. adapter ablation on the Lens-LeJEPA encoder
#   4. axion mass regression and synthetic 2x super-resolution on Models II and III
#   5. baselines: supervised models and the 300-epoch I-JEPA / Lens-JEPA reference encoders
# Usage: scripts/reproduce_paper.sh [extra --set overrides, e.g. device=cuda:1]
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA=("$@")
PY=${PYTHON:-python}

LENS=runs/pretrain/lens_lejepa/lens_lejepa_model_i_seed42/last_pretrain.pt
CTRL=runs/pretrain/lejepa/lejepa_model_i_seed42/last_pretrain.pt

[ -f "$LENS" ] || $PY -m lens_lejepa pretrain --config configs/pretrain/lens_lejepa.yaml --set "${EXTRA[@]}"
[ -f "$CTRL" ] || $PY -m lens_lejepa pretrain --config configs/pretrain/lejepa.yaml --set "${EXTRA[@]}"

cls() { $PY -m lens_lejepa finetune --config configs/downstream/classification.yaml --set "$@" "${EXTRA[@]}"; }

# Figure 2a / Table 7: label efficiency on Model II.
for shots in 100 500 1000 5000 null; do
  cls pretrained=$LENS adaptation=rslora shots=$shots run_name=lens_lejepa_rslora_shots${shots}
  cls pretrained=$LENS adaptation=lora   shots=$shots run_name=lens_lejepa_lora_shots${shots}
  cls pretrained=$CTRL adaptation=lora   shots=$shots run_name=lejepa_lora_shots${shots}
done

# Figure 2b: Model III, full data.
for pair in "$LENS rslora lens_lejepa_rslora" "$LENS lora lens_lejepa_lora" "$CTRL lora lejepa_lora"; do
  set -- $pair
  cls pretrained=$1 adaptation=$2 train_dataset=Model_III test_dataset=Model_III_test run_name=${3}_model_iii
done

# Table 5: adapter families (Model II budgets + Model III full).
for adapter in dora loraplus gated loha layerwise full; do
  extra=(); [ "$adapter" = layerwise ] && extra=("rank_schedule=[8,16,32,64]")
  for shots in 100 500 1000 5000 null; do
    cls pretrained=$LENS adaptation=$adapter shots=$shots "${extra[@]}" run_name=adapter_${adapter}_shots${shots}
  done
  cls pretrained=$LENS adaptation=$adapter train_dataset=Model_III test_dataset=Model_III_test "${extra[@]}" run_name=adapter_${adapter}_model_iii
done

# Regression and super-resolution (full data, Models II and III).
for data in Model_II Model_III; do
  $PY -m lens_lejepa finetune --config configs/downstream/regression.yaml \
      --set pretrained=$LENS adaptation=rslora train_dataset=$data test_dataset=${data}_test run_name=lens_lejepa_rslora_$data "${EXTRA[@]}"
  $PY -m lens_lejepa finetune --config configs/downstream/regression.yaml \
      --set pretrained=$CTRL adaptation=lora train_dataset=$data test_dataset=${data}_test run_name=lejepa_lora_$data "${EXTRA[@]}"
  for baseline in resnet18 lenspinn; do
    $PY -m lens_lejepa finetune --config configs/downstream/supervised.yaml \
        --set task=regression backbone=$baseline weight_decay=0.01 label_smoothing=0.0 \
              train_dataset=$data test_dataset=${data}_test run_name=${baseline}_regression_$data "${EXTRA[@]}"
  done
  for pair in "$LENS lens_lejepa" "$CTRL lejepa"; do
    set -- $pair
    $PY -m lens_lejepa finetune --config configs/downstream/super_resolution.yaml \
        --set pretrained=$1 adaptation=lora train_dataset=$data test_dataset=${data}_test run_name=${2}_lora_$data "${EXTRA[@]}"
  done
  for baseline in rcan bicubic; do
    $PY -m lens_lejepa finetune --config configs/downstream/super_resolution.yaml \
        --set backbone=$baseline train_dataset=$data test_dataset=${data}_test run_name=${baseline}_$data "${EXTRA[@]}"
  done
done

# Supervised classification baselines (Table 7).
for baseline in resnet18 vit vitsd lensiformer lenspinn; do
  for shots in 100 500 1000 5000 null; do
    $PY -m lens_lejepa finetune --config configs/downstream/supervised.yaml \
        --set backbone=$baseline shots=$shots run_name=${baseline}_shots${shots} "${EXTRA[@]}"
  done
  $PY -m lens_lejepa finetune --config configs/downstream/supervised.yaml \
      --set backbone=$baseline train_dataset=Model_III test_dataset=Model_III_test run_name=${baseline}_model_iii "${EXTRA[@]}"
done

# Reference self-supervised encoders at 3x the pretraining budget, adapted with LoRA r=32.
for method in ijepa lens_jepa lens_jepa_sym; do
  ckpt=runs/pretrain/$method/${method}_model_i_seed42/last_pretrain.pt
  [ -f "$ckpt" ] || $PY -m lens_lejepa pretrain --config configs/pretrain/$method.yaml --set "${EXTRA[@]}"
  backbone=lens_jepa; [ "$method" = ijepa ] && backbone=vit_base_3blocks
  for shots in 100 500 1000 5000 null; do
    cls backbone=$backbone pretrained=$ckpt adaptation=lora shots=$shots run_name=${method}_lora_shots${shots}
  done
  cls backbone=$backbone pretrained=$ckpt adaptation=lora train_dataset=Model_III test_dataset=Model_III_test run_name=${method}_lora_model_iii
done

$PY scripts/collect_results.py runs
