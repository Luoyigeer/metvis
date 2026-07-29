
## Quick start

```bash
cd code
pip install -r requirements.txt
# Install a CUDA PyTorch build if you have a GPU:
# https://pytorch.org/get-started/locally/

python demo.py
# Windows: run_demo.bat
# Linux/macOS: bash run_demo.sh
```

What `demo.py` does:
1. Checks for `noaa_pretrain.pth` and at least one FROSI image
2. Builds depth/trans caches for the sample images
3. Loads `VisibilityModel` and NOAA weights into the temporal branch
4. Runs one forward pass on a FROSI mini-batch (`aligner=None`)
5. Runs one short backward step (skip with `--forward-only`)
6. Prints `Smoke test PASSED`

Useful flags:

```bash
python demo.py --forward-only
python demo.py --image dataset/FROSI/Fog/100/your_sample.png
```

## Directory layout

```text
code/
├── demo.py / run_demo.bat / run_demo.sh   # entry
├── checkpoints/noaa_pretrain.pth          # upload
├── dataset/FROSI/Fog/{vis}/               # upload samples
├── dataset/NOAA/{year}/                   # optional (empty OK for smoke)
├── train_stage2_image.py
├── train_stage3_joint.py
├── train_noaa.py
└── requirements.txt
```

## Optional: fuller training

Only if you have enough data (images and NOAA CSVs for Stage-3 alignment):

```bash
python scripts/preprocess_aux_maps.py --dataset frosi
python train_stage2_image.py --dataset frosi --epochs 15
python train_stage3_joint.py --dataset frosi --epochs 15 --freeze_fusion 5
```

Optional Stage-1 retrain (needs a large NOAA tree):

```bash
python train_noaa.py --epochs 30
```

Other helpers:

```bash
python experiments/verify_dataflow.py
python experiments/run_baselines.py --dataset frosi
python experiments/run_module_ablation.py --dataset frosi
```

## Notes

- FROSI images use a fixed placeholder timestamp (`2020-01-01 00:00`).
- Smoke demo uses image-only alignment (`aligner=None`); NOAA CSVs are not required.
- Sample subsets only validate the pipeline.
