# PET-Bench

**Unveiling and Bridging the Functional Perception Gap in MLLMs: Atomic Visual Alignment and Hierarchical Evaluation via PET-Bench**

[![Paper](https://img.shields.io/badge/arXiv-2601.02737-b31b1b)](https://arxiv.org/abs/2601.02737)
[![Dataset](https://img.shields.io/badge/Hugging_Face-PET--Bench-yellow)](https://huggingface.co/datasets/TZT21999/PET-Bench)
[![Code](https://img.shields.io/badge/GitHub-PET--Bench-black)](https://github.com/yezanting/PET-Bench)

PET-Bench evaluates whether multimodal large language models (MLLMs) can interpret **functional and molecular activity in PET images**. It contains **52,308 question–answer pairs derived from 9,732 PET studies**, spanning eight data centers in Asia and Europe and four tracer families: FDG, PSMA, FAPI, and MET.

The benchmark follows a five-level clinical workflow, from identifying the tracer to diagnosing disease. Level 4 contains two subtasks, giving **six evaluation tasks** in total. Models receive PET images without CT or PET/CT fusion overlays, so they must interpret tracer biodistribution and uptake patterns from the functional signal itself.

## Contents

- [Why PET-Bench?](#why-pet-bench)
- [Workflow overview](#workflow-overview)
- [Tasks and data](#tasks-and-data)
- [Repository contents](#repository-contents)
- [Getting started](#getting-started)
- [Evaluation](#evaluation)
- [Atomic Visual Alignment](#atomic-visual-alignment)
- [Citation](#citation)
- [License and contact](#license-and-contact)

For implementation details, see the [dataset guide](docs/DATASET.md) and [evaluation guide](docs/EVALUATION.md).

## Why PET-Bench?

PET interpretation depends on tracer-specific physiological uptake, relative intensity, noise, and the spatial distribution of abnormal activity. PET-Bench organizes these skills into tasks that follow the clinical reading workflow, allowing each stage to be evaluated separately.

PET-Bench separates these skills into individual tasks to help locate failures: does a model misidentify the tracer, mistake noise for disease, fail to recognize an organ, overlook an abnormal focus, or misinterpret findings at diagnosis?

The accompanying [paper](https://arxiv.org/abs/2601.02737) describes the benchmark design, prompting protocols, and **Atomic Visual Alignment (AVA)** training strategy.

## Workflow overview

The workflow covers data curation, task construction, inference, and scoring:

1. **Prepare PET studies.** Collect multi-center, multi-tracer studies, normalize the PET volumes to SUV, and establish verified organ, lesion, and diagnostic annotations.
2. **Construct task inputs.** Select task-specific coronal slices, add organ or region cues where required, and retain up to 15 ordered slices for each diagnosis case.
3. **Build QA records.** Store image references, questions, shuffled answer options, and the corresponding ground-truth labels in task-specific JSON files.
4. **Prepare an evaluation run.** Download and extract the release, check image paths, load a model and matching processor, and configure a separate output directory for each task.
5. **Run inference.** Evaluate single-image tasks or multi-image diagnosis. For Level-5 CoT, use the six-step diagnostic prompt and preserve the generated rationale and final answer.
6. **Parse and audit outputs.** Apply the task's answer-extraction rule, check missing or failed samples, and retain raw outputs and run settings for reproducibility.

Steps 1–3 describe dataset curation; users of the released dataset start at step 4. The task hierarchy organizes the benchmark, but the example scripts evaluate each task independently. They do not automatically feed predicted answers from Levels 1–4 into Level 5.

## Tasks and data

| Level | Task | Visual input | Expected answer | Skill assessed |
| --- | --- | --- | --- | --- |
| 1 | Tracer Identification (TI) | One coronal PET slice | One option letter | Recognizing tracer-specific biodistribution |
| 2 | Image Quality Assessment (IQA) | One coronal PET slice | Two option letters | Assessing quality and artifact presence together |
| 3 | Organ Recognition (OR) | One PET slice with a highlighted organ | One option letter | Recognizing functional anatomy |
| 4a | Abnormality Identification (AI) | One coronal PET slice | One option letter: Normal or Abnormal | Separating physiological and pathological uptake |
| 4b | Abnormal Area Detection (AAD) | One PET slice with colored region boxes | One option letter identifying a box color | Localizing abnormal uptake among candidate regions |
| 5 | Disease Diagnosis (DD) | An ordered sequence of up to 15 PET slices | One diagnostic option letter | Integrating uptake distribution across slices |

The Level-5 reference set contains **471 cases**, with ground-truth diagnoses of lung cancer, lymphoma, and melanoma. Answer options may include additional diagnostic distractors; the set of ground-truth classes is not the complete option vocabulary.

The source studies include whole-body and total-body scanners, standard acquisitions, and reduced-count reconstructions. PET volumes are SUV-normalized before task-specific coronal images are selected. CT-derived masks support annotation during curation, but CT images are not supplied to the evaluated model.

The [dataset guide](docs/DATASET.md) explains sample counts, slice selection, JSON fields, and path resolution.

## Repository contents

```text
PET-Bench/
├── README.md
├── Fig1.jpg
├── docs/
│   ├── DATASET.md
│   └── EVALUATION.md
└── eval_code/
    └── Lingshu-7B/
        ├── Lingshu7B_eval_level1_PET_tracer_identification.py
        ├── Lingshu7B_eval_level2_PET_image_quality_v2_prompt.py
        ├── Lingshu7B_eval_level3_PET_Organ_Identification.py
        ├── Lingshu7B_eval_level4_PET_Normal_Abnormal_classification.py
        ├── Lingshu7B_eval_Level4_Region_box_Color_Choice_Localization_v2_last.py
        ├── Lingshu7B_eval_level5_PET_cancer_diagnose.py
        └── Lingshu7B_eval_level5_PET_cancer_diagnose_CoT.py
```

This repository provides **Lingshu-7B evaluation examples** for all six tasks and Level-5 CoT inference. Data are distributed separately on Hugging Face. AVA training scripts, fine-tuned checkpoints, and the auxiliary LLM judge implementation are not included in this checkout.

## Getting started

### 1. Clone the code and prepare an environment

```bash
git clone https://github.com/yezanting/PET-Bench.git
cd PET-Bench
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use a Python environment supported by your chosen PyTorch and model dependencies; Python 3.10 or 3.11 is a practical starting point. Install a CUDA-compatible PyTorch build following the [official installation instructions](https://pytorch.org/get-started/locally/), then install the inference dependencies:

```bash
python -m pip install "transformers==4.52.1" accelerate qwen-vl-utils pillow tqdm huggingface_hub
```

The [Lingshu model card](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) recommends Transformers 4.52.1. The released scripts use BF16, automatic device placement, and `flash_attention_2`; install a compatible [FlashAttention](https://github.com/Dao-AILab/flash-attention#installation-and-features) build for your CUDA/PyTorch environment before running them. These are setup instructions, not a fully pinned reproduction environment.

### 2. Obtain dataset access and download

The [Hugging Face dataset](https://huggingface.co/datasets/TZT21999/PET-Bench) is **gated**. Sign in on its page and complete the access request before downloading. Authenticate locally with an account that has access:

```bash
hf auth login
hf download TZT21999/PET-Bench --repo-type dataset --local-dir ./dataset
```

Alternatively, download through Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TZT21999/PET-Bench",
    repo_type="dataset",
    local_dir="./dataset",
)
```

The release packages image files in compressed shards. With GNU tar and zstd installed, verify and extract them:

```bash
cd dataset
sha256sum -c SHA256SUMS
mkdir -p data
for shard in shards/*.tar.zst; do
  tar --zstd -xf "$shard" -C data
done
cd ..
```

`SHA256SUMS` checks shard integrity; `manifest/manifest.jsonl` maps original file paths to shards. Keep relative paths preserved during extraction. See [release layout and path checks](docs/DATASET.md#download-layout-and-path-resolution) before configuring a task. A Git + Git LFS download alternative is documented there as well.

### 3. Configure and run a task

Obtain the [Lingshu-7B checkpoint](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B). In the chosen evaluation script, edit the configuration block under `if __name__ == "__main__":`:

| Setting | Meaning |
| --- | --- |
| `MODEL_PATH` | Local checkpoint directory or supported Hugging Face model ID |
| `PROCESSOR_PATH` | Matching processor directory or model ID |
| `JSON_FILE_PATH` | Task annotation JSON file |
| `DATASET_ROOT_DIR` | Directory joined with each record's relative image path |
| `OUTPUT_LOG_DIR` | Writable result directory; use a separate directory per experiment |

The scripts contain development-machine paths or placeholders and do not expose command-line flags for these settings. Replace them before execution. `DATASET_ROOT_DIR` must resolve the image paths in the JSON; it is not necessarily the JSON file's parent directory.

For example, after configuring Level 1:

```bash
python eval_code/Lingshu-7B/Lingshu7B_eval_level1_PET_tracer_identification.py
```

The [evaluation guide](docs/EVALUATION.md) maps every task to its script, describes outputs and CoT resume behavior, and explains how to adapt the examples to another model.

## Evaluation

PET-Bench reports answer accuracy separately for TI, IQA, OR, AI, AAD, and DD. Preserve each sample's option order: letters are positional labels, and the same category may correspond to different letters in different records.

Level 2 requires a joint answer: one selection for image quality and one for artifact presence. Level 5 uses multiple images in one model request. Its CoT variant follows six steps: tracer identification, expected physiological uptake, quality assessment, abnormal uptake detection, disease reasoning, and final diagnosis.

**Scoring details matter for reproduction.** The released examples use permissive letter matching, and missing images or processing failures can reduce the accuracy denominator. Their CoT script performs local answer parsing and reasoning-step logging; it does not run the paper's auxiliary accuracy or plausibility judges. Consult the [scoring notes](docs/EVALUATION.md#scoring-and-comparability) before comparing a new run with the paper.

For integration with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), convert the JSON annotations into a custom dataset and provide task-specific handling for Level-2 multi-label answers and Level-5 multi-image inputs. Conversion scripts and a PET-specific VLMEvalKit adapter are not included in this public checkout. See the [integration guidance](docs/EVALUATION.md#using-vlmevalkit).

## Atomic Visual Alignment

AVA uses supervised fine-tuning on **Levels 1–4 only**, with LoRA adaptation, to align models with lower-level PET perception. Level-5 diagnostic labels are excluded from training, and patients used for Level-5 testing are held out from AVA training across all lower-level tasks.

The AVA workflow is:

1. Establish a patient-level split before assembling training examples, keeping Level-5 test patients out of all training tasks and reconstructions.
2. Build the supervised training set from the tracer, quality, organ, and abnormality tasks in Levels 1–4.
3. Adapt the model with LoRA using those task labels, without including Level-5 diagnostic supervision.
4. Load the adapted checkpoint and evaluate held-out Level-5 cases with both direct-answer and CoT prompts.

The released evaluation examples can be adapted to a compatible fine-tuned checkpoint. Running AVA training and reproducing its patient split require additional artifacts not included in this checkout.

## Citation

If you use PET-Bench in your research, please cite:

```bibtex
@article{ye2026unveiling,
  title={Unveiling and Bridging the Functional Perception Gap in MLLMs: Atomic Visual Alignment and Hierarchical Evaluation via PET-Bench},
  author={Ye, Zanting and Niu, Xiaolong and Wu, Xuanbin and Han, Xu and Liu, Shengyuan and Hao, Jing and Peng, Zhihao and Sun, Hao and Lv, Jieqin and Wang, Fanghu and others},
  journal={arXiv preprint arXiv:2601.02737},
  year={2026},
  doi={10.48550/arXiv.2601.02737},
  url={https://arxiv.org/abs/2601.02737}
}
```

## License and contact

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated in the dataset documentation. Access is subject to the requirements on its Hugging Face page. This dataset license statement does not establish a separate license for the code; no code `LICENSE` file is currently included in this repository. Model checkpoints retain their respective upstream licenses.

For questions about access, reproduction, or collaboration, contact **Zanting Ye** at [yzt2861252880@gmail.com](mailto:yzt2861252880@gmail.com), or open a [GitHub issue](https://github.com/yezanting/PET-Bench/issues).

We thank the participating centers and nuclear medicine collaborators, and acknowledge [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), [Lingshu](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B), and the public PET resources described in the paper.
