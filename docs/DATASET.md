# PET-Bench dataset guide

[Back to README](../README.md) · [Evaluation guide](EVALUATION.md) · [Hugging Face dataset](https://huggingface.co/datasets/TZT21999/PET-Bench)

## Benchmark composition

PET-Bench contains 52,308 QA pairs derived from 9,732 PET studies across eight data centers. A study can contribute several images and tasks, so QA-pair counts are not counts of independent patients. Some source cohorts also include multiple dose/count reconstructions.

The following counts describe the curated reference annotation snapshot checked for this documentation. They sum to the paper's reported total. Recount your downloaded revision before reporting results; the gated Hub annotation contents were not independently downloaded during this documentation update.

| Task | QA pairs | Ground-truth categories in the reference snapshot |
| --- | ---: | --- |
| Level 1: Tracer identification | 8,694 | FDG, PSMA, FAPI, MET |
| Level 2: Image quality | 20,031 | Joint quality and artifact labels |
| Level 3: Organ recognition | 13,045 | Brain, heart, lung, liver, spleen, stomach, kidney, bladder |
| Level 4a: Abnormality identification | 7,702 | Normal, Abnormal |
| Level 4b: Abnormal area detection | 2,365 | Red, Yellow, Green, Blue region boxes |
| Level 5: Disease diagnosis | 471 | Lung Cancer (168), Lymphoma (135), Melanoma (168) |
| **Total** | **52,308** | |

These categories identify true labels. Multiple-choice options can contain other tracers, organs, or diagnoses as distractors. Labels and capitalization should be read from each record rather than reconstructed from a fixed vocabulary.

The paper's [Table 3](https://arxiv.org/html/2601.02737v2#S3.T3) describes the eight source groups:

| Source group | PET studies |
| --- | ---: |
| AutoPET | 1,611 |
| GHSG | 525 |
| University of Bern, Quadra | 1,750 |
| Ruijin, uExplorer | 2,100 |
| SMU Nanfang, uExplorer | 3,003 |
| SMU Nanfang, mCT | 310 |
| GPH-CM, Quadra | 150 |
| GPH-People, uExplorer | 283 |
| **Total** | **9,732** |

Bern, Ruijin, and Nanfang uExplorer totals include multiple dose reconstructions. The eight groups follow the paper's center/scanner definition and should not be interpreted as eight distinct institutions or 9,732 unique people.

## Image preparation and annotation

The [paper and appendices](https://arxiv.org/html/2601.02737v2) describe SUV normalization, task-specific coronal slice selection, and expert verification.

- **Levels 1–2:** sample the central 20% of the coronal slice stack, with a stride of five slices, to represent global biodistribution and quality.
- **Level 3:** use the coronal slice with the largest target-organ cross-section and highlight the organ. CT-derived segmentation masks are mapped to PET during annotation.
- **Level 4:** use informative lesion views for abnormality detection and colored candidate boxes for coarse localization. The task is region selection, not generation of a segmentation mask or bounding-box coordinates.
- **Level 5:** retain an ordered, sparse sequence of tumor-containing coronal slices, with up to 15 images per case, to preserve disease-distribution context.

Original volumes are SUV-normalized, but the supplied model inputs are rendered 2D images. The example evaluators read images through the model's vision utilities; they do not perform DICOM loading, NIfTI loading, SUV computation, or CT fusion. `nibabel` and `pydicom` are therefore not required for evaluating the released image-based tasks.

Annotation combines automated pre-labeling, clinical review, and senior nuclear medicine verification. CT contributes to annotation, not to the model's visual input. For AVA, the paper requires a patient-level holdout across tasks; randomly splitting individual images can leak patients or alternate reconstructions between training and diagnosis testing.

## Download layout and path resolution

Request access on the [dataset page](https://huggingface.co/datasets/TZT21999/PET-Bench), then follow the README's [download instructions](../README.md#2-obtain-dataset-access-and-download).

The Hub file listing inspected for this guide contains:

```text
dataset/
├── Disease_diagnosis/
│   └── pet_3class_sampled15_vqa_randomized_improve_prompt.json
├── Normal_Abnormal_Classification/
│   └── pet_normal_abnormal_classification_vqa_v2_shuffled.json
├── PET-Organ_Identification/
│   └── pet_organ_identification_vqa_v2.json
├── PET_Tracer_Identification/
│   └── pet_tracer_identification_qa.json
├── Region_box_Color_Choice_Localization/
│   └── pet_region_color_choice_vqa_v4_mapped_colors.json
├── SHA256SUMS
├── manifest/
│   └── manifest.jsonl
└── shards/
    ├── petbench-00000.tar.zst
    ├── petbench-00001.tar.zst
    └── petbench-00002.tar.zst
```

This is the **distribution layout**, not a claim that extracted images live beside the top-level JSON files. Level-2 annotations are not separately listed at the Hub root; inspect the extracted shard contents and manifest for that task. If a needed annotation is absent from the downloaded release, contact the authors rather than substitute another task's file.

The localization release uses a `mapped_colors` annotation filename; select that release's JSON and its matching images rather than assuming the development filename in the evaluation script is current.

To inspect archive paths without extracting a shard:

```bash
tar --zstd -tf dataset/shards/petbench-00000.tar.zst | head -n 30
```

The scripts resolve paths with `os.path.join(DATASET_ROOT_DIR, relative_path)`. If a record stores `images/image_000001.png`, set the root to the directory containing that task's `images/` folder. For diagnosis, the root should contain the `patient_slice_sequences/` folder referenced by its JSON. Avoid flattening or renaming extracted folders.

### Git + Git LFS alternative

With Git LFS installed and an approved account authenticated, you can download the distribution using Git. The Hugging Face token must also be available to Git's credential helper; logging in only for Python downloads may not configure Git credentials. See the [official authentication guide](https://huggingface.co/docs/huggingface_hub/guides/cli).

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/TZT21999/PET-Bench dataset-lfs
cd dataset-lfs
git lfs pull
sha256sum -c SHA256SUMS
mkdir -p data
for shard in shards/*.tar.zst; do
  tar --zstd -xf "$shard" -C data
done
```

## Annotation format

Task annotations are JSON arrays of records. The examples below illustrate the schema with synthetic paths and options; they are not actual benchmark cases.

### Single-image, single-answer tasks: Levels 1, 3, 4a, and 4b

```json
[
  {
    "image_path": "images/example.png",
    "question": "What is the radiotracer used in this PET scan?",
    "options": ["FDG", "PSMA", "FAPI", "MET"],
    "answer": "A",
    "category": "FDG",
    "dataset": "example_dataset"
  }
]
```

`image_path` is resolved relative to the task root. `options` defines the letter-to-text mapping: A is index 0, B is index 1, and so on. `answer` is the ground-truth letter; `category` is its semantic label. Do not reorder options without updating the answer.

### Multi-label image quality: Level 2

```json
[
  {
    "image_path": "images/example_quality.png",
    "question": "Assess image quality and artifact presence.",
    "options": ["Low Quality", "High Quality", "Artifacts Present", "No Artifacts"],
    "answer": ["A", "D"],
    "category": "Low Quality; No Artifacts",
    "dataset": "example_dataset"
  }
]
```

`answer` is a list of letters, representing one quality label and one artifact label. Options are shuffled per record, so their positions are not fixed. This task cannot be scored as ordinary single-choice VQA.

### Multi-image diagnosis: Level 5

```json
[
  {
    "image_paths": [
      "patient_slice_sequences/example_case/slice_000.png",
      "patient_slice_sequences/example_case/slice_001.png"
    ],
    "question": "What is the most likely primary diagnosis?",
    "options": ["Lung Cancer", "Lymphoma", "Melanoma", "Infection / Inflammation"],
    "answer": "B",
    "category": "Lymphoma",
    "dataset": "example_dataset"
  }
]
```

Preserve `image_paths` order and send all slices for one case in a single inference request. The reference evaluators cap input at 15 images. If a custom record has more than 15, they keep the first 15; they do not resample it during inference.

## Check paths before inference

Replace the two paths below with your annotation file and extracted task root. This checks every image reference without loading a model or printing case identifiers:

```python
import json
from pathlib import Path

annotation = Path("/absolute/path/to/task_annotations.json")
task_root = Path("/absolute/path/to/extracted/task")
records = json.loads(annotation.read_text(encoding="utf-8"))
missing = 0
references = 0
for record in records:
    paths = record["image_paths"] if "image_paths" in record else [record["image_path"]]
    for relative_path in paths:
        references += 1
        missing += not (task_root / relative_path).is_file()
print({"records": len(records), "image_references": references, "missing": missing})
assert missing == 0, "Resolve missing image paths before evaluation."
```

Record the dataset revision, task record counts, and checksums with your results. For AVA experiments, retain the patient mapping and split definition; anonymized task filenames alone do not establish patient independence.
