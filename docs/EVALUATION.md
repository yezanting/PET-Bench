# PET-Bench evaluation guide

[Back to README](../README.md) · [Dataset guide](DATASET.md)

## Task entry points

Run commands from the repository root after editing each script's configuration. All paths below are under `eval_code/Lingshu-7B/`.

| Task | Script |
| --- | --- |
| Level 1: Tracer identification | [Lingshu7B_eval_level1_PET_tracer_identification.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level1_PET_tracer_identification.py) |
| Level 2: Image quality | [Lingshu7B_eval_level2_PET_image_quality_v2_prompt.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level2_PET_image_quality_v2_prompt.py) |
| Level 3: Organ recognition | [Lingshu7B_eval_level3_PET_Organ_Identification.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level3_PET_Organ_Identification.py) |
| Level 4a: Normal/abnormal classification | [Lingshu7B_eval_level4_PET_Normal_Abnormal_classification.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level4_PET_Normal_Abnormal_classification.py) |
| Level 4b: Region localization | [Lingshu7B_eval_Level4_Region_box_Color_Choice_Localization_v2_last.py](../eval_code/Lingshu-7B/Lingshu7B_eval_Level4_Region_box_Color_Choice_Localization_v2_last.py) |
| Level 5: Direct diagnosis | [Lingshu7B_eval_level5_PET_cancer_diagnose.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level5_PET_cancer_diagnose.py) |
| Level 5: CoT diagnosis | [Lingshu7B_eval_level5_PET_cancer_diagnose_CoT.py](../eval_code/Lingshu-7B/Lingshu7B_eval_level5_PET_cancer_diagnose_CoT.py) |

The examples use `Qwen2_5_VLForConditionalGeneration`, the matching `AutoProcessor`, and `qwen_vl_utils.process_vision_info`. They load weights in BF16 with automatic device placement and FlashAttention 2. Follow the [environment instructions](../README.md#1-clone-the-code-and-prepare-an-environment) and the [Lingshu model card](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B).

## Configure a run

Edit all five settings in the selected script's `__main__` block. For example:

```python
MODEL_PATH = "lingshu-medical-mllm/Lingshu-7B"
PROCESSOR_PATH = MODEL_PATH
JSON_FILE_PATH = "/absolute/path/to/task_annotations.json"
DATASET_ROOT_DIR = "/absolute/path/to/extracted/task"
OUTPUT_LOG_DIR = "./eval_results/lingshu7b_level1"
```

Keep the output-directory setting used by the function call at the bottom of the script. Some scripts compute `OUTPUT_LOG_DIR` after the other constants; replace that assignment too. A local checkpoint directory is also supported.

Before inference, run the [path check](DATASET.md#check-paths-before-inference). The scripts' placeholder guards only recognize some placeholder strings, so an unchanged development path may fail or silently skip records.

### Direct-answer evaluation

```bash
python eval_code/Lingshu-7B/Lingshu7B_eval_level1_PET_tracer_identification.py
python eval_code/Lingshu-7B/Lingshu7B_eval_level2_PET_image_quality_v2_prompt.py
python eval_code/Lingshu-7B/Lingshu7B_eval_level3_PET_Organ_Identification.py
python eval_code/Lingshu-7B/Lingshu7B_eval_level4_PET_Normal_Abnormal_classification.py
python eval_code/Lingshu-7B/Lingshu7B_eval_Level4_Region_box_Color_Choice_Localization_v2_last.py
python eval_code/Lingshu-7B/Lingshu7B_eval_level5_PET_cancer_diagnose.py
```

Each command evaluates a separately configured task. The direct scripts iterate through the entire supplied JSON, use `max_new_tokens=128`, and write fresh logs. They do not provide a `--limit` option or resume support. For a smoke run, use a small, separately saved annotation subset and a distinct output directory; report its size and do not label the result as a full-benchmark score.

The direct prompt lists the question and lettered options and asks for the correct letter. Level 2 uses a dedicated prompt asking for two selections: one for quality and one for artifact presence. Its prompt formatter replaces the free-form record question with this standardized instruction.

### Level-5 CoT evaluation

```bash
python eval_code/Lingshu-7B/Lingshu7B_eval_level5_PET_cancer_diagnose_CoT.py
```

The function call exposes three additional settings:

| Setting | Default in the released example | Effect |
| --- | --- | --- |
| `max_images_per_patient` | `15` | Maximum slices supplied for a case |
| `max_new_tokens` | `1024` | Output budget for the six-step rationale and answer |
| `resume` | `True` | Skip patient IDs already present in the existing CoT CSV |

The CoT prompt asks the model to identify the tracer, reflect on physiological uptake, assess image quality, detect abnormal uptake, reason about disease, and conclude with `Final Answer: [Letter]`. It is defined in `format_prompt_for_vqa_with_cot` in the script. For exact manuscript prompting, compare this example with [Appendix C](https://arxiv.org/html/2601.02737v2#A3); the script contains expanded examples and is not a verbatim copy of the appendix template.

Patient IDs used for logs and resume are derived from the second slash-separated component of the first `image_paths` entry. The reference schema therefore expects paths shaped like `patient_slice_sequences/patient_id/slice_000.png`. If you change that layout, update the ID extraction and verify that resume IDs remain unique.

## Output files

Each direct-answer run writes:

```text
OUTPUT_LOG_DIR/
├── evaluation_log.csv
├── summary_accuracy.txt
└── summary_class_distribution.txt
```

The CSV contains the image path or patient ID, question, options, ground-truth answer, raw model output, and correctness flag. Distribution summaries compare ground-truth and predicted categories; they are not per-class accuracy reports.

The CoT run writes:

```text
OUTPUT_LOG_DIR/
├── evaluation_log_cot.csv
├── cot_reasoning_log.csv
├── summary_accuracy_cot.txt
└── summary_class_distribution_cot.txt
```

`cot_reasoning_log.csv` stores extracted reasoning-step text. It is not an LLM judge score. Some console and summary strings in the CoT example still say “Lingshu32B”; the loaded checkpoint is determined by `MODEL_PATH`, so record that path with the experiment.

### Resume behavior

With `resume=True`, the CoT script appends to existing CSVs and skips IDs already recorded in `evaluation_log_cot.csv`. A recorded processing-error row also counts as an existing ID; audit failed rows before expecting a resumed run to retry them.

The accuracy and class-distribution summaries are rewritten using **only newly evaluated samples in the current invocation**. They are not cumulative metrics for the appended CSV. A run with no new samples can therefore write a zero-sample summary. To obtain a full-run result, audit the combined CSV, remove duplicates according to your run definition, account for missing/failed cases, and recompute metrics from the complete sample set. Use a new directory or `resume=False` for a fresh run; `resume=False` overwrites existing logs.

## Scoring and comparability

Accuracy is the number of correct answers divided by the number of evaluated questions, usually reported as a percentage. The precise extraction rule and denominator must accompany a result. The current examples implement the following behavior:

| Case | Released example behavior | Implication |
| --- | --- | --- |
| Direct single-choice answer | Correct if the ground-truth letter appears anywhere as a whole word, case-insensitively | An output mentioning several options can be accepted; this is not unique-final-answer extraction |
| Level-2 answer | Correct if all ground-truth letters appear as whole words | Extra incorrect letters are not rejected; this is not exact-set matching |
| CoT answer | Uses the first `Final Answer: X` match, then falls back to ground-truth-letter presence | A missing or repeated final-answer marker changes the interpretation |
| Missing image | Sample is skipped before prediction counting | Incomplete image extraction can reduce the denominator |
| Processing exception | An error row is logged, generally without incrementing `total_predictions` | The CSV's failed rows and reported denominator may differ |
| No valid answer after successful generation | Marked incorrect | Invalid outputs remain part of the successfully generated sample count |

The category-extraction helpers also scan letters, and can select the first matching option even when the correctness parser finds a later ground-truth letter. Treat distribution files as diagnostics and retain raw outputs for auditing.

The paper describes final-option parsing in the main text and an auxiliary CoT accuracy judge in Appendix C. The released CoT example uses local regex parsing and does not call an auxiliary evaluator. Consequently, running these examples does not by itself establish an exact reproduction of all manuscript metrics.

For a strict evaluation, extract the predicted answer independently of the ground truth, require one unambiguous option for single-choice tasks, and use exact-set comparison for Level 2. Report any parser changes explicitly. Account for every intended benchmark record: distinguish correctness among completed predictions from evaluation coverage, and do not silently compare a partial run with a complete one.

### CoT correctness versus plausibility

The paper treats these as separate quantities:

- **Diagnostic accuracy:** whether the final diagnosis matches the ground-truth answer. The Appendix C accuracy judge receives the question, options, ground truth, and generated reasoning.
- **Reasoning plausibility:** a text-based evaluator scores logical coherence, medical accuracy, completeness, and depth, each from 0 to 0.25, summing to a score from 0 to 1. This evaluator is blind to the ground-truth label.

A high plausibility score does not verify that a rationale is grounded in the PET images. The auxiliary judge implementations are not included here. If you implement them, retain the prompt, evaluator model/version, truncation policy, raw responses, and parsing rules separately from the inference logs.

## Adapting another model

For another compatible Qwen2.5-VL checkpoint, replace both model and processor paths. For a different architecture or API, adapt model loading, image preprocessing, chat formatting, and inference while preserving benchmark record and task semantics:

1. Resolve the original image paths and preserve slice order.
2. Construct the same question and option mapping; use the dedicated two-label instruction for Level 2.
3. Supply all Level-5 images in one request and keep the same slice limit across models.
4. Decode only the generated response, without echoing the input prompt into answer scoring.
5. Save raw outputs and apply a documented extraction rule consistently across compared models.

Do not provide `answer`, `category`, or other ground-truth labels to the evaluated model. They are used for scoring and audit after generation.

## Using VLMEvalKit

PET-Bench annotations can be integrated with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), following its [custom dataset guidance](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md) and [quickstart](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md). This repository does not include the converter or custom evaluator.

Map each JSON record to an index, question, answer, category, lettered option columns, and the image representation required by your selected VLMEvalKit revision. For single-image tasks, resolve the original image reference. For Level 5, use the framework's supported multi-image representation or a custom dataset class; do not flatten each slice into an independent diagnosis question.

Level 2 needs a multi-label evaluator with a declared set-comparison policy. A standard single-choice accuracy evaluator is insufficient. Preserve original option order, prevent accidental record drops, and record both the VLMEvalKit commit and any prompt/parser modifications.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Dataset download returns 401/403 | Complete the access request and authenticate with the approved Hugging Face account |
| Tar fails to decompress a shard | Install zstd and use GNU tar with `--zstd`; verify `SHA256SUMS` first |
| Images are skipped or the evaluated count is unexpectedly small | Check task root resolution and verify every image referenced by the JSON |
| Model loading fails at FlashAttention | Install a CUDA/PyTorch-compatible FlashAttention build; an explicitly documented `attn_implementation="sdpa"` adaptation may be used on supported setups |
| BF16 or CUDA errors | Check GPU support and the installed PyTorch/CUDA combination against model dependencies |
| Out of GPU memory at Level 5 | Check model placement and image resolution; document any resolution or slice-count changes because they alter the comparison |
| A resumed run reports few or zero evaluated samples | Inspect the appended CoT CSV; the summary reflects only new samples |

For reproducible reporting, save dataset and model revisions, code commit, dependency versions, hardware, generation settings, prompt, parser, intended sample count, completed count, failure count, and raw-output logs. AVA runs also require the training configuration and patient-level split across Levels 1–5.
