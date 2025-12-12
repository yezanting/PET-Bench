import os
import json
import csv
import re
import torch
from tqdm import tqdm
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Assuming this is a utility from the model's repository


# =========================================================================
# == CHAIN-OF-THOUGHT PROMPT CONSTRUCTION
# =========================================================================

def format_prompt_for_vqa_with_cot(question: str, options: list) -> str:
    """
    Formats the question and options with a structured Chain-of-Thought (CoT) reasoning prompt.
    
    The CoT follows this diagnostic pathway:
    1. Tracer identification
    2. Physiological uptake reflection
    3. Image quality assessment
    4. Abnormal uptake detection
    5. Disease reasoning
    6. Final diagnosis
    """
    
    cot_prompt = f"""You are an expert nuclear medicine physician analyzing PET imaging. Please follow this systematic diagnostic reasoning process:

**Step 1: Tracer Identification**
First, identify the PET tracer used in this imaging study. Common tracers include:
- FDG (18F-Fluorodeoxyglucose): glucose metabolism imaging
- PSMA (Prostate-Specific Membrane Antigen): prostate cancer imaging
- Ga-68 DOTATATE: neuroendocrine tumor imaging
- Other specialized tracers

**Step 2: Physiological Uptake Reflection**
Based on the identified tracer, list which organs and regions should show normal physiological uptake. For example:
- FDG: brain, heart, liver, kidneys, bladder, bone marrow
- PSMA: salivary glands, liver, spleen, kidneys, small intestine
Consider the expected normal biodistribution pattern.

**Step 3: Image Quality Assessment**
Evaluate the image quality:
- Is the image clear and well-contrasted?
- Are there any significant artifacts or noise?
- Can anatomical structures be clearly identified?
- If image quality is poor, note that abnormal-appearing regions might be artifacts rather than true pathology.

**Step 4: Abnormal Uptake Detection**
Systematically examine the entire body and identify:
- Which organs or regions show uptake in the images
- Which of these uptake regions are BEYOND normal physiological distribution
- Describe the location, intensity, and pattern of abnormal uptake
- Distinguish between physiological uptake and pathological uptake

**Step 5: Disease Reasoning**
For each abnormal uptake region identified:
- Consider the differential diagnosis
- What diseases most commonly present with this uptake pattern?
- Consider the anatomical location and tracer characteristics
- Integrate all findings to form a comprehensive diagnosis

**Step 6: Final Diagnosis**
Based on your systematic analysis above, provide your final diagnosis.

{question}

Options:
"""
    
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        cot_prompt += f"{letter}. {option_text}\n"
    
    cot_prompt += """
Please provide your complete reasoning following the 6 steps above, and then clearly state your final answer as: "Final Answer: [Letter]"
"""
    
    return cot_prompt


# =========================================================================
# == CORE EVALUATION LOGIC
# =========================================================================

def parse_model_answer(model_output: str, correct_letter: str) -> bool:
    """
    Parses the model's text output to check for correctness.
    Looks for the answer in "Final Answer: X" format or standalone letter.
    """
    # First try to find "Final Answer: X" pattern
    final_answer_match = re.search(r"Final Answer:\s*([A-Z])", model_output, re.IGNORECASE)
    if final_answer_match:
        predicted_letter = final_answer_match.group(1).upper()
        return predicted_letter == correct_letter.upper()
    
    # Fallback: search for the letter anywhere in the output
    if re.search(f"\\b{correct_letter}\\b", model_output, re.IGNORECASE):
        return True
    
    return False


def get_predicted_category(model_output: str, options: list) -> str:
    """
    Parses the model's output to determine which category it chose.
    Prioritizes "Final Answer: X" format.
    """
    # First try to find "Final Answer: X" pattern
    final_answer_match = re.search(r"Final Answer:\s*([A-Z])", model_output, re.IGNORECASE)
    if final_answer_match:
        letter = final_answer_match.group(1).upper()
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            return options[idx]
    
    # Fallback: search for any letter
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        if re.search(f"\\b{letter}\\b", model_output, re.IGNORECASE):
            return option_text
    
    return "Unknown/No Answer"


def load_evaluated_patients(log_file_path: str) -> set:
    """
    Load the list of patient IDs that have already been evaluated from the log file.
    Returns a set of patient_ids that have been processed.
    """
    evaluated_patients = set()
    
    if not os.path.exists(log_file_path):
        return evaluated_patients
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row.get('patient_id', '').strip()
                if patient_id and patient_id != 'N/A':
                    evaluated_patients.add(patient_id)
        print(f"Found {len(evaluated_patients)} already evaluated patients in existing log.")
    except Exception as e:
        print(f"Warning: Could not read existing log file: {e}")
        print("Starting fresh evaluation.")
    
    return evaluated_patients


def extract_reasoning_steps(model_output: str) -> dict:
    """
    Extracts the reasoning from each CoT step for logging and analysis.
    Returns a dictionary with extracted reasoning for each step.
    """
    reasoning = {
        "step1_tracer": "",
        "step2_physiological": "",
        "step3_quality": "",
        "step4_abnormal": "",
        "step5_disease": "",
        "step6_final": ""
    }
    
    # Extract Step 1: Tracer Identification
    step1_match = re.search(r"\*\*Step 1:.*?\*\*(.*?)(?=\*\*Step 2:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step1_match:
        reasoning["step1_tracer"] = step1_match.group(1).strip()[:200]  # Limit length
    
    # Extract Step 2: Physiological Uptake
    step2_match = re.search(r"\*\*Step 2:.*?\*\*(.*?)(?=\*\*Step 3:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step2_match:
        reasoning["step2_physiological"] = step2_match.group(1).strip()[:200]
    
    # Extract Step 3: Image Quality
    step3_match = re.search(r"\*\*Step 3:.*?\*\*(.*?)(?=\*\*Step 4:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step3_match:
        reasoning["step3_quality"] = step3_match.group(1).strip()[:200]
    
    # Extract Step 4: Abnormal Uptake
    step4_match = re.search(r"\*\*Step 4:.*?\*\*(.*?)(?=\*\*Step 5:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step4_match:
        reasoning["step4_abnormal"] = step4_match.group(1).strip()[:200]
    
    # Extract Step 5: Disease Reasoning
    step5_match = re.search(r"\*\*Step 5:.*?\*\*(.*?)(?=\*\*Step 6:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step5_match:
        reasoning["step5_disease"] = step5_match.group(1).strip()[:200]
    
    # Extract Step 6: Final Diagnosis
    step6_match = re.search(r"\*\*Step 6:.*?\*\*(.*?)(?=Final Answer:|$)", model_output, re.DOTALL | re.IGNORECASE)
    if step6_match:
        reasoning["step6_final"] = step6_match.group(1).strip()[:200]
    
    return reasoning


def evaluate_lingshu_on_multi_slice_vqa_cot(
        model_path: str,
        processor_path: str,
        json_path: str,
        dataset_root_dir: str,
        output_dir: str,
        max_images_per_patient: int = 15,
        max_new_tokens: int = 1024,  # Increased for CoT reasoning
        resume: bool = True  # Enable resume by default
):
    """
    Main function to evaluate the Lingshu model on a multi-slice VQA dataset
    using Chain-of-Thought (CoT) reasoning.
    """
    print("--- Starting Lingshu Multi-Slice VQA Evaluation with CoT ---")

    # --- 1. Setup Model and Processor ---
    print(f"Step 1/6: Loading model from '{model_path}'...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(processor_path)
    except Exception as e:
        print(f"Error loading model or processor: {e}")
        return
    print("Model and processor loaded successfully.")

    # --- 2. Load and Prepare Data ---
    print(f"Step 2/6: Loading JSON data from '{json_path}'...")
    with open(json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"Found {len(qa_data)} QA pairs to evaluate.")

    # --- 3. Setup Logging with Resume Support ---
    print(f"Step 3/6: Setting up output directory and log files...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Main evaluation log
    log_file_path = os.path.join(output_dir, 'evaluation_log_cot.csv')
    cot_log_file_path = os.path.join(output_dir, 'cot_reasoning_log.csv')
    
    # Check for existing evaluations if resume is enabled
    evaluated_patients = set()
    if resume:
        print("Resume mode enabled. Checking for existing evaluations...")
        evaluated_patients = load_evaluated_patients(log_file_path)
        if evaluated_patients:
            print(f"Will skip {len(evaluated_patients)} already evaluated patients.")
    
    # Determine file mode: append if resuming and file exists, otherwise write new
    file_exists = os.path.exists(log_file_path)
    file_mode = 'a' if (resume and file_exists) else 'w'
    cot_file_exists = os.path.exists(cot_log_file_path)
    cot_file_mode = 'a' if (resume and cot_file_exists) else 'w'
    
    log_file = open(log_file_path, file_mode, newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if file_mode == 'w':  # Only write header for new files
        log_writer.writerow([
            "patient_id", "image_paths", "question", "options", "ground_truth_letter",
            "ground_truth_text", "model_raw_output", "is_correct"
        ])
    
    cot_log_file = open(cot_log_file_path, cot_file_mode, newline='', encoding='utf-8')
    cot_log_writer = csv.writer(cot_log_file)
    if cot_file_mode == 'w':  # Only write header for new files
        cot_log_writer.writerow([
            "patient_id", "step1_tracer", "step2_physiological", "step3_quality",
            "step4_abnormal", "step5_disease", "step6_final", "is_correct"
        ])

    correct_predictions, total_predictions = 0, 0
    gt_counts, pred_counts = Counter(), Counter()
    skipped_count = 0

    # --- 4. Run Evaluation Loop ---
    remaining_samples = len(qa_data) - len(evaluated_patients)
    print(f"Step 4/6: Running inference with CoT reasoning (max {max_images_per_patient} images/patient)...")
    print(f"Total samples: {len(qa_data)}, Already evaluated: {len(evaluated_patients)}, Remaining: {remaining_samples}")
    try:
        for item in tqdm(qa_data, desc="Evaluating with CoT"):
            try:
                # --- Multi-image path handling ---
                image_paths_relative = item['image_paths']
                patient_id = image_paths_relative[0].split('/')[1] if image_paths_relative else "N/A"
                
                # Skip if already evaluated
                if patient_id in evaluated_patients:
                    skipped_count += 1
                    continue

                if len(image_paths_relative) > max_images_per_patient:
                    image_paths_relative = image_paths_relative[:max_images_per_patient]

                image_full_paths = [os.path.join(dataset_root_dir, p) for p in image_paths_relative]
                
                if not all(os.path.exists(p) for p in image_full_paths):
                    print(f"\nWarning: Skipping patient {patient_id} due to one or more missing image slices.")
                    continue

                question = item['question']
                options = item['options']
                ground_truth_letter = item['answer']
                ground_truth_text = item['category']
                
                # Use CoT prompt
                prompt_text = format_prompt_for_vqa_with_cot(question, options)

                # --- Multi-image payload structure ---
                content = []
                for path in image_full_paths:
                    content.append({"type": "image", "image": path})
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]
                
                # Process inputs
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

                # Generate with more tokens for CoT reasoning
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text_list = processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                model_output = output_text_list[0] if output_text_list else ""

                # Parse answer
                is_correct = parse_model_answer(model_output, ground_truth_letter)
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                # Update counters
                gt_counts[ground_truth_text] += 1
                predicted_category = get_predicted_category(model_output, options)
                pred_counts[predicted_category] += 1

                # Log main results
                log_writer.writerow([
                    patient_id, str(image_paths_relative), question, str(options),
                    ground_truth_letter, ground_truth_text, model_output, is_correct
                ])

                # Extract and log CoT reasoning steps
                reasoning_steps = extract_reasoning_steps(model_output)
                cot_log_writer.writerow([
                    patient_id,
                    reasoning_steps["step1_tracer"],
                    reasoning_steps["step2_physiological"],
                    reasoning_steps["step3_quality"],
                    reasoning_steps["step4_abnormal"],
                    reasoning_steps["step5_disease"],
                    reasoning_steps["step6_final"],
                    is_correct
                ])

            except Exception as e:
                # Error handling
                p_id = "N/A"
                if 'image_paths' in item and item['image_paths']:
                    p_id = item['image_paths'][0].split('/')[1]
                print(f"\nError processing item for patient {p_id}: {e}")
                
                log_writer.writerow([
                    p_id, str(item.get('image_paths')), item.get('question'),
                    str(item.get('options')), item.get('answer'), item.get('category'),
                    f"ERROR: {e}", False
                ])
                cot_log_writer.writerow([p_id, "", "", "", "", "", "", False])
                
                gt_counts[item.get('category', 'Unknown')] += 1
                pred_counts["Processing Error"] += 1
                continue
    finally:
        log_file.close()
        cot_log_file.close()

    # --- 5. Calculate Final Accuracy ---
    print("\nStep 5/6: Calculating final results...")
    print(f"Skipped {skipped_count} already evaluated patients.")
    print(f"Newly evaluated: {total_predictions} patients.")
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    # --- 6. Write Summary Reports ---
    print("Step 6/6: Writing summary reports...")
    
    # Accuracy summary
    accuracy_file_path = os.path.join(output_dir, 'summary_accuracy_cot.txt')
    with open(accuracy_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Lingshu32B Chain-of-Thought (CoT) Reasoning Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        if resume and skipped_count > 0:
            f.write(f"Resume Mode: Skipped {skipped_count} already evaluated patients\n")
        f.write(f"Newly Evaluated Questions: {total_predictions}\n")
        f.write(f"Correct Predictions (New): {correct_predictions}\n")
        f.write(f"Accuracy (New Samples): {accuracy:.2f}%\n\n")
        f.write("CoT Reasoning Steps:\n")
        f.write("  1. Tracer Identification\n")
        f.write("  2. Physiological Uptake Reflection\n")
        f.write("  3. Image Quality Assessment\n")
        f.write("  4. Abnormal Uptake Detection\n")
        f.write("  5. Disease Reasoning\n")
        f.write("  6. Final Diagnosis\n")
    print(f"Accuracy summary saved to: {accuracy_file_path}")

    # Class distribution summary
    distribution_file_path = os.path.join(output_dir, 'summary_class_distribution_cot.txt')
    with open(distribution_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Ground Truth Class Distribution ---\n")
        for category, count in sorted(gt_counts.items()):
            f.write(f"{category}: {count}\n")
        f.write("\n--- Model Prediction Distribution ---\n")
        for category, count in sorted(pred_counts.items()):
            f.write(f"{category}: {count}\n")
    print(f"Class distribution summary saved to: {distribution_file_path}")

    # Final console output
    print("\n" + "=" * 60)
    print("--- Lingshu32B CoT Evaluation Complete ---")
    print("=" * 60)
    if resume and skipped_count > 0:
        print(f"Skipped (Already Done):    {skipped_count}")
    print(f"Newly Evaluated:           {total_predictions}")
    print(f"Correct Predictions (New): {correct_predictions}")
    print(f"Accuracy (New Samples):    {accuracy:.2f}%")
    print(f"\nDetailed logs saved to:")
    print(f"  - Main log:      {log_file_path}")
    print(f"  - CoT reasoning: {cot_log_file_path}")
    print("=" * 60 + "\n")


# =========================================================================
# == CONFIGURATION AND EXECUTION
# =========================================================================
if __name__ == "__main__":
    # ======================== [ACTION REQUIRED] ========================
    # Update these paths for your Lingshu32B model
    MODEL_PATH = "/mnt/hanxu/Xuhan-yzt/Medical-VLM-model/Lingshu7B"
    PROCESSOR_PATH = "/mnt/hanxu/Xuhan-yzt/Medical-VLM-model/Lingshu7B"
    # ===================================================================

    # Dataset paths
    JSON_FILE_PATH = "/mnt/hanxu/Xuhan-yzt/PET-benchmark-data/disease_diagnosis_select_15slices/PET_3Class_Sampled15_VQA/pet_3class_sampled15_vqa_randomized_improve_prompt.json"
    DATASET_ROOT_DIR = "/mnt/hanxu/Xuhan-yzt/PET-benchmark-data/disease_diagnosis_select_15slices/PET_3Class_Sampled15_VQA"

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_LOG_DIR = f"./eval_results/{script_name}_CoT"
    
    # Call the CoT evaluation function
    evaluate_lingshu_on_multi_slice_vqa_cot(
        model_path=MODEL_PATH,
        processor_path=PROCESSOR_PATH,
        json_path=JSON_FILE_PATH,
        dataset_root_dir=DATASET_ROOT_DIR,
        output_dir=OUTPUT_LOG_DIR,
        max_images_per_patient=15,
        max_new_tokens=1024,  # Increased for detailed CoT reasoning
        resume=True  # Enable resume mode (set to False to start fresh)
    )
