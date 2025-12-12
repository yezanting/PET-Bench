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
# == CORE EVALUATION LOGIC (WITH ROBUST MULTI-LABEL PROMPT)
# =========================================================================

def format_prompt_for_multilabel_vqa(question: str, options: list) -> str:
    """
    Formats the question and options for a multi-label task with a robust prompt
    that does not assume option order.
    """
    # This new prompt describes the task without mentioning specific letters for each category.
    # It instructs the model on the reasoning process it should follow.
    prompt = (f"Assess this PET image based on two separate criteria: 1. Overall image quality. 2. Presence of artifacts.\n\n"
              f"From the options below, select the two letters that best describe your complete assessment, "
              f"choosing one for each criterion.\n\n"
              "Options:\n")
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        prompt += f"{letter}. {option_text}\n"
    prompt += "\nPlease provide the letters of the two most appropriate options."
    return prompt

def parse_model_answer_multilabel(model_output: str, correct_letters: list) -> bool:
    """
    Parses the model's output for a multi-label task.
    Returns True ONLY if ALL correct letters are found.
    """
    return all(re.search(f"\\b{letter}\\b", model_output, re.IGNORECASE) for letter in correct_letters)

def get_predicted_categories_multilabel(model_output: str, options: list) -> list:
    """
    Parses the model's output to find all categories it chose in a multi-label context.
    Returns a list of the chosen option texts.
    """
    predicted_categories = []
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        if re.search(f"\\b{letter}\\b", model_output, re.IGNORECASE):
            predicted_categories.append(option_text)
    if not predicted_categories:
        return ["Unknown/No Answer"]
    return predicted_categories

def evaluate_lingshu_on_vqa(
        model_path: str,
        processor_path: str,
        json_path: str,
        dataset_root_dir: str,
        output_dir: str
):
    """
    Main function to evaluate the Lingshu model on a multi-label VQA dataset.
    """
    print("--- Starting Lingshu Multi-Label VQA Evaluation (Robust Prompt v2.1) ---")

    # --- 1. Setup Model and Processor ---
    print(f"Step 1/6: Loading model from '{model_path}'...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
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

    # --- 3. Setup Logging ---
    print(f"Step 3/6: Setting up output directory and log file...")
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'evaluation_log.csv')
    log_file = open(log_file_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "image_path", "question", "options", "ground_truth_letters",
        "ground_truth_texts", "model_raw_output", "is_correct"
    ])

    correct_predictions, total_predictions = 0, 0
    gt_counts, pred_counts = Counter(), Counter()

    # --- 4. Run Evaluation Loop ---
    print("Step 4/6: Running inference on all QA pairs...")
    try:
        for item in tqdm(qa_data, desc="Evaluating"):
            try:
                image_relative_path = item['image_path']
                image_full_path = os.path.join(dataset_root_dir, image_relative_path)
                if not os.path.exists(image_full_path):
                    continue

                question, options, ground_truth_letters = item['question'], item['options'], item['answer']
                ground_truth_texts = [options[ord(letter) - ord('A')] for letter in ground_truth_letters]
                prompt_text = format_prompt_for_multilabel_vqa(question, options)
                messages = [{"role": "user", "content": [{"type": "image", "image": image_full_path}, {"type": "text", "text": prompt_text}]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text_list = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                model_output = output_text_list[0] if output_text_list else ""

                is_correct = parse_model_answer_multilabel(model_output, ground_truth_letters)
                if is_correct: correct_predictions += 1
                total_predictions += 1

                gt_counts.update(ground_truth_texts)
                predicted_categories = get_predicted_categories_multilabel(model_output, options)
                pred_counts.update(predicted_categories)

                log_writer.writerow([image_relative_path, question, str(options), str(ground_truth_letters), str(ground_truth_texts), model_output, is_correct])

            except Exception as e:
                print(f"\nError processing item {item.get('image_path', 'N/A')}: {e}")
                log_writer.writerow([item.get('image_path', 'N/A'), item.get('question', 'N/A'), str(item.get('options', 'N/A')), str(item.get('answer', 'N/A')), "ERROR", f"ERROR: {e}", False])
                gt_counts.update([options[ord(letter) - ord('A')] for letter in item.get('answer', [])])
                pred_counts["Processing Error"] += 1
                continue
    finally:
        log_file.close()

    # --- 5. Calculate Final Accuracy ---
    print("\nStep 5/6: Calculating final results...")
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    # --- 6. Write Summary Files ---
    print("Step 6/6: Writing summary reports...")
    accuracy_file_path = os.path.join(output_dir, 'summary_accuracy.txt')
    with open(accuracy_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Questions Evaluated: {total_predictions}\n")
        f.write(f"Correct Predictions (all labels must match): {correct_predictions}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    print(f"Accuracy summary saved to: {accuracy_file_path}")

    distribution_file_path = os.path.join(output_dir, 'summary_class_distribution.txt')
    with open(distribution_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Ground Truth Class Distribution ---\n")
        for category, count in sorted(gt_counts.items()): f.write(f"{category}: {count}\n")
        f.write("\n--- Model Prediction Distribution ---\n")
        for category, count in sorted(pred_counts.items()): f.write(f"{category}: {count}\n")
    print(f"Class distribution summary saved to: {distribution_file_path}")

    print("\n--- Evaluation Complete ---")
    print(f"Total Questions Evaluated: {total_predictions}")
    print(f"Correct Predictions:       {correct_predictions}")
    print(f"Accuracy:                  {accuracy:.2f}%")
    print(f"Detailed CSV log saved to: {log_file_path}")
    print("---------------------------\n")

# =========================================================================
# == CONFIGURATION AND EXECUTION
# =========================================================================
if __name__ == "__main__":
    MODEL_PATH = "/mnt/Xuhan-yzt/Medical-VLM-model/Lingshu7B"
    PROCESSOR_PATH = "/mnt/Xuhan-yzt/Medical-VLM-model/Lingshu7B"
    JSON_FILE_PATH = "/mnt/Xuhan-yzt/PET-benchmark-data/PET-image_quality/PET_Image_Quality_QA_EN_dedup_v2/pet_image_quality_qa_en_dedup_v2.json"
    DATASET_ROOT_DIR = "/mnt/Xuhan-yzt/PET-benchmark-data/PET-image_quality/PET_Image_Quality_QA_EN_dedup_v2"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_LOG_DIR = f"/mnt/Xuhan-yzt/VLMEvalKit-main/eval_results/{script_name}_results"

    evaluate_lingshu_on_vqa(
        model_path=MODEL_PATH,
        processor_path=PROCESSOR_PATH,
        json_path=JSON_FILE_PATH,
        dataset_root_dir=DATASET_ROOT_DIR,
        output_dir=OUTPUT_LOG_DIR
    )