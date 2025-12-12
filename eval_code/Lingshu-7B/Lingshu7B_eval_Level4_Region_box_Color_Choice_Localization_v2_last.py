import os
import json
import csv
import re
import torch
from tqdm import tqdm
from collections import Counter  # Import Counter for easy counting
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Assuming this is a utility from the model's repository


# =========================================================================
# == CORE EVALUATION LOGIC
# =========================================================================

def format_prompt_for_vqa(question: str, options: list) -> str:
    """
    Formats the question and options into a clear, structured prompt for the VLM.
    """
    prompt = f"{question}\n\nOptions:\n"
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        prompt += f"{letter}. {option_text}\n"
    prompt += "\nPlease provide the letter of the correct option.Your answer should only contain the letters of the selected options."
    return prompt


def parse_model_answer(model_output: str, correct_letter: str, options: list) -> bool:
    """
    Parses the model's text output to check for correctness using robust regex.
    """
    if re.search(f"\\b{correct_letter}\\b", model_output, re.IGNORECASE):
        return True
    return False


# --- NEW HELPER FUNCTION ---
def get_predicted_category(model_output: str, options: list) -> str:
    """
    Parses the model's output to determine which category it chose.
    Returns the full text of the chosen option.
    """
    for i, option_text in enumerate(options):
        letter = chr(ord('A') + i)
        # Check if the model's output contains this option's letter as a whole word
        if re.search(f"\\b{letter}\\b", model_output, re.IGNORECASE):
            return option_text  # Return the corresponding category text
    return "Unknown/No Answer"  # Fallback if no valid option is found


def evaluate_lingshu_on_vqa(
        model_path: str,
        processor_path: str,
        json_path: str,
        dataset_root_dir: str,
        output_dir: str
):
    """
    Main function to evaluate the Lingshu model on a VQA dataset.
    """
    print("--- Starting Lingshu VQA Evaluation ---")

    # --- 1. Setup Model and Processor ---
    print(f"Step 1/6: Loading model from '{model_path}'...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
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
        "image_path", "question", "options", "ground_truth_letter",
        "ground_truth_text", "model_raw_output", "is_correct"
    ])

    correct_predictions = 0
    total_predictions = 0
    # --- MODIFICATION: Initialize counters ---
    gt_counts = Counter()
    pred_counts = Counter()

    # --- 4. Run Evaluation Loop ---
    print("Step 4/6: Running inference on all QA pairs...")
    try:
        for item in tqdm(qa_data, desc="Evaluating"):
            try:
                image_relative_path = item['image_path']
                image_full_path = os.path.join(dataset_root_dir, image_relative_path)

                if not os.path.exists(image_full_path):
                    print(f"Warning: Image not found, skipping. Path: {image_full_path}")
                    continue

                question, options, ground_truth_letter, ground_truth_text = item['question'], item['options'], item[
                    'answer'], item['category']
                prompt_text = format_prompt_for_vqa(question, options)
                messages = [{"role": "user", "content": [{"type": "image", "image": image_full_path},
                                                         {"type": "text", "text": prompt_text}]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                   return_tensors="pt").to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                         zip(inputs.input_ids, generated_ids)]
                output_text_list = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=False)
                model_output = output_text_list[0] if output_text_list else ""

                is_correct = parse_model_answer(model_output, ground_truth_letter, options)

                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                # --- MODIFICATION: Update counters ---
                gt_counts[ground_truth_text] += 1
                predicted_category = get_predicted_category(model_output, options)
                pred_counts[predicted_category] += 1

                log_writer.writerow(
                    [image_relative_path, question, str(options), ground_truth_letter, ground_truth_text, model_output,
                     is_correct])

            except Exception as e:
                print(f"\nError processing item {item.get('image_path', 'N/A')}: {e}")
                log_writer.writerow(
                    [item.get('image_path', 'N/A'), item.get('question', 'N/A'), str(item.get('options', 'N/A')),
                     item.get('answer', 'N/A'), item.get('category', 'N/A'), f"ERROR: {e}", False])
                # Also count this as an error in predictions
                gt_counts[item.get('category', 'Unknown')] += 1
                pred_counts["Processing Error"] += 1
                continue
    finally:
        log_file.close()

    # --- 5. Calculate Final Accuracy ---
    print("\nStep 5/6: Calculating final results...")
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    # --- 6. Write Summary Files ---
    print("Step 6/6: Writing summary reports...")

    # Write accuracy to txt file
    accuracy_file_path = os.path.join(output_dir, 'summary_accuracy.txt')
    with open(accuracy_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Questions Evaluated: {total_predictions}\n")
        f.write(f"Correct Predictions: {correct_predictions}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    print(f"Accuracy summary saved to: {accuracy_file_path}")

    # Write class distribution to txt file
    distribution_file_path = os.path.join(output_dir, 'summary_class_distribution.txt')
    with open(distribution_file_path, 'w', encoding='utf-8') as f:
        f.write("--- Ground Truth Class Distribution ---\n")
        for category, count in sorted(gt_counts.items()):
            f.write(f"{category}: {count}\n")

        f.write("\n--- Model Prediction Distribution ---\n")
        for category, count in sorted(pred_counts.items()):
            f.write(f"{category}: {count}\n")
    print(f"Class distribution summary saved to: {distribution_file_path}")

    # --- Final Console Output ---
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
    JSON_FILE_PATH = "/mnt/Xuhan-yzt/PET-benchmark-data/Level4_Region_box_Color_Choice_Localization_v2_last/PET_Region_Color_Choice_VQA_v4/pet_region_color_choice_vqa_v4.json"
    DATASET_ROOT_DIR = "/mnt/Xuhan-yzt/PET-benchmark-data/Level4_Region_box_Color_Choice_Localization_v2_last/PET_Region_Color_Choice_VQA_v4"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    OUTPUT_LOG_DIR = f"/mnt/Xuhan-yzt/VLMEvalKit-main/eval_results/{script_name}_results"

    if "/path/to/your" in JSON_FILE_PATH or "/path/to/your" in  DATASET_ROOT_DIR:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Please update the file paths in the script before running. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        evaluate_lingshu_on_vqa(
            model_path=MODEL_PATH,
            processor_path=PROCESSOR_PATH,
            json_path=JSON_FILE_PATH,
            dataset_root_dir=DATASET_ROOT_DIR,
            output_dir=OUTPUT_LOG_DIR
        )