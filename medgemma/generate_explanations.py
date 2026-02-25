
import json
import os
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse

def generate_explanations(image_dir, json_input_path, json_output_path, save_debug_images, debug_image_dir, model_id, num_samples=None):
    """
    Generates explanations for medical findings in images using a multimodal model.

    Args:
        image_dir (str): Path to the directory containing the images.
        json_input_path (str): Path to the input JSON file with findings.
        json_output_path (str): Path to save the output JSON file with explanations.
        save_debug_images (bool): Whether to save images with bounding boxes for debugging.
        debug_image_dir (str): Directory to save debug images.
        model_id (str): The model ID for the multimodal model.
        num_samples (int, optional): Number of samples to process. Defaults to all.
    """
    # --- Model and Processor Initialization ---
    print("Loading model and processor...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded.")

    # Create the directory for debug images if it doesn't exist
    if save_debug_images:
        os.makedirs(debug_image_dir, exist_ok=True)

    # Load the JSON data
    with open(json_input_path, 'r') as f:
        data = json.load(f)

    if num_samples is not None:
        data = data[:num_samples]
    
    new_data = []

    for item in tqdm(data, desc="Processing images"):
        if 'ImageID' not in item:
            new_data.append(item)
            continue

        image_id = item['ImageID']
        image_path = os.path.join(image_dir, image_id)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            item_copy = item.copy()
            if 'findings' in item_copy:
                for finding in item_copy['findings']:
                    finding['medgemma_explanation'] = "Image not found."
            new_data.append(item_copy)
            continue

        try:
            image = Image.open(image_path)
            width, height = image.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        new_item = item.copy()
        new_item['findings'] = []

        if 'findings' in item:
            for finding_index, finding in enumerate(item['findings']):
                new_finding = finding.copy()
                if 'boxes' in finding and finding['boxes']:
                    box = finding['boxes'][0]
                    
                    x_min = int(box[0] * width)
                    y_min = int(box[1] * height)
                    x_max = int(box[2] * width)
                    y_max = int(box[3] * height)

                    cropped_image = image.crop((x_min, y_min, x_max, y_max))
                    
                    overlaid_image = image.copy().convert("RGB")
                    draw = ImageDraw.Draw(overlaid_image)
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=10)

                    if save_debug_images:
                        finding_label_str = "_".join(finding.get('labels', ['finding'])).replace(" ", "_")
                        debug_image_path = os.path.join(debug_image_dir, f"{os.path.splitext(image_id)[0]}_{finding_index}_{finding_label_str}.png")
                        overlaid_image.save(debug_image_path)

                    finding_label = ", ".join(finding.get('labels', ['this finding']))

                    prompt_text = (
                        f"You are an expert radiologist analyzing a chest X-ray. "
                        f"An area of interest is marked with a red box in the first image, and the content of that box is shown in the second image. "
                        f"The finding identified in this section is '{finding_label}'. "
                        f"Please describe the key visual features that confirm this finding, looking at both the context in the full image and the details in the cropped image. When you describe the cropped image, please reference it as 'the bounding box' instead."
                        f"Your explanation should be concise and no more than two sentences."
                    )
                    
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image", "image": overlaid_image},
                            {"type": "image", "image": cropped_image}
                        ]}
                    ]

                    try:
                        inputs = processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=True,
                            return_dict=True, return_tensors="pt"
                        ).to(model.device, dtype=torch.bfloat16)

                        input_len = inputs["input_ids"].shape[-1]

                        with torch.inference_mode():
                            generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                            explanation = generation[0][input_len:]

                        decoded_explanation = processor.decode(explanation, skip_special_tokens=True)
                        new_finding['medgemma_explanation'] = decoded_explanation
                    except Exception as e:
                        print(f"Error during model inference for {image_id}: {e}")
                        new_finding['medgemma_explanation'] = "Error during model inference."
                else:
                    new_finding['medgemma_explanation'] = "No bounding box found."
                
                new_item['findings'].append(new_finding)

        new_data.append(new_item)

    with open(json_output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"Processing complete. Output saved to {json_output_path}")

if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_script_dir)

    parser = argparse.ArgumentParser(description="Generate explanations for medical findings in images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory where images are stored (e.g. local_sampled/).")
    parser.add_argument("--json_input_path", type=str, default=os.path.join(_project_dir, "data", "localize_small.json"), help="Path to the input JSON file.")
    parser.add_argument("--json_output_path", type=str, default=os.path.join(_project_dir, "data", "localize_small_with_explanations.json"), help="Path to save the output JSON file.")
    parser.add_argument("--save_debug_images", action='store_true', help="Save images with bounding boxes for debugging.")
    parser.add_argument("--debug_image_dir", type=str, default=os.path.join(_script_dir, "overlay_debug"), help="Directory to save debug images.")
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it", help="HuggingFace model ID or local path for the multimodal model.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process from the JSON file.")

    args = parser.parse_args()

    generate_explanations(
        args.image_dir,
        args.json_input_path,
        args.json_output_path,
        args.save_debug_images,
        args.debug_image_dir,
        args.model_id,
        args.num_samples
    )
