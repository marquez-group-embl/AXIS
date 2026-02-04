import argparse
import os
from PIL import Image
import torch
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel, PeftConfig
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_token', dest='huggingface_token', type=str,
                    default=os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument('--vis_model_checkpoint', dest='vis_model_checkpoint', type=str)
    parser.add_argument('--uv_model_checkpoint', dest='uv_model_checkpoint', type=str)
    parser.add_argument('--vis_image_path', dest='vis_image_path', type=str, help="Path to the visible light input image")
    parser.add_argument('--uv_image_path', dest='uv_image_path', type=str, help="Path to the UV light input image")
    parser.add_argument('--image_size', dest='image_size', type=int,
                        default=518)
    parser.add_argument('--color_mode', dest='color_mode', type=str,
                        default="grayscale")
    return parser.parse_args()

def load_model(model_checkpoint, image_size, device):
    print(f'Loading model from checkpoint: {model_checkpoint}')
    
    # Load image processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        model_checkpoint,
        size={"height": image_size, "width": image_size}
    )
    
    # Load base model
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True,
    )
    
    try:
        print('Loading PEFT Model')
        config = PeftConfig.from_pretrained(model_checkpoint)
        model = PeftModel.from_pretrained(model, model_checkpoint)
        print("PEFT model loaded successfully")
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        print("Using base model without PEFT")
    
    model.to(device)
    model.eval()
    return model, image_processor

def preprocess_image(image_path, image_processor, color_mode, image_size, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    
    # Define transforms
    load_image = [v2.ToImage(), v2.Grayscale(3)] if color_mode == 'grayscale' else [v2.ToImage()]
    transforms = v2.Compose(
        load_image +
        [
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    
    # Apply transforms
    image = image.convert("RGB")
    image_tensor = transforms(image)
    return image_tensor.unsqueeze(0).to(device)

def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Login to Hugging Face
    login(token=args.huggingface_token)
    vis_crystal_prob=None
    uv_crystal_prob=None
    if args.vis_model_checkpoint and args.vis_image_path:
        # Load model and processor
        model, image_processor = load_model(args.vis_model_checkpoint, args.image_size, device)
    
        # Preprocess image
        inputs = preprocess_image(args.vis_image_path, image_processor, args.color_mode, args.image_size, device)
    
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
            vis_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            vis_predictions = torch.argmax(outputs.logits, dim=-1)
    
            # Print only crystal probability (label 0)
            vis_crystal_prob = vis_probabilities[0][0].item()

    if args.uv_model_checkpoint and args.uv_image_path:
        # Load model and processor
        model, image_processor = load_model(args.uv_model_checkpoint, args.image_size, device)
    
        # Preprocess image
        inputs = preprocess_image(args.uv_image_path, image_processor, args.color_mode, args.image_size, device)
    
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
            uv_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            uv_predictions = torch.argmax(outputs.logits, dim=-1)
    
            # Print only crystal probability (label 0)
            uv_crystal_prob = uv_probabilities[0][0].item()
    if vis_crystal_prob and not uv_crystal_prob:
        print(f"Visible light crystal probability: {vis_crystal_prob:.2%}")
        result = vis_crystal_prob
    elif not vis_crystal_prob and uv_crystal_prob:
        print(f"UV light crystal probability: {uv_crystal_prob:.2%}")
        result = uv_crystal_prob
    elif vis_crystal_prob and uv_crystal_prob:
        combined_prob = max(vis_crystal_prob, uv_crystal_prob)
        print(f"Visible light crystal probability: {vis_crystal_prob:.2%}")
        print(f"UV light crystal probability: {uv_crystal_prob:.2%}")
        print(f"Argmax crystal probability: {combined_prob:.2%}")
        result = combined_prob
    return result
if __name__ == "__main__":
    main()
