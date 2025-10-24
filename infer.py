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
    parser.add_argument('--model_checkpoint', dest='model_checkpoint', type=str,
                        default="apersonnaz/AXIS-foundation")
    parser.add_argument('--image_path', dest='image_path', type=str,
                        required=True, help="Path to the input image")
    parser.add_argument('--image_size', dest='image_size', type=int,
                        default=518)
    parser.add_argument('--color_mode', dest='color_mode', type=str,
                        default="grayscale")
    return parser.parse_args()

def load_model(args, device):
    print(os.getenv("HUGGINGFACE_TOKEN"))
    print(args)
    # Login to Hugging Face
    login(token=args.huggingface_token)
    # Load image processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_checkpoint,
        size={"height": args.image_size, "width": args.image_size}
    )
    
    # Load base model
    model = AutoModelForImageClassification.from_pretrained(
        args.model_checkpoint,
        ignore_mismatched_sizes=True,
    )
    
    try:
        print('Loading PEFT Model')
        config = PeftConfig.from_pretrained(args.model_checkpoint)
        model = PeftModel.from_pretrained(model, args.model_checkpoint)
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
    
    # Load model and processor
    model, image_processor = load_model(args, device)
    
    # Preprocess image
    inputs = preprocess_image(args.image_path, image_processor, args.color_mode, args.image_size, device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Print only crystal probability (label 0)
    crystal_prob = probabilities[0][0].item()
    print(f"CRYSTAL probability: {crystal_prob:.2%}")

if __name__ == "__main__":
    main()
