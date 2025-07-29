import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
import torch.nn.functional as F
from in_progress.dataset import classifier_transforms

# It's assumed 'classifier_transforms' is defined as it was in your training dataset file.
# If not, you must redefine it here. We are recreating it based on standard practices.


# Define the class names
CLASS_NAMES = ['axion', 'cdm', 'no_sub'] # Modify these to your actual class names

def predict(npy_path, model_path, device):
    """
    Loads a model and predicts the class of a single .npy file.

    Args:
        npy_path (str): The path to the .npy file.
        model_path (str): The path to the saved model checkpoint (.pt file).
        device (str): The device to run inference on ('cuda' or 'cpu').
    """
    # 1. Load the model architecture
    model = resnet18(weights=None) # We don't need pretrained weights for inference, just the architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    # 2. Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        state_dict = checkpoint['model_state_dict'] 
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")  # just that one simple trick
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    model.to(device)
    model.eval()

    # 3. Load and preprocess the input .npy file
    try:
        # Assuming the .npy file contains a NumPy array that can be converted to an image
        # The shape should be (H, W, C) or (C, H, W)
        image_data = np.load(npy_path)
        print(f"Loaded .npy file from: {npy_path} with shape: {image_data.shape}")
        
        # Apply the same transforms used during training
        input_tensor = classifier_transforms(image_data).float()
        
        # Add a batch dimension (B, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)

    except FileNotFoundError:
        print(f"Error: Input .npy file not found at {npy_path}")
        return
    except Exception as e:
        print(f"An error occurred while processing the .npy file: {e}")
        return

    # 4. Perform inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        
        # Apply Softmax to get probabilities
        probabilities = F.softmax(output, dim=1)
        
        # Get the top prediction
        top_prob, top_catid = torch.max(probabilities, 1)
        predicted_class_index = top_catid.item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = top_prob.item()

    # 5. Output the result
    print("\n--- Prediction Result ---")
    print(f"Predicted Class: '{predicted_class_name}' (Index: {predicted_class_index})")
    print(f"Confidence: {confidence:.4f}")
    print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ResNet18 classifier.")
    parser.add_argument(
        '--npy_path', 
        type=str, 
        required=True,
        help='Path to the input .npy file.'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='/speech/advait/rooshil/nanoDiT/in_progress/classifier_training_resnet18/classifier_ckpts/resnet18_best.pt',
        help='Path to the trained model checkpoint (.pt file).'
    )
    
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predict(args.npy_path, args.model_path, device)
