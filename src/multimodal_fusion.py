# === src/multimodal_fusion.py ===
import torch
import open_clip
from PIL import Image

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def perform_multimodal_fusion(image_paths, user_reports):
    embeddings = []
    for i, image_path in enumerate(image_paths):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            user_tokens = tokenizer(user_reports[i]).unsqueeze(0).to(device)
            text_features = clip_model.encode_text(user_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            fused = torch.cat([image_features, text_features], dim=-1)
            embeddings.append(fused.squeeze(0))

    return embeddings  # ✅ DON'T FORGET THIS
