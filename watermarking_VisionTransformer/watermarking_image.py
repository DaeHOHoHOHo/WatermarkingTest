import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

image_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.batch_norm = nn.BatchNorm1d(self.num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(p=0.2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, patch_size * patch_size * 3)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.batch_norm(x)
        x = self.dropout(x + self.pos_embed)
        x = self.transformer(x)
        x_cls = x.mean(dim=1)
        logits = self.classifier(x_cls)
        x_recon = self.decoder(x)
        x_recon = x_recon.transpose(1, 2).reshape(-1, 3, image_size, image_size)
        return logits, x_recon

model = ViT(image_size=image_size, patch_size=4, num_classes=10, embed_dim=64, num_heads=8, num_layers=11)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

def add_watermark(image, watermark, alpha=0.1):
    return image * (1 - alpha) + watermark * alpha

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def watermark_image(model, image_path):
    image = process_image(image_path)
    
    watermark = torch.randn(1, 3, image_size, image_size).to(device) * 0.1
    
    watermarked_image = add_watermark(image, watermark)

    with torch.no_grad():
        _, recon_image = model(watermarked_image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.cpu().squeeze().permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title("Original")
    axes[1].imshow(watermarked_image.cpu().squeeze().permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("Watermarked")
    axes[2].imshow(recon_image.cpu().squeeze().permute(1, 2, 0).clamp(0, 1))
    axes[2].set_title("Reconstructed by Model")
    for ax in axes:
        ax.axis("off")
    plt.show()

image_path = 'path_to_your_image.jpg'
watermark_image(model, image_path)
