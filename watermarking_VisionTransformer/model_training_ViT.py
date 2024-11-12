import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

image_size = 32
patch_size = 4
embed_dim = 64
num_heads = 8
num_layers = 11
num_classes = 10
batch_size = 128
num_epochs = 2000
learning_rate = 0.059

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

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

def add_watermark(images, watermark, alpha=0.1):
    watermarked_images = images * (1 - alpha) + watermark * alpha
    return watermarked_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(image_size=image_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

model.apply(init_weights)

criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_recon = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

watermark = torch.randn(1, 3, image_size, image_size).to(device) * 0.1

best_val_accuracy = 0
patience = 2000
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        watermarked_inputs = add_watermark(inputs, watermark)

        optimizer.zero_grad()
        outputs, recon_images = model(watermarked_inputs)
        loss_cls = criterion_cls(outputs, labels)
        loss_recon = criterion_recon(recon_images, inputs)
        loss = loss_cls + 0.015 * loss_recon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    validation_accuracy = 100 * correct / total
    print('Epoch %d validation accuracy: %.2f%%' % (epoch + 1, validation_accuracy))
    if validation_accuracy > best_val_accuracy:
        best_val_accuracy = validation_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping")
            break

print('Finished Training')

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print('Test accuracy: %.2f%%' % test_accuracy)

with torch.no_grad():
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images = images.to(device)
    
    watermarked_images = add_watermark(images, watermark)
    
    _, recon_images = model(watermarked_images)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    for idx in range(3):
        plt.subplot(3, 3, idx + 1)
        plt.imshow(images[idx].cpu().permute(1, 2, 0))
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(3, 3, idx + 4)
        plt.imshow(watermarked_images[idx].cpu().permute(1, 2, 0))
        plt.title("Watermarked")
        plt.axis('off')
        
        plt.subplot(3, 3, idx + 7)
        plt.imshow(recon_images[idx].cpu().permute(1, 2, 0))
        plt.title("Reconstructed")
        plt.axis('off')
    plt.show()
