from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class VinaFood21(Dataset):
    def __init__(self, root: str, class_to_idx=None):
        root = Path(root)
        if class_to_idx is None:
            classes = sorted(
                folder.name
                for folder in root.iterdir()
                if folder.is_dir()
            )
            class_to_idx = {
                name: idx for idx, name in enumerate(classes)
            }
        self.class_to_idx = class_to_idx
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), #(224,224)
            transforms.ToTensor(), #/255
        ])
        self._data = []
        for class_name, label in self.class_to_idx.items():
            folder = root / class_name
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self._data.append({
                        "image_path": image_path,
                        "label": label,
                    })

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]
        with Image.open(item["image_path"]) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return {
            "image": image,          # (3, 224, 224)
            "label": item["label"],   # nhãn (0->20)
        }
