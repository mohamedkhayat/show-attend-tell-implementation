from src.dataset.AnnotationDataset import AnnotationDataset
import matplotlib.pyplot as plt
from src.dataset.transforms_factory import get_transforms

if __name__ == "__main__":
    ds = AnnotationDataset("data/flicker8k", split_type="train")
    img, label = next(iter(ds))
    print(f"caption : {ds.vocab.decode(label)}")
    plt.imshow(img)
    plt.show()
    transforms = get_transforms("vgg19")
    print(transforms)
