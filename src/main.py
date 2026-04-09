from dataset import AnnotationDataset
import matplotlib.pyplot as plt



if __name__=="__main__":
    ds = AnnotationDataset("data/flicker8k", split_type="train")
    img, label = next(iter(ds))
    print(f"caption : {ds.vocab.decode(label)}")
    plt.imshow(img)
    plt.show()
