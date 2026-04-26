# =============================================================================
# transforms_factory.py — Pipeline de préprocessing des images
#
# Rôle : préparer les images brutes pour qu'elles soient compatibles avec
# VGG19 pré-entraîné sur ImageNet (taille 224×224, normalisation ImageNet).
#
# Deux pipelines distincts :
#   - train : avec augmentation (flip, couleur, grayscale) pour réduire l'overfitting
#   - val/test : normalisation uniquement (pas d'augmentation au test)
#
# FIX appliqué :
#   Ajout de v2.ToImage() en tête des deux pipelines.
#   CAUSE : AnnotationDataset charge les images via numpy (np.asarray + PIL),
#   mais les transforms v2 de torchvision attendent un PIL.Image ou un tensor,
#   pas un numpy array directement.
#   ERREUR levée sans ce fix :
#   "TypeError: Unexpected type <class 'numpy.ndarray'>"
#   SOLUTION : v2.ToImage() convertit numpy array → torchvision Image tensor
#   avant d'appliquer les transforms suivantes.
# =============================================================================

import torchvision.transforms.v2 as v2
from src.dataset.model_factory import MODEL_WEIGHTS_MAPPING


def get_transforms(model):
    """Retourne les pipelines de transforms train et val pour le modèle donné.

    Args:
        model : nom du modèle ("vgg19") — clé dans MODEL_WEIGHTS_MAPPING

    Returns:
        (train_transforms, val_transforms) — tuple de v2.Compose
    """
    # Récupère les transforms officiels du modèle (resize, crop, normalize ImageNet)
    weights = MODEL_WEIGHTS_MAPPING[model].DEFAULT
    model_preprocess = weights.transforms()  # Resize(256) + CenterCrop(224) + Normalize

    train_transforms = v2.Compose(
        [
            # FIX : convertit numpy array → tensor torchvision (nécessaire car
            # AnnotationDataset retourne np.asarray(PIL.Image))
            v2.ToImage(),

            v2.RandomHorizontalFlip(p=0.5),                              # augmentation spatiale
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # variation de couleur
            v2.RandomGrayscale(p=0.05),                                   # robustesse aux images N&B

            model_preprocess,  # Resize + CenterCrop(224) + Normalize(ImageNet stats)
        ]
    )

    # Val/test : pas d'augmentation — on veut une évaluation reproductible et stable
    test_transforms = v2.Compose([v2.ToImage(), model_preprocess])

    return train_transforms, test_transforms
