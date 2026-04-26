# =============================================================================
# model_factory.py — Mapping modèles → poids pré-entraînés
#
# Rôle : centraliser la correspondance entre le nom d'un modèle (ex: "vgg19")
# et ses poids officiels torchvision, pour que transforms_factory.py puisse
# récupérer le pipeline de préprocessing correct (resize, normalize, etc.)
# sans hardcoder les paramètres ImageNet partout.
#
# Extensible : pour ajouter ResNet50 par exemple, il suffit d'ajouter :
#   from torchvision.models import ResNet50_Weights
#   MODEL_WEIGHTS_MAPPING["resnet50"] = ResNet50_Weights
#
# Utilisé dans transforms_factory.py :
#   weights = MODEL_WEIGHTS_MAPPING["vgg19"].DEFAULT
#   model_preprocess = weights.transforms()  → Resize(256) + CenterCrop(224) + Normalize
# =============================================================================

from torchvision.models import VGG19_Weights

# Dictionnaire nom_modèle → classe de poids torchvision
MODEL_WEIGHTS_MAPPING = {"vgg19": VGG19_Weights}
