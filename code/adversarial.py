
from torch import nn
import foolbox


ATTACKS = {
    "fgsm": foolbox.attacks.FGSM,
    "iterative_fgsm": foolbox.attacks.IterativeGradientSignAttack,
    "deep_fool": foolbox.attacks.DeepFoolAttack,
    "saliency_map": foolbox.attacks.SaliencyMapAttack,
    "single_pixel": foolbox.attacks.SinglePixelAttack,
    "local_search": foolbox.attacks.LocalSearchAttack,
    "boundary_attack": foolbox.attacks.BoundaryAttack,
}

CRITERIA = {
    "untargeted_misclassify": foolbox.criteria.Misclassification(),
    "untargeted_top5_misclassify": foolbox.criteria.TopKMisclassification(5),
    "targeted_correct_class": foolbox.criteria.TargetClass,
}


def generate_adversarial_examples(model, attack, dataset, criterion, 
                                  pixel_bounds=(0, 255), num_classes=10, 
                                  channel_axis=1, cuda=True, preprocessing=None,
                                  target_class=None):
    if isinstance(model, nn.Module):
        preprocessing = (0, 1) if preprocessing is None else preprocessing
        model = foolbox.models.PyTorchModel(
            model, pixel_bounds, num_classes, cuda=cuda, 
            preprocessing=preprocessing)
    if isinstance(attack, str):
        attack = ATTACKS[attack.strip().lower()]
    if isinstance(criterion, str):
        criterion = CRITERIA[criterion.strip().lower()]
        if target_class is not None:
            criterion = criterion(target_class)

    attack = attack(model=model, criterion=criterion)
    return [attack(image, label) for image, label in dataset]


def adversarial_eval(model, adversarial_dataset):
    pass