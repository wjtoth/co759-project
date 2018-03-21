
import numpy as np
import foolbox
from torch import nn


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
    examples = []
    for image, label in dataset:
        examples.append((attack(image, label, epsilons=[.1]), label))
    return examples


def adversarial_eval(model, adversarial_dataset, criterion="untargeted_misclassify", 
                     batch_size=64, target_class=None):
    failures = []
    for i in range(len(adversarial_dataset) // batch_size):
        examples = adversarial_dataset[i*batch_size:(i+1)*batch_size]
        images = [example[0] for example in examples]
        labels = [example[1] for example in examples]
        if images[0].ndim == 4:
            images = np.concat(images)
        elif images[0].ndim == 3:
            images = np.concat([image[None,:] for image in images])
        else:
            raise ValueError("Can't handle an adversarial example of that shape!")
        logits = model.batch_predictions(images)
        predictions = np.argmax(logits, axis=1)
        if criterion == "untargeted_misclassify":
            failures.extend(
                [predictions[j] != labels[j] for j in range(batch_size)])
        elif criterion == "untargeted_top5_misclassify":
            predictions = np.argpartition(logits, -5)[-5:]
            failures.extend(
                [labels[j] not in predictions[j] for j in range(batch_size)])
        elif criterion == "targeted_correct_class":
            failures.extend(
                [predictions[j] == target_class for j in range(batch_size)])
        else:
            raise ValueError("Unsupported criterion!")

    return failures / (batch_size * (len(adversarial_dataset)//batch_size))
