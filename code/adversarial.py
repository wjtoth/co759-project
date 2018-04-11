
import numpy as np
from torch import nn

try:
    import foolbox

    ATTACKS = {
    "fgsm": {"class": foolbox.attacks.FGSM, 
             "kwargs": {}},
    "iterative_fgsm": {"class": foolbox.attacks.IterativeGradientSignAttack,
                       "kwargs": {"steps": 10}},
    "deep_fool": {"class": foolbox.attacks.DeepFoolAttack, 
                  "kwargs": {"steps": 25, "subsample": 10}},
    "saliency_map": {"class": foolbox.attacks.SaliencyMapAttack,
                     "kwargs": {"max_iter": 200}},
    "single_pixel": {"class": foolbox.attacks.SinglePixelAttack,
                     "kwargs": {"max_pixels": 200}},
    "local_search": {"class": foolbox.attacks.LocalSearchAttack,
                     "kwargs": {"R": 25}},
    "boundary_attack": {"class": foolbox.attacks.BoundaryAttack,
                        "kwargs": {"iterations": 50, "log_every_n_steps": 500}},
    }

    CRITERIA = {
        "untargeted_misclassify": foolbox.criteria.Misclassification(),
        "untargeted_top5_misclassify": foolbox.criteria.TopKMisclassification(5),
        "targeted_correct_class": foolbox.criteria.TargetClass,
    }

except ImportError:
    ATTACKS = {"error foolbox not available"}
    print('foolbox was not loaded...')

def generate_adversarial_examples(model, attack, dataset, criterion, 
                                  pixel_bounds=(0, 255), num_classes=10, 
                                  channel_axis=1, cuda=True, preprocessing=None,
                                  target_class=None, epsilon=.25):
    if isinstance(model, nn.Module):
        preprocessing = (0, 1) if preprocessing is None else preprocessing
        model = foolbox.models.PyTorchModel(
            model, pixel_bounds, num_classes, cuda=cuda, 
            preprocessing=preprocessing)
    if isinstance(attack, str):
        attack_string = attack.strip().lower()
        attack = ATTACKS[attack_string]
        attack_kwargs = attack["kwargs"]
        attack = attack["class"]
        if "fgsm" in attack_string:
            attack_kwargs["epsilons"] = [epsilon]
    if isinstance(criterion, str):
        criterion = CRITERIA[criterion.strip().lower()]
        if target_class is not None:
            criterion = criterion(target_class)

    attack = attack(model=model, criterion=criterion)
    examples = []
    for image, label in dataset:
        adversarial_input = attack(image, int(label), **attack_kwargs)
        if adversarial_input is None:
            # Misclassified input or adversarial example not found
            examples.append(None)
        else:
            examples.append((adversarial_input, image, label))

    return examples, model


def adversarial_eval(foolbox_model, adversarial_dataset, 
                     criterion="untargeted_misclassify", 
                     batch_size=64, target_class=None, 
                     against_labels=False):
    failures = []
    adv_count = len([example for example in adversarial_dataset 
                     if example is not None])
    for i in range(len(adversarial_dataset) // batch_size):
        examples = adversarial_dataset[i*batch_size:(i+1)*batch_size]
        # Misclassified inputs / attack failures are ignored
        examples = [example for example in examples if example is not None]
        if not examples:
            continue
        adv_images = [example[0] for example in examples]
        orig_images = [example[1] for example in examples]
        labels = [example[2] for example in examples]
        if adv_images[0].ndim == 4:
            adv_images = np.concatenate(adv_images)
            orig_images = np.concatenate(orig_images)
        elif adv_images[0].ndim == 3:
            adv_images = np.concatenate([image[None,:] for image in adv_images])
            orig_images = np.concatenate([image[None,:] for image in orig_images])
        else:
            raise ValueError("Can't handle an adversarial example of that shape!")
        adv_logits = foolbox_model.batch_predictions(adv_images)
        orig_logits = foolbox_model.batch_predictions(orig_images)
        adv_predictions = np.argmax(adv_logits, axis=1)
        orig_predictions = np.argmax(orig_logits, axis=1)
        if against_labels:
            good_values = labels
        else:
            good_values = orig_predictions
        if criterion == "untargeted_misclassify":
            failures.extend([int(adv_predictions[j]) != int(good_values[j]) 
                             for j in range(batch_size)])
        elif criterion == "untargeted_top5_misclassify":
            adv_predictions = np.argpartition(logits, -5)[-5:]
            failures.extend([int(good_values[j]) not in adv_predictions[j] 
                             for j in range(batch_size)])
        elif criterion == "targeted_correct_class":
            failures.extend([int(adv_predictions[j]) == target_class 
                             for j in range(batch_size)])
        else:
            raise ValueError("Unsupported criterion!")

    print("Failures:", failures.count(True))
    print("Adversarial examples:", adv_count)
    return failures.count(True) / adv_count
