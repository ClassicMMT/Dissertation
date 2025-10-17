"""
This file contains the implementation of the proposed method.
"""


def compute_threshold(trained_model, calib_loader, uncertainty_measure, alpha, verbose=False, device="mps"):
    """
    This is a PyTorch implementation of Algorithm 2.

    Args:
        trained_model: PyTorch model
        calib_loader: PyTorch DataLoader with calibration data
        uncertainty_measure: one of ["entropy", "information content", "probability gap"]
        alpha: quantile level (alpha=0.05 => 95% quantile)

    Returns: lambda
    """
    import torch
    from src.utils import calculate_probability_gap, calculate_entropy, calculate_information_content

    assert uncertainty_measure in (
        "entropy",
        "information content",
        "probability gap",
    ), 'uncertainty_measure must be in ["entropy", "information content", "probability gap"]'

    with torch.no_grad():
        trained_model.eval()

        uncertainties = []

        for i, (features, _) in enumerate(calib_loader):
            if verbose:
                print(f"Calibration Batch: {i+1}/{len(calib_loader)}")
            features = features.to(device)

            # Get logits
            logits = trained_model(features)

            # calculate uncertainty
            if uncertainty_measure == "entropy":
                uncertainty = calculate_entropy(logits, apply_softmax=True)
            elif uncertainty_measure == "information content":
                uncertainty = calculate_information_content(logits, apply_softmax=True)
            elif uncertainty_measure == "probability gap":
                uncertainty = calculate_probability_gap(logits, apply_softmax=True)

            # save
            uncertainties.append(uncertainty)

    # combine
    uncertainties = torch.cat(uncertainties)

    # compute thresholds
    if uncertainty_measure in ("entropy", "information content"):
        lambda_ = torch.quantile(uncertainties, 1 - alpha, interpolation="higher")
    else:
        lambda_ = torch.quantile(uncertainties, alpha, interpolation="lower")

    return lambda_


def identify_uncertain_points(
    trained_model, calibration_loader, test_loader, uncertainty_measure, alpha, verbose=False, device="mps"
):
    """
    This is a PyTorch implementation of Algorithm 1.

    Args:
        trained_model: PyTorch model
        test_loader: PyTorch DataLoader with test data
        alpha: quantile level (alpha=0.05 => 95% quantile)

    Returns: a boolean tensor of whether points are highly uncertain
    """
    import torch
    from src.utils import calculate_entropy, calculate_information_content, calculate_probability_gap

    assert uncertainty_measure in (
        "entropy",
        "information content",
        "probability gap",
    ), 'uncertainty_measure must be in ["entropy", "information content", "probability gap"]'

    lambda_ = compute_threshold(
        trained_model,
        calib_loader=calibration_loader,
        uncertainty_measure=uncertainty_measure,
        alpha=alpha,
        verbose=verbose,
        device=device,
    )

    with torch.no_grad():
        trained_model.eval()

        uncertainties = []

        for i, (features, _) in enumerate(test_loader):
            if verbose:
                print(f"Test Batch: {i+1}/{len(test_loader)}")
            features = features.to(device)

            # get logits
            logits = trained_model(features)

            # calculate test uncertainty
            if uncertainty_measure == "entropy":
                uncertainty = calculate_entropy(logits, apply_softmax=True)
            elif uncertainty_measure == "information content":
                uncertainty = calculate_information_content(logits, apply_softmax=True)
            elif uncertainty_measure == "probability gap":
                uncertainty = calculate_probability_gap(logits, apply_softmax=True)

            # save
            uncertainties.append(uncertainty)

    # combine
    uncertainties = torch.cat(uncertainties)

    # calculate which points are uncertain
    if uncertainty_measure in ("entropy", "information content"):
        uncertain_points = uncertainties >= lambda_
    else:
        uncertain_points = uncertainties <= lambda_

    return uncertain_points


######################### More efficient implementations for method evaluation #########################


def compute_threshold_all(trained_model, calib_loader, alpha, verbose=False, device="mps"):
    """
    This is a PyTorch implementation of Algorithm 2.
    Note that this implementation calculates all three
    uncertainty thresholds simultaneously for computational reasons.

    Args:
        trained_model: PyTorch model
        calib_loader: PyTorch DataLoader with calibration data
        alpha: quantile level (alpha=0.05 => 95% quantile)

    Returns: (entropy_threshold, information_content_threshold, probability_gap_threshold)
    """
    import torch
    from src.utils import calculate_probability_gap, calculate_entropy, calculate_information_content

    with torch.no_grad():
        trained_model.eval()

        entropies = []
        information_contents = []
        probability_gaps = []

        for i, (features, _) in enumerate(calib_loader):
            if verbose:
                print(f"Calibration Batch: {i+1}/{len(calib_loader)}")
            features = features.to(device)

            # Get logits
            logits = trained_model(features)

            # calculate uncertainty
            entropy = calculate_entropy(logits, apply_softmax=True)
            information_content = calculate_information_content(logits, apply_softmax=True)
            probability_gap = calculate_probability_gap(logits, apply_softmax=True)

            # save
            entropies.append(entropy)
            information_contents.append(information_content)
            probability_gaps.append(probability_gap)

    # combine
    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)

    # compute thresholds
    entropy_threshold = torch.quantile(entropies, 1 - alpha, interpolation="higher")
    information_content_threshold = torch.quantile(information_contents, 1 - alpha, interpolation="higher")
    probability_gap_threshold = torch.quantile(probability_gaps, alpha, interpolation="lower")

    return entropy_threshold, information_content_threshold, probability_gap_threshold


def identify_uncertain_points_all(trained_model, calib_loader, test_loader, alpha, verbose=False, device="mps"):
    """
    This is a PyTorch implementation of Algorithm 1. Uncertain points are from the test loader.

    Note that this implementation slightly varies from the algorithm
    for computational and ease of use reasons.

    Note that this algorithm does NOT require thresholds calculated in advance.

    Args:
        trained_model: PyTorch model
        test_loader: PyTorch DataLoader with test data
        alpha: quantile level (alpha=0.05 => 95% quantile)

    Returns: {
        "highly_uncertain_entropy": highly_uncertain_entropy,
        "highly_uncertain_information_content": highly_uncertain_information_content,
        "highly_uncertain_probability_gap": highly_uncertain_probability_gap,
        "highly_uncertain_any": highly_uncertain_any,
        "highly_uncertain_all": highly_uncertain_all,
        "is_correct_predictions": is_correct_predictions,
    }
    """
    import torch
    from src.utils import calculate_entropy, calculate_information_content, calculate_probability_gap

    # Get thresholds from Algorithm 2
    entropy_threshold, information_content_threshold, probability_gap_threshold = compute_threshold_all(
        trained_model, calib_loader, alpha, verbose=verbose, device=device
    )

    with torch.no_grad():
        trained_model.eval()

        entropies = []
        information_contents = []
        probability_gaps = []
        is_correct_predictions = []

        for i, (features, labels) in enumerate(test_loader):
            if verbose:
                print(f"Test Batch: {i+1}/{len(test_loader)}")
            features = features.to(device)
            labels = labels.to(device)

            # get logits
            logits = trained_model(features)

            # this part is ONLY needed for method evaluation. Not required in practice.
            predicted_probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = predicted_probabilities.argmax(dim=-1)
            is_correct = predictions == labels

            # calculate test uncertainty
            entropy = calculate_entropy(logits, apply_softmax=True)
            information_content = calculate_information_content(logits, apply_softmax=True)
            probability_gap = calculate_probability_gap(logits, apply_softmax=True)

            # save
            entropies.append(entropy)
            information_contents.append(information_content)
            probability_gaps.append(probability_gap)
            is_correct_predictions.append(is_correct)

    # combine
    entropies = torch.cat(entropies)
    information_contents = torch.cat(information_contents)
    probability_gaps = torch.cat(probability_gaps)
    is_correct_predictions = torch.cat(is_correct_predictions)

    highly_uncertain_entropy = entropies >= entropy_threshold
    highly_uncertain_information_content = information_contents >= information_content_threshold
    highly_uncertain_probability_gap = probability_gaps <= probability_gap_threshold
    highly_uncertain_any = (
        highly_uncertain_entropy | highly_uncertain_information_content | highly_uncertain_probability_gap
    )
    highly_uncertain_all = (
        highly_uncertain_entropy & highly_uncertain_information_content & highly_uncertain_probability_gap
    )

    return {
        "highly_uncertain_entropy": highly_uncertain_entropy,
        "highly_uncertain_information_content": highly_uncertain_information_content,
        "highly_uncertain_probability_gap": highly_uncertain_probability_gap,
        "highly_uncertain_any": highly_uncertain_any,
        "highly_uncertain_all": highly_uncertain_all,
        "is_correct_predictions": is_correct_predictions,
    }


######################### Case Study 3 #########################


def get_uncertainty_features(trained_model, loader, return_loader=False, verbose=False, device="mps"):
    """
    An implementation of Functions 1 and 2, required for Algorithm 4.

    Returns (features, labels)

    Args:
        * trained_model: a trained PyTorch model
        * loader: PyTorch data loader
        * return_loader: if True, returns (loader, dataset)
    """

    import torch
    from src.utils import calculate_entropy, calculate_information_content, calculate_probability_gap

    with torch.no_grad():

        out_features = []
        out_labels = []

        trained_model.eval()
        for i, (features, labels) in enumerate(loader):
            if verbose:
                print(f"Batch: {i+1}/{len(loader)}")
            features = features.to(device)
            labels = labels.to(device)

            # get logits and predictions
            logits = trained_model(features)
            preds = logits.argmax(dim=-1)
            is_misclassified = preds != labels

            # calculate uncertainty
            entropies = calculate_entropy(logits)
            information_contents = calculate_information_content(logits)
            gaps = calculate_probability_gap(logits)

            # make features
            batch_data = torch.stack(
                (
                    entropies,
                    information_contents,
                    gaps,
                )
            )

            out_features.append(batch_data)
            out_labels.append(is_misclassified.int())

        out_features = torch.cat(out_features, dim=-1).cpu().T
        out_labels = torch.cat(out_labels, dim=0).cpu()

    if return_loader:
        from src.datasets import create_loaders

        out_loader, out_dataset = create_loaders(out_features, out_labels)
        return out_loader, out_dataset

    return out_features, out_labels
