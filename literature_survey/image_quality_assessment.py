"""
Image Quality Assessment for Fetal Ultrasound Images
This script implements 2 Full-Reference and 2 No-Reference metrics specifically useful for fetal ultrasound imaging.

Full-Reference Methods:
    1. SSIM (Structural Similarity Index) - Range: [-1, 1], Higher is better
    2. FSIM (Feature Similarity Index) - Range: [0, 1], Higher is better

No-Reference Methods:
    1. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) - Range: [0, 100], Lower is better
    2. NIQE (Natural Image Quality Evaluator) - Range: [0, ∞], Lower is better
"""

import sys
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import convolve
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')


# ======================== FULL-REFERENCE METHODS ========================

def calculate_ssim(reference_img, test_img):
    """
    Calculate Structural Similarity Index (SSIM)

    Args:
        reference_img: Reference/ground truth image
        test_img: Test/degraded image

    Returns:
        float: SSIM value in range [-1, 1], where 1 means identical images

    Quality Interpretation:
        > 0.9  : Excellent quality
        0.7-0.9: Good quality
        0.5-0.7: Fair quality
        < 0.5  : Poor quality
    """
    # Convert to grayscale if needed
    if len(reference_img.shape) == 3:
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        reference_gray = reference_img

    if len(test_img.shape) == 3:
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    else:
        test_gray = test_img

    # Calculate SSIM
    score, _ = ssim(reference_gray, test_gray, full=True)

    return score


def calculate_fsim(reference_img, test_img):
    """
    Calculate Feature Similarity Index (FSIM)

    Args:
        reference_img: Reference/ground truth image
        test_img: Test/degraded image

    Returns:
        float: FSIM value in range [0, 1], where 1 means identical images

    Quality Interpretation:
        > 0.9  : Excellent quality
        0.8-0.9: Good quality
        0.6-0.8: Fair quality
        < 0.6  : Poor quality
    """
    # Convert to grayscale if needed
    if len(reference_img.shape) == 3:
        img1 = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    else:
        img1 = reference_img

    if len(test_img.shape) == 3:
        img2 = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    else:
        img2 = test_img

    # Convert to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants
    T1 = 0.85  # Constant for phase congruency
    T2 = 160   # Constant for gradient magnitude

    # Calculate gradient magnitude using Scharr operator
    def gradient_magnitude(img):
        gx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        gy = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        return np.sqrt(gx**2 + gy**2)

    # Calculate phase congruency (simplified version)
    def phase_congruency_simple(img):
        # Using edge detection as proxy for phase congruency
        edges = cv2.Canny(img.astype(np.uint8), 50, 150)
        return edges.astype(np.float64) / 255.0

    # Get gradient magnitudes
    gm1 = gradient_magnitude(img1)
    gm2 = gradient_magnitude(img2)

    # Get phase congruency
    pc1 = phase_congruency_simple(img1)
    pc2 = phase_congruency_simple(img2)

    # Calculate similarity for gradient magnitude
    S_gm = (2 * gm1 * gm2 + T2) / (gm1**2 + gm2**2 + T2)

    # Calculate similarity for phase congruency
    S_pc = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)

    # Combine similarities
    S_L = S_pc * S_gm

    # Calculate maximum of phase congruency maps
    PC_m = np.maximum(pc1, pc2)

    # Calculate FSIM
    fsim = np.sum(S_L * PC_m) / np.sum(PC_m + 1e-10)

    return fsim


# ======================== NO-REFERENCE METHODS ========================

def calculate_brisque(img):
    """
    Calculate BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

    Args:
        img: Input image

    Returns:
        float: BRISQUE score in range [0, 100], where lower is better

    Quality Interpretation:
        0-20  : Excellent quality
        20-40 : Good quality
        40-60 : Fair quality
        60-80 : Poor quality
        80-100: Very poor quality
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float64)

    # Normalize to zero mean and unit variance
    def compute_features(img):
        # MSCN (Mean Subtracted Contrast Normalized) coefficients
        mu = cv2.GaussianBlur(img, (7, 7), 7/6)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(img * img, (7, 7), 7/6)
        sigma = np.sqrt(np.abs(sigma - mu_sq))

        # Avoid division by zero
        mscn = (img - mu) / (sigma + 1)

        # Calculate features from MSCN coefficients
        alpha, beta_l, beta_r = estimate_aggd_params(mscn)

        return [alpha, (beta_l + beta_r) / 2]

    def estimate_aggd_params(vec):
        # Estimate parameters of Asymmetric Generalized Gaussian Distribution
        vec = vec.flatten()

        # Calculate gamma parameters
        gam = np.arange(0.2, 10, 0.001)
        r_gam = (gamma(2/gam) ** 2) / (gamma(1/gam) * gamma(3/gam))

        # Calculate left and right variance
        left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
        right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))

        # Calculate gamma mean
        gamma_mean = np.sqrt(np.mean(vec ** 2))

        # Prevent division by zero
        if gamma_mean < 1e-10:
            gamma_mean = 1e-10

        # Calculate r
        r = np.mean(np.abs(vec)) ** 2 / np.mean(vec ** 2)
        r = np.clip(r, 0.2, 0.999)

        # Find best matching gamma
        array_position = np.argmin(np.abs(r_gam - r))
        alpha = gam[array_position]

        return alpha, left_std, right_std

    # Extract features
    features = compute_features(gray)

    # Simplified BRISQUE score (actual BRISQUE uses SVR on extracted features)
    # This is a simplified version that gives an approximation
    # Lower alpha and higher beta indicate better quality
    score = abs(features[0] - 1.0) * 50 + (1.0 / (features[1] + 0.1)) * 10
    score = np.clip(score, 0, 100)

    return score


def calculate_niqe(img):
    """
    Calculate NIQE (Natural Image Quality Evaluator)

    Args:
        img: Input image

    Returns:
        float: NIQE score in range [0, ∞], where lower is better

    Quality Interpretation:
        0-3 : Excellent quality
        3-5 : Good quality
        5-7 : Fair quality
        7-10: Poor quality
        > 10: Very poor quality
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float64)

    # Patch size
    patch_size = 96

    # Calculate MSCN coefficients
    mu = cv2.GaussianBlur(gray, (7, 7), 7/6)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(gray * gray, (7, 7), 7/6)
    sigma = np.sqrt(np.abs(sigma - mu_sq))

    # MSCN coefficients
    mscn = (gray - mu) / (sigma + 1)

    # Extract patches
    h, w = mscn.shape
    num_patches = 0
    feature_vectors = []

    for i in range(0, h - patch_size, patch_size // 2):
        for j in range(0, w - patch_size, patch_size // 2):
            patch = mscn[i:i+patch_size, j:j+patch_size]

            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Extract features from patch
                patch_mean = np.mean(patch)
                patch_var = np.var(patch)

                feature_vectors.append([patch_mean, patch_var])
                num_patches += 1

    if num_patches == 0:
        return 10.0  # Return poor quality if no patches found

    # Convert to numpy array
    features = np.array(feature_vectors)

    # Calculate multivariate Gaussian parameters
    mu_features = np.mean(features, axis=0)

    # Natural scene statistics parameters (pre-trained on pristine images)
    # These are simplified approximations
    mu_pristine = np.array([0.0, 1.0])  # Mean of pristine images

    # Calculate NIQE score as distance from pristine statistics
    niqe_score = np.linalg.norm(mu_features - mu_pristine)

    return niqe_score


# ======================== HELPER FUNCTIONS ========================

def interpret_quality(metric_name, score):
    """
    Interpret the quality score and provide a description
    """
    if metric_name == "SSIM":
        if score > 0.9:
            return "Excellent quality"
        elif score > 0.7:
            return "Good quality"
        elif score > 0.5:
            return "Fair quality"
        else:
            return "Poor quality"

    elif metric_name == "FSIM":
        if score > 0.9:
            return "Excellent quality"
        elif score > 0.8:
            return "Good quality"
        elif score > 0.6:
            return "Fair quality"
        else:
            return "Poor quality"

    elif metric_name == "BRISQUE":
        if score < 20:
            return "Excellent quality"
        elif score < 40:
            return "Good quality"
        elif score < 60:
            return "Fair quality"
        elif score < 80:
            return "Poor quality"
        else:
            return "Very poor quality"

    elif metric_name == "NIQE":
        if score < 3:
            return "Excellent quality"
        elif score < 5:
            return "Good quality"
        elif score < 7:
            return "Fair quality"
        elif score < 10:
            return "Poor quality"
        else:
            return "Very poor quality"

    return "Unknown"


def print_results(metric_name, score, score_range, interpretation):
    """
    Print formatted results
    """
    print(f"\n{'='*60}")
    print(f"Metric: {metric_name}")
    print(f"{'='*60}")
    print(f"Score: {score:.4f}")
    print(f"Range: {score_range}")
    print(f"Quality Assessment: {interpretation}")
    print(f"{'='*60}")


# ======================== MAIN FUNCTION ========================

def assess_image_quality(test_img_path, reference_img_path=None):
    """
    Assess image quality using multiple metrics

    Args:
        test_img_path: Path to the test image
        reference_img_path: Path to the reference image (optional, for FR metrics)
    """
    # Load test image
    test_img = cv2.imread(test_img_path)
    if test_img is None:
        print(f"Error: Could not load test image from {test_img_path}")
        return

    print("\n" + "="*60)
    print("FETAL ULTRASOUND IMAGE QUALITY ASSESSMENT")
    print("="*60)
    print(f"Test Image: {test_img_path}")

    # Full-Reference Metrics (if reference image is provided)
    if reference_img_path:
        reference_img = cv2.imread(reference_img_path)
        if reference_img is None:
            print(f"Error: Could not load reference image from {reference_img_path}")
        else:
            print(f"Reference Image: {reference_img_path}")
            print("\n" + "="*60)
            print("FULL-REFERENCE METRICS")
            print("="*60)

            # Calculate SSIM
            ssim_score = calculate_ssim(reference_img, test_img)
            interpretation = interpret_quality("SSIM", ssim_score)
            print_results("SSIM (Structural Similarity Index)",
                         ssim_score,
                         "[-1, 1] (Higher is better)",
                         interpretation)

            # Calculate FSIM
            fsim_score = calculate_fsim(reference_img, test_img)
            interpretation = interpret_quality("FSIM", fsim_score)
            print_results("FSIM (Feature Similarity Index)",
                         fsim_score,
                         "[0, 1] (Higher is better)",
                         interpretation)

    # No-Reference Metrics
    print("\n" + "="*60)
    print("NO-REFERENCE METRICS")
    print("="*60)

    # Calculate BRISQUE
    brisque_score = calculate_brisque(test_img)
    interpretation = interpret_quality("BRISQUE", brisque_score)
    print_results("BRISQUE (Blind Image Spatial Quality)",
                 brisque_score,
                 "[0, 100] (Lower is better)",
                 interpretation)

    # Calculate NIQE
    niqe_score = calculate_niqe(test_img)
    interpretation = interpret_quality("NIQE", niqe_score)
    print_results("NIQE (Natural Image Quality Evaluator)",
                 niqe_score,
                 "[0, ∞] (Lower is better)",
                 interpretation)

    print("\n" + "="*60)
    print("ASSESSMENT COMPLETE")
    print("="*60 + "\n")


# ======================== EXAMPLE USAGE ========================

if __name__ == "__main__":
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Please provide at least one image path.")
        print("Example: python image_quality_assessment.py test_image.png [reference_image.png]")

        # Assess good quality image
        print("\n" + "="*60)
        print("="*60)
        assess_image_quality('/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/new_parsed_dataset/02_BPD_HC_Transthalamic/INDTNCHE0100195_visit_1_Head_1_Transthalamic plane_1.png', '/home/htic/fetalai_data/vasanth/datasets_to_work_with/parsed/old_parsed_biometry/correct_size/bpd/51.png')


    elif len(sys.argv) == 2:
        # No-reference only
        assess_image_quality(sys.argv[1])
    else:
        # Full-reference + No-reference
        assess_image_quality(sys.argv[1], sys.argv[2])
