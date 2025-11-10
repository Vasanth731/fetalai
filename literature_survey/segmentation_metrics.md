# Segmentation Metrics: Comprehensive Reference Table

| Metric Name | Paper/Original Reference | Year | GitHub Repository Link |
|------------|-------------------------|------|----------------------|
| **Dice Similarity Coefficient (DSC)** | Dice, L. R. "Measures of the Amount of Ecologic Association Between Species" | 1945 | https://github.com/frankkramer-lab/miseval |
| **Sørensen-Dice Index** | Sørensen, T. "A method of establishing groups of equal amplitude in plant sociology" | 1948 | https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch |
| **Intersection over Union (IoU) / Jaccard Index** | Jaccard, P. "The distribution of the flora in the alpine zone" | 1912 | https://github.com/martinkersner/py-img-seg-eval |
| **Mean IoU (mIoU)** | Long et al. "Fully Convolutional Networks for Semantic Segmentation" (CVPR) | 2015 | https://github.com/martinkersner/py-img-seg-eval |
| **Pixel Accuracy** | Standard classification metric adapted for segmentation | N/A | https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch |
| **Sensitivity / Recall / True Positive Rate** | Standard classification metric | N/A | https://github.com/alexdaniel654/Segmentation_Metrics |
| **Specificity / True Negative Rate** | Standard classification metric | N/A | https://github.com/alexdaniel654/Segmentation_Metrics |
| **Precision / Positive Predictive Value** | Standard classification metric | N/A | https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch |
| **F1-Score** | Standard classification metric (harmonic mean of precision and recall) | N/A | https://github.com/frankkramer-lab/miseval |
| **Hausdorff Distance (HD)** | Hausdorff, F. "Grundzüge der Mengenlehre" | 1914 | https://github.com/deepmind/surface-distance |
| **Average Hausdorff Distance (AHD)** | Taha & Hanbury "Metrics for evaluating 3D medical image segmentation" (BMC Medical Imaging) | 2015 | https://github.com/frankkramer-lab/miseval |
| **95th Percentile Hausdorff Distance (HD95)** | Commonly used variant in medical imaging | N/A | https://github.com/deepmind/surface-distance |
| **Balanced Average Hausdorff Distance** | Khalil et al. "On the usage of average Hausdorff distance for segmentation performance assessment" (European Radiology Experimental) | 2020 | N/A |
| **Mean Surface Distance (MSD)** | Standard distance-based metric | N/A | https://github.com/alexdaniel654/Segmentation_Metrics |
| **Average Surface Distance (ASD)** | Standard distance-based metric | N/A | https://github.com/deepmind/surface-distance |
| **Cohen's Kappa** | Cohen, J. "A coefficient of agreement for nominal scales" (Educational and Psychological Measurement) | 1960 | https://github.com/frankkramer-lab/miseval |
| **Rand Index** | Rand, W. M. "Objective Criteria for the Evaluation of Clustering Methods" | 1971 | https://github.com/frankkramer-lab/miseval |
| **Volume Similarity** | Taha & Hanbury "Metrics for evaluating 3D medical image segmentation" | 2015 | https://github.com/alexdaniel654/Segmentation_Metrics |
| **Boundary IoU (BIoU)** | Cheng et al. "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (CVPR) | 2021 | https://github.com/bowenc0221/boundary-iou-api |
| **Trimap IoU** | Used in segmentation benchmarks (e.g., Adobe benchmarks) | N/A | N/A |
| **Panoptic Quality (PQ)** | Kirillov et al. "Panoptic Segmentation" (CVPR) | 2019 | https://github.com/cocodataset/panopticapi |
| **Segmentation Quality (SQ)** | Component of Panoptic Quality | 2019 | https://github.com/cocodataset/panopticapi |
| **Recognition Quality (RQ)** | Component of Panoptic Quality | 2019 | https://github.com/cocodataset/panopticapi |
| **Generalized Dice Score** | Sudre et al. "Generalised Dice overlap as a deep learning loss function" (MICCAI Workshop) | 2017 | https://docs.monai.io |
| **Frequency Weighted IoU** | Standard weighted variant of IoU | N/A | N/A |
| **Boundary F-Measure** | Martin et al. "A database of human segmented natural images" (ICCV) | 2001 | N/A |
| **Average Precision (AP)** | COCO evaluation metric adapted from object detection | N/A | https://github.com/cocodataset/cocoapi |
| **Boundary AP** | Cheng et al. "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (CVPR) | 2021 | https://github.com/bowenc0221/boundary-iou-api |
| **Boundary PQ** | Cheng et al. "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation" (CVPR) | 2021 | https://github.com/bowenc0221/boundary-iou-api |
| **Volume Overlap Error (VOE)** | Heimann et al. "Comparison and Evaluation of Methods for Liver Segmentation" (IEEE TMI) | 2009 | N/A |
| **Relative Volume Difference (RVD)** | Taha & Hanbury "Metrics for evaluating 3D medical image segmentation" | 2015 | https://github.com/alexdaniel654/Segmentation_Metrics |
| **False Positive Rate (FPR)** | Standard classification metric | N/A | https://github.com/frankkramer-lab/miseval |
| **False Negative Rate (FNR)** | Standard classification metric | N/A | https://github.com/frankkramer-lab/miseval |
| **Matthews Correlation Coefficient (MCC)** | Matthews, B. W. "Comparison of the predicted and observed secondary structure" | 1975 | https://github.com/frankkramer-lab/miseval |
| **Normalized Surface Distance (NSD)** | Nikolov et al. "Deep learning to achieve clinically applicable segmentation" | 2018 | https://github.com/deepmind/surface-distance |

## Additional Notes

### Popular GitHub Repositories for Segmentation Metrics:

1. **MISeval** (https://github.com/frankkramer-lab/miseval) - Comprehensive medical image segmentation evaluation library
   - Paper: Müller et al. "MISeval: a Metric Library for Medical Image Segmentation Evaluation" (2022)

2. **Surface Distance** (https://github.com/deepmind/surface-distance) - DeepMind's library for surface-based metrics
   - Includes: Hausdorff Distance, Average Surface Distance, Surface Dice, etc.

3. **MONAI** (https://github.com/Project-MONAI/MONAI) - Medical Open Network for AI
   - Comprehensive metrics implementation: https://docs.monai.io/en/stable/metrics.html

4. **Segmentation Metrics PyTorch** (https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch)
   - PyTorch implementation of common metrics

5. **TorchMetrics** (https://github.com/Lightning-AI/torchmetrics)
   - Lightning AI's comprehensive metrics library including segmentation metrics

### Key Survey Papers:

1. **Wang et al.** "Image segmentation evaluation: a survey of methods" - Artificial Intelligence Review (2020)
   - DOI: 10.1007/s10462-020-09830-9

2. **Müller & Soto-Rey** "Towards a Guideline for Evaluation Metrics in Medical Image Segmentation" - arXiv (2022)
   - arXiv: 2202.05273

3. **Taha & Hanbury** "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool" - BMC Medical Imaging (2015)
   - DOI: 10.1186/s12880-015-0068-x

4. **Zhang et al.** "Image Segmentation Evaluation: A Survey of Unsupervised Methods" - Computer Vision and Image Understanding (2008)

### Metric Categories:

1. **Overlap-based metrics**: Dice, IoU, Volume Similarity
2. **Distance-based metrics**: Hausdorff Distance, Average Surface Distance
3. **Pixel/Voxel-based metrics**: Pixel Accuracy, Sensitivity, Specificity
4. **Statistical metrics**: Cohen's Kappa, Rand Index
5. **Boundary-based metrics**: Boundary IoU, Boundary F-Measure
6. **Task-specific metrics**: Panoptic Quality (for panoptic segmentation)

### Implementation Considerations:

- **For Medical Imaging**: Use DSC, IoU, HD95, and ASD as primary metrics
- **For Natural Images**: Use mIoU and Pixel Accuracy as primary metrics
- **For Instance Segmentation**: Use AP at different IoU thresholds
- **For Boundary Quality**: Use Boundary IoU, HD, or Average Surface Distance
- **Avoid**: Using only Pixel Accuracy for imbalanced datasets (very high background class)

### Historical Context:

- **1912**: Jaccard Index introduced
- **1945-1948**: Dice/Sørensen coefficient independently developed
- **1914**: Hausdorff Distance introduced
- **2015**: Comprehensive 3D medical image segmentation metrics analysis (Taha & Hanbury)
- **2019**: Panoptic Segmentation and PQ metric introduced
- **2021**: Boundary IoU introduced for better boundary evaluation
- **2022**: MISeval library and evaluation guidelines published
