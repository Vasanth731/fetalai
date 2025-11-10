# Image Quality Metrics: Comprehensive Reference Table

## Category 1: Full-Reference (FR) Metrics
*Metrics that require a reference (original/ground truth) image for comparison*

| Metric Name | Paper/Original Reference | Year | GitHub Repository Link |
|------------|-------------------------|------|----------------------|
| **Mean Squared Error (MSE)** | Standard mathematical metric | N/A | https://github.com/photosynthesis-team/piq |
| **Root Mean Squared Error (RMSE)** | Standard mathematical metric | N/A | https://github.com/andrewekhalel/sewar |
| **Peak Signal-to-Noise Ratio (PSNR)** | Standard engineering metric | N/A | https://github.com/photosynthesis-team/piq |
| **Structural Similarity Index (SSIM)** | Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP) | 2004 | https://github.com/photosynthesis-team/piq |
| **Multi-Scale SSIM (MS-SSIM)** | Wang et al. "Multiscale structural similarity for image quality assessment" (Asilomar Conference) | 2003 | https://github.com/photosynthesis-team/piq |
| **Information-Weighted SSIM (IW-SSIM)** | Wang & Li "Information Content Weighting for Perceptual Image Quality Assessment" (IEEE TIP) | 2011 | https://github.com/photosynthesis-team/piq |
| **Complex Wavelet SSIM (CW-SSIM)** | Sampat et al. "Complex Wavelet Structural Similarity" (Asilomar Conference) | 2009 | N/A |
| **Visual Information Fidelity (VIF)** | Sheikh & Bovik "Image Information and Visual Quality" (IEEE TIP) | 2006 | https://github.com/aizvorski/video-quality |
| **Feature Similarity Index (FSIM)** | Zhang et al. "FSIM: A Feature Similarity Index for Image Quality Assessment" (IEEE TIP) | 2011 | https://github.com/photosynthesis-team/piq |
| **Feature Similarity Index for Color (FSIMc)** | Extension of FSIM for color images | 2011 | https://github.com/photosynthesis-team/piq |
| **Gradient Magnitude Similarity Deviation (GMSD)** | Xue et al. "Gradient Magnitude Similarity Deviation" (IEEE TIP) | 2014 | https://github.com/photosynthesis-team/piq |
| **Multi-Scale GMSD (MS-GMSD)** | Zhang et al. "Multi-Scale Gradient Magnitude Similarity Deviation" | 2017 | https://github.com/photosynthesis-team/piq |
| **Visual Saliency-Induced Index (VSI)** | Zhang et al. "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment" (IEEE TIP) | 2014 | https://github.com/photosynthesis-team/piq |
| **Most Apparent Distortion (MAD)** | Larson & Chandler "Most Apparent Distortion" (IEEE TIP) | 2010 | N/A |
| **Learned Perceptual Image Patch Similarity (LPIPS)** | Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR) | 2018 | https://github.com/richzhang/PerceptualSimilarity |
| **Deep Image Structure and Texture Similarity (DISTS)** | Ding et al. "Image Quality Assessment: Unifying Structure and Texture Similarity" (IEEE TPAMI) | 2020 | https://github.com/photosynthesis-team/piq |
| **Perceptual Image-Error Assessment through Pairwise Preference (PieAPP)** | Prashnani et al. "PieAPP: Perceptual Image-Error Assessment through Pairwise Preference" (CVPR) | 2018 | https://github.com/photosynthesis-team/piq |
| **HaarPSI** | Reisenhofer et al. "A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment" (Signal Processing: Image Communication) | 2018 | https://github.com/photosynthesis-team/piq |
| **Mean Deviation Similarity Index (MDSI)** | Nafchi et al. "Mean Deviation Similarity Index" (IEEE TIP) | 2016 | https://github.com/photosynthesis-team/piq |
| **Spectral Residual Based Similarity (SR-SIM)** | Zhang & Li "SR-SIM: A Fast and High Performance IQA Index based on Spectral Residual" (ICIP) | 2012 | https://github.com/photosynthesis-team/piq |
| **Content Score** | Gatys et al. "Image Style Transfer Using Convolutional Neural Networks" (CVPR) | 2016 | https://github.com/photosynthesis-team/piq |
| **Style Score** | Gatys et al. "Image Style Transfer Using Convolutional Neural Networks" (CVPR) | 2016 | https://github.com/photosynthesis-team/piq |
| **Universal Quality Index (UQI)** | Wang & Bovik "A Universal Image Quality Index" (IEEE Signal Processing Letters) | 2002 | https://github.com/andrewekhalel/sewar |
| **Video Multi-Method Assessment Fusion (VMAF)** | Netflix "Toward a Practical Perceptual Video Quality Metric" | 2016 | https://github.com/Netflix/vmaf |
| **PSNR-HVS** | Egiazarian et al. "New Full-Reference Quality Metrics Based on HVS" | 2006 | https://github.com/rolinh/VQMT |
| **PSNR-HVS-M** | Ponomarenko et al. "On between-coefficient contrast masking of DCT basis functions" | 2007 | https://github.com/rolinh/VQMT |
| **Shift-Tolerant LPIPS (ST-LPIPS)** | Ghildyal & Liu "Shift-tolerant Perceptual Similarity Metric" (ECCV) | 2022 | https://github.com/abhijay9/ShiftTolerant-LPIPS |
| **E-LPIPS** | Kettunen et al. "E-LPIPS: Robust Perceptual Image Similarity via Random Transformation Ensembles" | 2019 | https://github.com/mkettune/elpips |
| **ERGAS** | Ranchin & Wald "Fusion of high spatial and spectral resolution images" | 2000 | https://github.com/andrewekhalel/sewar |
| **Spatial Correlation Coefficient (SCC)** | Zhou et al. "A wavelet transform method to merge Landsat TM and SPOT panchromatic data" | 1998 | https://github.com/andrewekhalel/sewar |
| **Relative Average Spectral Error (RASE)** | Remote sensing quality metric | N/A | https://github.com/andrewekhalel/sewar |
| **Spectral Angle Mapper (SAM)** | Kruse et al. "The Spectral Image Processing System (SIPS)" | 1993 | https://github.com/andrewekhalel/sewar |
| **PSNR-B (Block-Sensitive PSNR)** | Yim & Bovik "Quality Assessment of Deblocked Images" | 2011 | https://github.com/andrewekhalel/sewar |

## Category 2: No-Reference (NR) / Blind Metrics
*Metrics that do NOT require a reference image*

| Metric Name | Paper/Original Reference | Year | GitHub Repository Link |
|------------|-------------------------|------|----------------------|
| **Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)** | Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain" (IEEE TIP) | 2012 | https://github.com/EadCat/NIQA |
| **Natural Image Quality Evaluator (NIQE)** | Mittal et al. "Making a Completely Blind Image Quality Analyzer" (IEEE Signal Processing Letters) | 2013 | https://github.com/EadCat/NIQA |
| **Perception-based Image Quality Evaluator (PIQE)** | Venkatanath et al. "Blind Image Quality Evaluation Using Perception Based Features" (NCC) | 2015 | https://github.com/EadCat/NIQA |
| **Inception Score (IS)** | Salimans et al. "Improved Techniques for Training GANs" (NeurIPS) | 2016 | https://github.com/photosynthesis-team/piq |
| **Fréchet Inception Distance (FID)** | Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS) | 2017 | https://github.com/photosynthesis-team/piq |
| **Kernel Inception Distance (KID)** | Bińkowski et al. "Demystifying MMD GANs" (ICLR) | 2018 | https://github.com/photosynthesis-team/piq |
| **DIIVINE** | Moorthy & Bovik "Blind Image Quality Assessment: From Natural Scene Statistics to Perceptual Quality" (IEEE TIP) | 2011 | http://live.ece.utexas.edu/research/Quality/nrqa.htm |
| **CORNIA** | Ye & Doermann "No-Reference Image Quality Assessment Using Visual Codebooks" (IEEE TIP) | 2012 | N/A |
| **HOSA** | Xu et al. "Blind Image Quality Assessment Based on High Order Statistics Aggregation" (IEEE TIP) | 2016 | N/A |
| **RankIQA** | Liu et al. "RankIQA: Learning from Rankings for No-Reference Image Quality Assessment" (ICCV) | 2017 | https://github.com/EadCat/NIQA |
| **MetaIQA** | Zhu et al. "MetaIQA: Deep Meta-learning for No-Reference Image Quality Assessment" (CVPR) | 2020 | https://github.com/EadCat/NIQA |
| **MUSIQ** | Ke et al. "MUSIQ: Multi-scale Image Quality Transformer" (ICCV) | 2021 | https://github.com/google-research/google-research/tree/master/musiq |
| **TReS** | Golestaneh et al. "No-Reference Image Quality Assessment via Transformers" (CVPR) | 2022 | N/A |
| **CLIPIQA** | Wang et al. "Exploring CLIP for Assessing the Look and Feel of Images" (AAAI) | 2023 | N/A |
| **Quality with No Reference (QNR)** | Alparone et al. "Multispectral and Panchromatic Data Fusion Assessment Without Reference" | 2008 | https://github.com/andrewekhalel/sewar |
| **Spatial Distortion Index (D_S)** | Alparone et al. "Multispectral and Panchromatic Data Fusion Assessment Without Reference" | 2008 | https://github.com/andrewekhalel/sewar |
| **Spectral Distortion Index (D_lambda)** | Alparone et al. "Multispectral and Panchromatic Data Fusion Assessment Without Reference" | 2008 | https://github.com/andrewekhalel/sewar |

## Hybrid / Distribution-Based Metrics
*Metrics that compare distributions of image sets*

| Metric Name | Paper/Original Reference | Year | GitHub Repository Link |
|------------|-------------------------|------|----------------------|
| **Fréchet Video Distance (FVD)** | Unterthiner et al. "Towards Accurate Generative Models of Video" (ICLR) | 2019 | https://github.com/google-research/google-research/tree/master/frechet_video_distance |
| **Fréchet Audio Distance (FAD)** | Kilgour et al. "Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms" | 2019 | N/A |
| **CLIP Maximum Mean Discrepancy (CMMD)** | Jayasumana et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation" (CVPR) | 2024 | N/A |
| **Precision and Recall for Distributions** | Sajjadi et al. "Assessing Generative Models via Precision and Recall" (NeurIPS) | 2018 | N/A |
| **Density and Coverage** | Naeem et al. "Reliable Fidelity and Diversity Metrics for Generative Models" (ICML) | 2020 | N/A |

---

## Popular GitHub Repositories for Image Quality Metrics

### 1. **PIQ (PyTorch Image Quality)**
**Link:** https://github.com/photosynthesis-team/piq
- **Description:** Comprehensive collection of FR and NR metrics in PyTorch
- **Metrics Included:** PSNR, SSIM, MS-SSIM, VIF, FSIM, GMSD, MS-GMSD, VSI, LPIPS, DISTS, PieAPP, HaarPSI, MDSI, BRISQUE, and more
- **Features:** Both functional and class-based interfaces, can be used as loss functions

### 2. **SEWAR (Segmentation and Image Quality Assessment)**
**Link:** https://github.com/andrewekhalel/sewar
- **Description:** All image quality metrics in one package
- **Metrics Included:** MSE, RMSE, PSNR, UQI, SSIM, MS-SSIM, ERGAS, SCC, RASE, SAM, VIF, PSNR-B
- **Features:** Command-line support, Python library

### 3. **PerceptualSimilarity (LPIPS Official)**
**Link:** https://github.com/richzhang/PerceptualSimilarity
- **Description:** Official implementation of LPIPS metric
- **Features:** Multiple network backends (AlexNet, VGG, SqueezeNet), can be used as perceptual loss

### 4. **NIQA (No-Reference Image Quality Assessment)**
**Link:** https://github.com/EadCat/NIQA
- **Description:** Collection of NR-IQA algorithms
- **Metrics Included:** BRISQUE, NIQE, PIQE, RankIQA, MetaIQA

### 5. **Video Quality**
**Link:** https://github.com/aizvorski/video-quality
- **Description:** Video quality metrics reference implementation
- **Metrics Included:** VIF, SSIM, PSNR, MS-SSIM

### 6. **VQMT (Video Quality Measurement Tool)**
**Link:** https://github.com/rolinh/VQMT
- **Description:** Fast implementations of quality metrics for video
- **Metrics Included:** PSNR, SSIM, MS-SSIM, VIF, PSNR-HVS, PSNR-HVS-M

### 7. **FFmpeg Quality Metrics**
**Link:** https://github.com/slhck/ffmpeg-quality-metrics
- **Description:** Calculate quality metrics using FFmpeg
- **Metrics Included:** VMAF, PSNR, SSIM, VIF, MS-SSIM

---

## Key Papers and Surveys

### Comprehensive Reviews:
1. **Wang & Bovik** "Modern Image Quality Assessment" - Synthesis Lectures on Image, Video, and Multimedia Processing (2006)

2. **Lin & Zhang** "FSIM: A Feature Similarity Index for Image Quality Assessment" - IEEE TIP (2011)

3. **Sheikh & Bovik** "Image Information and Visual Quality" - IEEE TIP (2006)

4. **Mittal et al.** "Making a Completely Blind Image Quality Analyzer" - IEEE Signal Processing Letters (2013)

5. **Zhang et al.** "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" - CVPR (2018)

6. **Jayasumana et al.** "Rethinking FID: Towards a Better Evaluation Metric for Image Generation" - arXiv (2024)

---

## Metric Selection Guidelines

### For Natural Images:
- **Full-Reference:** SSIM, MS-SSIM, LPIPS (for perceptual similarity)
- **No-Reference:** NIQE, BRISQUE

### For Compressed Images:
- **Full-Reference:** PSNR, SSIM, MS-SSIM
- **No-Reference:** BRISQUE

### For Generative Models (GANs, Diffusion Models):
- **Distribution-based:** FID, IS, KID
- **Perceptual:** LPIPS, DISTS
- **Modern Alternative:** CMMD (CLIP-based)

### For Video Quality:
- **Full-Reference:** VMAF, SSIM, PSNR, VIF
- **Distribution-based:** FVD

### For Medical Imaging:
- **Full-Reference:** SSIM, VIF, FSIM
- **No-Reference:** BRISQUE, NIQE

### For High Dynamic Range (HDR):
- **Specialized:** HDR-VDP, HDR-VQM

---

## Important Considerations

### Full-Reference Metrics:
**Strengths:**
- More accurate as they have ground truth
- Better correlation with human perception (especially perceptual metrics)
- Suitable for algorithm development and optimization

**Limitations:**
- Require reference image (not always available)
- Cannot be used for evaluating generated images without pairs
- May not capture semantic differences

### No-Reference Metrics:
**Strengths:**
- Work without reference images
- Useful for real-world scenarios
- Can evaluate generated images

**Limitations:**
- Less accurate than FR metrics
- Often trained on specific distortion types
- May not generalize well across domains

### Distribution-Based Metrics (FID, IS, KID):
**Strengths:**
- Evaluate entire distributions
- Capture diversity and quality
- Standard for generative models

**Limitations:**
- Require large sample sizes
- Biased estimators (especially FID)
- Limited to specific feature extractors (Inception, CLIP)
- May not align with human perception in all cases

---

## Historical Timeline

- **1912** - Jaccard Index (adapted for images later)
- **2002** - Universal Quality Index (UQI)
- **2004** - SSIM
- **2006** - VIF, PSNR-HVS
- **2011** - FSIM, DIIVINE
- **2012** - BRISQUE, SR-SIM
- **2013** - NIQE, GMSD
- **2014** - VSI
- **2015** - PIQE
- **2016** - IS, MDSI, HaarPSI, Content/Style Scores
- **2017** - FID, MS-GMSD, RankIQA
- **2018** - LPIPS, PieAPP, KID, HaarPSI
- **2019** - FVD, E-LPIPS
- **2020** - DISTS, MetaIQA
- **2021** - MUSIQ
- **2022** - TReS, ST-LPIPS
- **2023** - CLIPIQA, R-LPIPS
- **2024** - CMMD

---

## Computational Considerations

### Fast Metrics (Real-time capable):
- MSE, RMSE, PSNR
- Simple SSIM implementations
- BRISQUE (lightweight)

### Medium Speed:
- MS-SSIM
- VIF
- FSIM
- NIQE

### Slow Metrics:
- LPIPS (requires deep network)
- FID/KID (requires many samples)
- PieAPP
- DISTS

---

## Programming Language Support

### Python:
- Most metrics available (PIQ, SEWAR, scikit-image, OpenCV)
- Best support for deep learning metrics

### MATLAB:
- Built-in support for SSIM, PSNR, BRISQUE, NIQE, PIQE
- Image Processing Toolbox provides most FR metrics

### C/C++:
- OpenCV provides PSNR, SSIM
- VQMT provides fast implementations

### Command Line:
- FFmpeg for PSNR, SSIM, VMAF
- SEWAR CLI tool

---

## Citation Information

When using these metrics in research, always cite the original papers. For example:

**SSIM:**
```
@article{wang2004image,
  title={Image quality assessment: from error visibility to structural similarity},
  author={Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
  journal={IEEE transactions on image processing},
  volume={13},
  number={4},
  pages={600--612},
  year={2004}
}
```

**FID:**
```
@inproceedings{heusel2017gans,
  title={GANs trained by a two time-scale update rule converge to a local nash equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in neural information processing systems},
  pages={6626--6637},
  year={2017}
}
```

**LPIPS:**
```
@inproceedings{zhang2018unreasonable,
  title={The unreasonable effectiveness of deep features as a perceptual metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={586--595},
  year={2018}
}
```
