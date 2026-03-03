
# Multi-Model GRB Light Curve Reconstructor

**A Unified Computational Framework for Temporal Reconstruction and Gap Mitigation in Gamma-Ray Burst Light Curves**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Abstract

This repository contains the implementation of a comparative framework for reconstructing Gamma-Ray Burst (GRB) light curves, specifically targeting the mitigation of temporal data gaps inherent to satellite observations. The framework integrates two distinct methodological paradigms: a deep learning approach utilizing an **Attention-based U-Net architecture**, and a statistical approach employing **Quartic Smoothing Splines (QSS)**.

The primary objective is to enhance the fidelity of light curve feature extraction—specifically the plateau emission end time ($T_a$) and luminosity ($L_a$)—thereby reducing uncertainty in the Dainotti relation. This code supports the analysis presented in *Kaushal et al. (2025)* regarding the application of machine learning and statistical smoothing to high-energy astrophysics time-series data.

## 2. Theoretical Motivation

### 2.1 The Gap Problem
GRB light curves observed by instruments such as *Swift* (BAT/XRT) often suffer from temporal gaps due to orbital constraints, Earth occultation, or instrumental downtime. These gaps disrupt the continuity required for accurate determination of the plateau phase parameters.

### 2.2 Log-Space Transformation
Gamma-ray flux $F(t)$ generally follows a power-law decay. To stabilize variance and linearize the temporal evolution for neural network processing, the framework operates in logarithmic space. Given time $t$ and flux $F$:

$$
x = \log_{10}(t), \quad y = \log_{10}(F)
$$

Asymmetric flux errors $\sigma_{F+}, \sigma_{F-}$ are propagated to logarithmic space using standard error propagation theory:

$$
\sigma_{\log F} \approx \frac{\sigma_F}{F \ln(10)}
$$

## 3. Methodological Implementations

### 3.1 Model 1: Attention U-Net (Deep Learning)

This model adapts the U-Net architecture, originally designed for biomedical image segmentation, for 1D time-series reconstruction. The core innovation is the integration of attention gates (AGs) into the skip connections.

#### 3.1.1 Architectural Topology
The network comprises an encoder-decoder structure:
*   **Encoder**: Extracts hierarchical features at progressively lower temporal resolutions via Max Pooling.
*   **Bottleneck**: Captures global context at the coarsest resolution.
*   **Decoder**: Upsamples features using UpSampling1D layers.

#### 3.1.2 Attention Mechanism
Attention Gates ($AG$) filter the features propagated through skip connections. Let $x^l \in \mathbb{R}^{N \times C}$ be the feature map from the encoder and $g$ be the gating signal from the decoder. The attention coefficients $\alpha \in [0,1]$ are computed as:

$$
q_{att} = \psi^T(\sigma_1(W_x^T x^l + W_g^T g + b_g)) + b_\psi
$$

$$
\alpha = \sigma_2(q_{att})
$$

Where $\sigma_1$ is a ReLU activation and $\sigma_2$ is a Sigmoid activation. The output is the element-wise multiplication: $\hat{x}^l = x^l \cdot \alpha$. This allows the model to suppress irrelevant background noise and focus on salient temporal features (e.g., flares, plateau breaks).

#### 3.1.3 Confidence Interval Estimation
Since the U-Net is deterministic, uncertainty is quantified via a stochastic error synthesis. The distribution of observed errors is fitted to a Normal distribution. During inference, synthetic noise is sampled and added to the mean prediction to generate a distribution of possible realizations. The 95% confidence interval is derived from the percentiles of these Monte Carlo samples.

### 3.2 Model 2: Quartic Smoothing Spline (Statistical)

This approach utilizes non-parametric regression to fit a smooth curve through the observed data points.

#### 3.2.1 Mathematical Formulation
A smoothing spline minimizes the penalized sum of squares:

$$
SS(f) = \sum_{i=1}^{N} \left( y_i - f(x_i) \right)^2 + \lambda \int_{x_1}^{x_N} \left( f^{(k)}(x) \right)^2 dx
$$

Where:
*   $k=4$ denotes the quartic order (cubic penalty on curvature), providing a higher degree of smoothness compared to standard cubic splines ($k=3$).
*   $\lambda$ is the smoothing parameter.

#### 3.2.2 Smoothing Parameter Optimization
The smoothing factor $s$ is dynamically calculated as $s = N \times \text{multiplier}$, where $N$ is the number of valid data points. This ensures the smoothing adapts to the density of the dataset.

#### 3.2.3 Error Synthesis
To generate synthetic error bars for reconstructed points, the framework analyzes the empirical distribution of the input flux errors. It fits both Normal and Laplace distributions, selecting the best fit via log-likelihood maximization. This accounts for the often leptokurtic (heavy-tailed) nature of astrophysical errors which standard Gaussian assumptions fail to capture.

## 4. Installation

### 4.1 Environment Setup
It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/your-username/grb-reconstructor.git
cd grb-reconstructor

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Dependencies (`requirements.txt`)
```text
streamlit>=1.28.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.1.0
scipy>=1.9.0
tensorflow>=2.12.0
requests>=2.28.0
```

## 5. Usage

### 5.1 Authentication
This application is secured with a password mechanism for restricted access. You must create a `.streamlit/secrets.toml` file in the root directory:

```toml
# .streamlit/secrets.toml
password = "your_secure_password_here"
```

### 5.2 Running the Application
Execute the Streamlit server via the terminal:

```bash
streamlit run Research.py
```

### 5.3 Configuring Models
1.  **Select Model**: Use the sidebar to choose between "Attention U-Net" or "Quartic Smoothing Spline".
2.  **Input Data**: Provide a raw GitHub URL to a CSV file. The CSV must contain columns for Time, Flux, and corresponding positive/negative errors.
3.  **Hyperparameters**:
    *   **Epochs/Batch Size**: Relevant for the U-Net model training convergence.
    *   **Spline Degree/Multiplier**: Relevant for the QSS model to control smoothness vs. data fidelity.

### 5.4 Outputs
The application generates:
1.  **Reconstructed Plot**: Visualizing observed points, the mean prediction curve, and the 95% confidence region.
2.  **Downloadable CSV**: Containing interpolated time steps, reconstructed flux values, and synthetic error margins.
3.  **High-Res PDF**: Vector-graphics format plot suitable for publication.

## 6. Data Requirements

The input CSV structure should conform to the standard GRB light curve format. The script automatically detects columns using keyword matching.

| Recommended Column Names | Description |
| :--- | :--- |
| `t` or `time` | Time since trigger (seconds). |
| `flux` | Flux density ($erg \, cm^{-2} \, s^{-1}$). |
| `pos_flux_err` | Positive error bound on flux. |
| `neg_flux_err` | Negative error bound on flux. |

*Note: Negative time or flux values are automatically filtered during preprocessing.*

## 7. Repository Structure

```
├── Research.py             # Main Streamlit application logic
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # Authentication credentials (User created)
└── README.md               # Documentation
```

## 8. Citation

If you use this code or the methodologies implemented herein, please cite the associated paper:

```bibtex
@article{KAUSHAL2025100519,
title = {Multi-Model Framework for Reconstructing Gamma-Ray Burst Light Curves},
journal = {Journal of High Energy Astrophysics},
pages = {100519},
year = {2025},
issn = {2214-4048},
doi = {https://doi.org/10.1016/j.jheap.2025.100519},
url = {https://www.sciencedirect.com/science/article/pii/S2214404825002009},
author = {A. Kaushal and A. Manchanda and M.G. Dainotti and K. Gupta and Z. Nogala and A. Madhan and S. Naqi and Ritik Kumar and V. Oad and N. Indoriya and Krishnanjan Sil and D.H. Hartmann and M. Bogdan and A. Pollo and J.X. Prochaska and N. Fraija and D.Debnath},
keywords = {γ-ray bursts, statistical methods, machine learning, light curve reconstruction},
}
```

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 10. Acknowledgements

We acknowledge the use of public data from the Swift data archive. This implementation relies on the robust scientific stack provided by the Python community, specifically `TensorFlow`, `SciPy`, and `Streamlit`.
