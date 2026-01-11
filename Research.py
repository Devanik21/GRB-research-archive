import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats as st_scipy
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout, 
                                     Input, Conv1D, UpSampling1D, Concatenate,
                                     MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras import initializers
from scipy.interpolate import UnivariateSpline
import os
import time

# --- CONFIGURATION & SEEDS ---
st.set_page_config(page_title="Multi-Model GRB Reconstructor", page_icon="🔭", layout="wide")

# --- AUTHENTICATION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Detailed and welcoming sign-in page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("<h1 style='text-align: center; color: #2E86C1;'>GRB Research Portal</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Restricted Access</h3>", unsafe_allow_html=True)
            st.markdown("""
            You are accessing the **Multi-Model Gamma-Ray Burst Light Curve Reconstructor**.
            
            This secure environment provides access to:
            *   **Attention U-Net** Implementations
            *   **Quadratic Smoothing Spline** Models
            *   **Advanced Error Analysis**
            
            Please verify your credentials to proceed to the laboratory.
            """)
            st.divider()
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            if "password_correct" in st.session_state:
                st.error("Access Denied: Incorrect Password")

    return False

if not check_password():
    st.stop()

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- EXPANDED CATALOG OF FAMOUS BURSTS ---
# --- MAIN APPLICATION ---
st.title("GRB Light Curve Reconstructor - Paper Implementations")

# --- SIDEBAR CONTROLS ---
st.sidebar.subheader("6. Paper Implementations")
paper_model_select = st.sidebar.selectbox("Select Paper Model", 
                                          ["None", 
                                           "Model 1: Attention U-Net (GRB 231210B)",
                                           "Model 2: Quadratic Smoothing Spline (QSS)",
                                           "Model 3: Coming Soon"])

# --- UI FEATURE 1: Advanced Configuration (Sidebar) ---
with st.sidebar.expander("⚙️ Advanced Configuration"):
    if "Attention U-Net" in paper_model_select:
        conf_epochs = st.slider("Training Epochs", 100, 1000, 500, step=50, help="Number of passes through the entire training dataset.")
        conf_batch = st.selectbox("Batch Size", [32, 64, 128, 256], index=2, help="Number of samples per gradient update.")
    elif "Quadratic Smoothing" in paper_model_select:
        conf_k = st.slider("Spline Degree (k)", 3, 5, 4, help="Degree of the smoothing spline. 3=Cubic, 4=Quartic, 5=Quintic.")
        conf_s_mult = st.slider("Smoothing Factor Multiplier", 0.1, 2.0, 1.0, step=0.1, help="Adjusts the smoothing parameter s (s = N * multiplier).")
    else:
        st.info("Select a model to configure parameters.")

run_paper_btn = False
if paper_model_select == "Model 1: Attention U-Net (GRB 231210B)":
    dataset_url = st.sidebar.text_input("Dataset URL (GitHub Raw)", 
        value="https://raw.githubusercontent.com/Devanik21/Bi-LSTM-light-curve-reconstruction-sample/refs/heads/main/GRB%20Data/GRB231210B_trimmed.csv")
    run_paper_btn = st.sidebar.button("Run Paper Model 1", type="primary")
elif paper_model_select == "Model 2: Quadratic Smoothing Spline (QSS)":
    dataset_url = st.sidebar.text_input("Dataset URL (GitHub Raw)", 
        value="https://raw.githubusercontent.com/Devanik21/Bi-LSTM-light-curve-reconstruction-sample/refs/heads/main/GRB%20Data/GRB231210B_trimmed.csv",
        key="qss_url")
    run_paper_btn = st.sidebar.button("Run Paper Model 2", type="primary")
if run_paper_btn and paper_model_select == "Model 1: Attention U-Net (GRB 231210B)":
    st.subheader("Paper Model 1: Attention U-Net on GRB 231210B")
    
    # --- UI FEATURE 2: Paper Context ---
    with st.expander("📄 Methodology Overview", expanded=True):
        st.markdown("""
        **Abstract:** This implementation utilizes a 1D Attention U-Net architecture to reconstruct Gamma-Ray Burst light curves. 
        The model employs an encoder-decoder structure with skip connections gated by attention mechanisms to focus on relevant temporal features.
        
        **Key Features:** Multi-scale feature extraction, Attention gates, Robustness to gaps.
        """)
    
    def AttentionBlock1D(x, g, inter_channels):
        """Attention mechanism for U-Net"""
        from tensorflow.keras.layers import Conv1D, ReLU
        theta_x = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(x)
        phi_g = Conv1D(inter_channels, kernel_size=1, strides=1, padding="same")(g)
        f = ReLU()(theta_x + phi_g)
        psi_f = Conv1D(1, kernel_size=1, strides=1, padding="same", activation="sigmoid")(f)
        return x * psi_f

    def UNetWithAttention1D(input_shape):
        """Attention U-Net architecture from paper"""
        from tensorflow.keras.layers import (Conv1D, MaxPooling1D, UpSampling1D, 
                                             Flatten, Dense, concatenate)
        
        inputs = Input(shape=input_shape)

        # Encoder
        conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(inputs)
        conv1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(conv1)
        pool1 = MaxPooling1D(pool_size=2, padding='same')(conv1)

        conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(pool1)
        conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(conv2)
        pool2 = MaxPooling1D(pool_size=2, padding='same')(conv2)

        conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(pool2)
        conv3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', 
                      kernel_initializer='he_uniform')(conv3)
        pool3 = MaxPooling1D(pool_size=2, padding='same')(conv3)

        # Bottleneck
        bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', 
                           kernel_initializer='he_uniform')(pool3)
        bottleneck = Conv1D(256, kernel_size=3, activation='relu', padding='same', 
                           kernel_initializer='he_uniform')(bottleneck)

        # Decoder
        upconv3 = UpSampling1D(size=2)(bottleneck)
        attention3 = AttentionBlock1D(conv3, upconv3, inter_channels=64)
        concat3 = concatenate([upconv3, attention3], axis=-1)
        conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(concat3)
        conv_dec3 = Conv1D(128, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(conv_dec3)

        upconv2 = UpSampling1D(size=2)(conv_dec3)
        attention2 = AttentionBlock1D(conv2, upconv2, inter_channels=32)
        concat2 = concatenate([upconv2, attention2], axis=-1)
        conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(concat2)
        conv_dec2 = Conv1D(64, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(conv_dec2)

        upconv1 = UpSampling1D(size=2)(conv_dec2)
        attention1 = AttentionBlock1D(conv1, upconv1, inter_channels=16)
        concat1 = concatenate([upconv1, attention1], axis=-1)
        conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(concat1)
        conv_dec1 = Conv1D(32, kernel_size=3, activation='relu', padding='same', 
                          kernel_initializer='he_uniform')(conv_dec1)

        outputs = Conv1D(1, kernel_size=1, activation=None)(conv_dec1)
        outputs = Flatten()(outputs)
        outputs = Dense(input_shape[0], activation="linear")(outputs)

        model = Model(inputs, outputs)
        return model
    
    def train_attention_unet():
        with st.spinner("Loading data and training Attention U-Net model..."):
            start_time = time.time()
            # Load Data
            try:
                if "username/repo" in dataset_url:
                    st.warning("Using placeholder URL. Please update the Dataset URL in the sidebar.")
                    t_mock = np.logspace(1, 5, 50)
                    f_mock = 1e-10 * (t_mock**-1.5) * (1 + 0.1*np.random.randn(50))
                    trimmed_data = pd.DataFrame({'t': t_mock, 'flux': f_mock, 
                                                'pos_flux_err': 0.1*f_mock, 
                                                'neg_flux_err': 0.1*f_mock})
                else:
                    response = requests.get(dataset_url)
                    if response.status_code == 200:
                        trimmed_data = pd.read_csv(io.StringIO(response.text))
                    else:
                        st.error(f"Failed to load data: HTTP {response.status_code}")
                        return
            except Exception as e:
                st.error(f"Error: {e}")
                return

            # --- UI FEATURE 3: Data Inspection ---
            with st.expander("🔍 Inspect Input Data"):
                st.dataframe(trimmed_data.head(), use_container_width=True)
                st.caption(f"Loaded {len(trimmed_data)} rows. Columns: {list(trimmed_data.columns)}")

            grb_name = "GRB231210B"
            
            # Preprocessing
            cols = trimmed_data.columns
            t_col = next((c for c in cols if 'time' in c.lower() or 't' == c.lower()), cols[0])
            f_col = next((c for c in cols if 'flux' in c.lower()), cols[1])

            ts = trimmed_data[t_col].values
            fluxes = trimmed_data[f_col].values
            
            mask = (ts > 0) & (fluxes > 0)
            ts, fluxes = ts[mask], fluxes[mask]
            
            train_x_denorm = np.log10(ts)
            train_y_denorm = np.log10(fluxes)
            
            # Error handling
            pos_err_col = next((c for c in cols if 'pos' in c.lower() and 'flux' in c.lower()), None)
            neg_err_col = next((c for c in cols if 'neg' in c.lower() and 'flux' in c.lower()), None)
            
            if pos_err_col and neg_err_col:
                pos_flux_err = trimmed_data[pos_err_col].values[mask]
                neg_flux_err = trimmed_data[neg_err_col].values[mask]
                fluxes_error = (pos_flux_err - neg_flux_err) / 2
                logfluxerrs = fluxes_error / (fluxes * np.log(10))
                lower_err_log = logfluxerrs
                upper_err_log = logfluxerrs
            else:
                fluxes_error = 0.1 * fluxes
                logfluxerrs = fluxes_error / (fluxes * np.log(10))
                lower_err_log = logfluxerrs
                upper_err_log = logfluxerrs
            
            # Generate reconstruction points
            gaps = np.diff(train_x_denorm)
            min_gap = 0.05
            recon_log_t = [train_x_denorm[0]]
            total_span = train_x_denorm[-1] - train_x_denorm[0]
            
            fraction = 0.3 if len(ts) > 100 else 0.4
            n_points = max(20, int(fraction * len(ts)))
            
            for i in range(len(ts) - 1):
                gap_size = train_x_denorm[i+1] - train_x_denorm[i]
                if gap_size > min_gap:
                    interval_points = max(2, int(n_points * gap_size / total_span))
                    interval = np.linspace(train_x_denorm[i], train_x_denorm[i+1], 
                                         interval_points, endpoint=True)
                    recon_log_t.extend(interval[1:])
            
            test_x_denorm = np.array(recon_log_t)
            test_x_denorm = np.unique(test_x_denorm)
            
            # Build and train Attention U-Net
            model = UNetWithAttention1D(input_shape=(1, 1))
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            
            X_train = train_x_denorm.reshape(-1, 1, 1)
            y_train = train_y_denorm.reshape(-1, 1, 1)
            
            # --- UI FEATURE 4: Model Architecture ---
            with st.expander("🧠 Model Architecture Details"):
                string_io = io.StringIO()
                model.summary(print_fn=lambda x: string_io.write(x + '\n'))
                st.code(string_io.getvalue(), language='text')
            
            history = model.fit(X_train, y_train, epochs=conf_epochs, verbose=0, batch_size=conf_batch)
            
            # Predictions
            x_test = test_x_denorm.reshape(-1, 1, 1)
            mean_prediction_denorm = model.predict(x_test, verbose=0).flatten()
            
            # Generate noise and confidence intervals
            errparameters = st_scipy.norm.fit(logfluxerrs)
            err_dist = st_scipy.norm(loc=errparameters[0], scale=errparameters[1])
            recon_errorbar = err_dist.rvs(size=len(mean_prediction_denorm))
            recon_errorbar = np.where(recon_errorbar < 0, 0, recon_errorbar)
            
            point_specific_noise = np.array([
                st_scipy.norm(loc=pred, scale=err).rvs() - pred
                for pred, err in zip(mean_prediction_denorm, recon_errorbar)
            ])
            
            log_reconstructed_flux = mean_prediction_denorm + point_specific_noise
            
            # 95% CI
            num_samples = 1000
            random_samples = np.array([
                st_scipy.norm(loc=0, scale=err).rvs(num_samples)
                for err in recon_errorbar
            ]).T
            
            jiggled_realizations = mean_prediction_denorm + random_samples
            lower_denorm = np.percentile(jiggled_realizations, 2.5, axis=0)
            upper_denorm = np.percentile(jiggled_realizations, 97.5, axis=0)
            
            end_time = time.time()
            
            # Plotting with EXACT format from document
            fig = plt.figure(figsize=(10, 6))
            
            # a) Plot original data with updated y-errors
            plt.errorbar(train_x_denorm, train_y_denorm, zorder=4, 
                        yerr=[lower_err_log, upper_err_log], linestyle="")
            
            # b) Plot reconstructed points with synthetic error bars
            plt.errorbar(test_x_denorm, log_reconstructed_flux, linestyle='none', 
                        yerr=np.abs(recon_errorbar), marker='o', capsize=5, 
                        color='yellow', zorder=3, label="Reconstructed Points")
            
            # c) Scatter original observed points on top
            plt.scatter(train_x_denorm, train_y_denorm, zorder=5, label="Observed Points")
            
            # d) Plot the mean prediction curve
            plt.plot(test_x_denorm, mean_prediction_denorm, label="Mean Prediction", zorder=2)
            
            # e) Add 95% confidence interval shading
            plt.fill_between(test_x_denorm.flatten(), lower_denorm, upper_denorm, 
                           alpha=0.5, color='orange', label="95% Confidence Region", zorder=1)
            
            plt.legend(loc='lower left')
            plt.xlabel('log$_{10}$(Time) (s)', fontsize=15)
            plt.ylabel('log$_{10}$(Flux) ($erg\\,cm^{-2}\\,s^{-1}$)', fontsize=15)
            plt.title(f'Attention U-Net on {grb_name}', fontsize=18)
            st.pyplot(fig)
            
            # --- UI FEATURE 5: High-Res Download ---
            fn = f"{grb_name}_attention_unet_plot.pdf"
            img = io.BytesIO()
            plt.savefig(img, format='pdf', bbox_inches='tight')
            st.download_button(label="📥 Download High-Res Plot (PDF)", data=img, file_name=fn, mime="application/pdf")
            
            # --- UI FEATURE 6: Detailed Metrics Dashboard ---
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            mse_val = history.history['loss'][-1]
            col1.metric("Final MSE", f"{mse_val:.6f}")
            col2.metric("Execution Time", f"{end_time - start_time:.2f} s")
            col3.metric("Reconstructed Points", len(test_x_denorm))
            
            # --- UI FEATURE 7: Training History ---
            with st.expander("📈 Training Loss Curve"):
                st.line_chart(history.history['loss'])
                
            # --- UI FEATURE 8: Residual Analysis ---
            with st.expander("📉 Residual Analysis"):
                residuals = train_y_denorm.flatten() - model.predict(X_train, verbose=0).flatten()
                fig_res, ax_res = plt.subplots(figsize=(10, 2))
                ax_res.scatter(train_x_denorm, residuals, alpha=0.6)
                ax_res.axhline(0, color='r', linestyle='--')
                ax_res.set_title("Residuals (Observed - Predicted)")
                st.pyplot(fig_res)
            
            # Build combined DataFrame
            combined_df = trimmed_data.copy(deep=True)
            new_rows = []
            for i in range(len(test_x_denorm)):
                logt_pt = test_x_denorm[i]
                t_lin = 10 ** logt_pt
                pos_t_lin = 10 ** (logt_pt + 0.01 * logt_pt)
                neg_t_lin = 10 ** (logt_pt - 0.01 * logt_pt)
                flux_lin = 10 ** log_reconstructed_flux[i]
                pos_f_lin = 10 ** (log_reconstructed_flux[i] + recon_errorbar[i])
                neg_f_lin = 10 ** (log_reconstructed_flux[i] - recon_errorbar[i])
                new_rows.append({
                    "t": t_lin,
                    "pos_t_err": abs(pos_t_lin - t_lin),
                    "neg_t_err": abs(t_lin - neg_t_lin),
                    "flux": flux_lin,
                    "pos_flux_err": abs(pos_f_lin - flux_lin),
                    "neg_flux_err": abs(flux_lin - neg_f_lin)
                })
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([combined_df, new_df], ignore_index=True)
            
            st.success(f"Reconstruction Complete for {grb_name}")
            st.caption(f"Final MSE: {history.history['loss'][-1]:.6f}")
            csv_buffer = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Combined Data (CSV)", data=csv_buffer, 
                             file_name=f"{grb_name}_attention_unet.csv", mime="text/csv")
            
            # --- UI FEATURE 9: Citation ---
            st.markdown("---")
            with st.expander("📚 Cite this Implementation"):
                st.code("""@article{GRBReconstruction2024,
  title={Multi-Model Light Curve Reconstruction for Gamma-Ray Bursts},
  author={Research Team},
  journal={Astrophysical Data Analysis},
  year={2024}
}""", language="tex")

    train_attention_unet()


elif run_paper_btn and paper_model_select == "Model 2: Quadratic Smoothing Spline (QSS)":
    st.subheader("Paper Model 2: Quartic Smoothing Spline on GRB 231210B")
    
    with st.expander("📄 Methodology Overview", expanded=True):
        st.markdown("""
        **Abstract:** This model applies a Quartic (k=4) Smoothing Spline to approximate the GRB light curve. 
        It balances the trade-off between fitting the data points and maintaining curve smoothness, controlled by the smoothing factor $s$.
        """)
    
    def train_qss():
        with st.spinner("Loading data and training Quartic Smoothing Spline..."):
            start_time = time.time()
            # --- 1. Load Data (Streamlit specific adaptation) ---
            try:
                if "username/repo" in dataset_url:
                    st.warning("Using placeholder URL. Please update the Dataset URL in the sidebar.")
                    # Placeholder data generation for demo
                    t_mock = np.logspace(1, 5, 50)
                    f_mock = 1e-10 * (t_mock**-1.5) * (1 + 0.1*np.random.randn(50))
                    trimmed_data = pd.DataFrame({
                        't': t_mock, 'flux': f_mock, 
                        'pos_flux_err': 0.1*f_mock, 'neg_flux_err': 0.1*f_mock,
                        'pos_t_err': 0.01*t_mock, 'neg_t_err': 0.01*t_mock
                    })
                else:
                    response = requests.get(dataset_url)
                    if response.status_code == 200:
                        trimmed_data = pd.read_csv(io.StringIO(response.text))
                    else:
                        st.error(f"Failed to load data: HTTP {response.status_code}")
                        return
            except Exception as e:
                st.error(f"Error: {e}")
                return

            with st.expander("🔍 Inspect Input Data"):
                st.dataframe(trimmed_data.head(), use_container_width=True)
                st.caption(f"Loaded {len(trimmed_data)} rows.")

            grb_name = "GRB231210B"
            
            # --- 2. Data Preprocessing (Matching run.py logic) ---
            # Standardize column names
            if len(trimmed_data.columns) == 6:
                trimmed_data.columns = ["t", "pos_t_err", "neg_t_err", "flux", "pos_flux_err", "neg_flux_err"]
            elif len(trimmed_data.columns) >= 7:
                 # Handle potential index column
                trimmed_data.columns = ["0", "t", "pos_t_err", "neg_t_err", "flux", "pos_flux_err", "neg_flux_err"][:len(trimmed_data.columns)]
            
            # Sort and extract
            trimmed_data = trimmed_data.sort_values(by="t")
            ts = trimmed_data["t"].to_numpy()
            fluxes = trimmed_data["flux"].to_numpy()
            
            # Extract errors (Robust handling for missing error columns)
            if "pos_t_err" in trimmed_data.columns:
                positive_ts_err = trimmed_data["pos_t_err"].to_numpy()
                negative_ts_err = trimmed_data["neg_t_err"].to_numpy()
                positive_fluxes_err = trimmed_data["pos_flux_err"].to_numpy()
                negative_fluxes_err = trimmed_data["neg_flux_err"].to_numpy()
            else:
                # Fallback if specific error columns are missing
                positive_ts_err = 0.01 * ts
                negative_ts_err = 0.01 * ts
                positive_fluxes_err = 0.1 * fluxes
                negative_fluxes_err = 0.1 * fluxes

            # Filter valid data
            mask = (ts > 0) & (fluxes > 0)
            ts, fluxes = ts[mask], fluxes[mask]
            positive_ts_err, negative_ts_err = positive_ts_err[mask], negative_ts_err[mask]
            positive_fluxes_err, negative_fluxes_err = positive_fluxes_err[mask], negative_fluxes_err[mask]

            # Log transformation
            log_ts = np.log10(ts)
            log_fluxes = np.log10(fluxes)

            # Log-scale errors calculation
            pos_fluxes = fluxes + positive_fluxes_err
            neg_fluxes = fluxes + negative_fluxes_err
            lower_err_log = log_fluxes - np.log10(neg_fluxes)
            upper_err_log = np.log10(pos_fluxes) - log_fluxes
            
            # Synthetic sampling error prep
            ts_err = (positive_ts_err - negative_ts_err) / 2.0
            flux_err = (positive_fluxes_err - negative_fluxes_err) / 2.0
            log_ts_err = ts_err / (ts * np.log(10))
            log_flux_err = flux_err / (fluxes * np.log(10))

            # --- 3. Normalization ---
            log_ts_mean = np.mean(log_ts)
            log_ts_std = np.std(log_ts)
            log_flux_mean = np.mean(log_fluxes)
            log_flux_std = np.std(log_fluxes)

            log_ts_norm = (log_ts - log_ts_mean) / log_ts_std
            log_flux_norm = (log_fluxes - log_flux_mean) / log_flux_std

            # --- 4. Gap-Aware Grid Construction ---
            min_gap = 0.05
            recon_log_t = [log_ts[0]]
            total_span = log_ts[-1] - log_ts[0]
            
            # Dynamic density based on dataset size
            if len(ts) > 500: fraction = 0.05
            elif len(ts) > 250: fraction = 0.1
            elif len(ts) > 100: fraction = 0.3
            else: fraction = 0.4
            n_points = max(20, int(fraction * len(ts)))

            for i in range(len(ts) - 1):
                gap_size = log_ts[i+1] - log_ts[i]
                if gap_size > min_gap:
                    interval_points = max(2, int(n_points * gap_size / total_span))
                    interval = np.linspace(log_ts[i], log_ts[i+1], interval_points, endpoint=True)
                    recon_log_t.extend(interval[1:])
            
            recon_log_t = np.array(recon_log_t)
            recon_t = 10**recon_log_t
            recon_t = np.unique(recon_t)
            log_recon_t = np.log10(recon_t).reshape(-1, 1)

            # --- 5. Spline Fitting (Quartic k=4) ---
            N = len(log_ts_norm)
            spline = UnivariateSpline(
                x=log_ts_norm.flatten(),
                y=log_flux_norm.flatten(),
                k=conf_k,   # Configurable degree
                s=N * conf_s_mult    # Configurable Smoothing factor
            )

            # Residuals for CI
            pred_norm_train = spline(log_ts_norm.flatten())
            resid_norm = log_flux_norm.flatten() - pred_norm_train
            sigma_resid = np.std(resid_norm)

            # Expand grid for large gaps (filling holes)
            expanded = log_recon_t.copy()
            for i in range(len(log_ts) - 1):
                lowb = log_ts[i]
                upb = log_ts[i + 1]
                if np.abs(upb - lowb) >= 0.1:
                    n_pts = min(5, int(5 * np.abs(upb - lowb) / 0.1))
                    segment = np.linspace(lowb, upb, num=n_pts).reshape(-1, 1)
                    expanded = np.vstack((expanded, segment))
            expanded = np.sort(expanded, axis=0)

            # Normalize expanded grid
            expanded_norm = ((expanded - log_ts_mean) / log_ts_std).flatten()

            # --- 6. Prediction & Error Simulation ---
            mean_norm_recon = spline(expanded_norm)
            
            # 95% Confidence Interval
            lower_norm_recon = mean_norm_recon - 1.96 * sigma_resid
            upper_norm_recon = mean_norm_recon + 1.96 * sigma_resid

            # Denormalize
            mean_denorm_log = (mean_norm_recon * log_flux_std) + log_flux_mean
            lower_denorm_log = (lower_norm_recon * log_flux_std) + log_flux_mean
            upper_denorm_log = (upper_norm_recon * log_flux_std) + log_flux_mean

            # Fit distribution to errors (Norm vs Laplace)
            logfluxerrs = (positive_fluxes_err - negative_fluxes_err) / (2 * fluxes * np.log(10))
            distributions = [st_scipy.norm, st_scipy.laplace]
            fits = {}
            for dist in distributions:
                params = dist.fit(logfluxerrs)
                loglikelihood = np.sum(dist.logpdf(logfluxerrs, *params))
                fits[dist.name] = (params, loglikelihood)
            
            best_dist_name = max(fits, key=lambda d: fits[d][1])
            best_params = fits[best_dist_name][0]
            best_dist = getattr(st_scipy, best_dist_name)

            # Generate synthetic noise
            rand_noise = []
            for j in range(len(mean_norm_recon)):
                noise = 3.5 * (best_dist.rvs(*best_params, size=1)[0] - best_params[0])
                rand_noise.append(noise)
            rand_noise = np.array(rand_noise)

            recon_norm_flux = mean_norm_recon + rand_noise
            recon_denorm_log = (recon_norm_flux * log_flux_std) + log_flux_mean

            # Sample synthetic error bars
            loc_f, scale_f = st_scipy.norm.fit(log_flux_err)
            sampled_flux_errs = st_scipy.norm(loc=loc_f, scale=scale_f).rvs(size=len(expanded))
            
            loc_t, scale_t = st_scipy.norm.fit(log_ts_err)
            sampled_time_errs = st_scipy.norm(loc=loc_t, scale=scale_t).rvs(size=len(expanded))

            end_time = time.time()

            # --- 7. Plotting (Streamlit adaptation) ---
            # Prepare plotting variables
            test_x_denorm = expanded.flatten()
            log_reconstructed_flux = recon_denorm_log.flatten()
            
            fig = plt.figure(figsize=(10, 6))
            
            # a) Original data
            plt.errorbar(log_ts, log_fluxes, yerr=[lower_err_log, upper_err_log], 
                        zorder=4, linestyle="", fmt='none', ecolor='gray')
            
            # b) Reconstructed points
            plt.errorbar(test_x_denorm, log_reconstructed_flux, 
                        yerr=np.abs(sampled_flux_errs), linestyle='none', 
                        marker='o', capsize=5, color='yellow', zorder=3, 
                        label="Reconstructed Points")
            
            # c) Observed points scatter
            plt.scatter(log_ts, log_fluxes, zorder=5, label="Observed Points", color='blue')
            
            # d) Mean prediction curve
            plt.plot(test_x_denorm, mean_denorm_log, label="Mean Prediction", zorder=2, color='green')
            
            # e) 95% Confidence Region
            plt.fill_between(test_x_denorm, lower_denorm_log, upper_denorm_log, 
                           alpha=0.5, color='orange', label="95% Confidence Region", zorder=1)
            
            plt.legend(loc='lower left')
            plt.xlabel('log$_{10}$(Time) (s)', fontsize=15)
            plt.ylabel('log$_{10}$(Flux) ($erg\\,cm^{-2}\\,s^{-1}$)', fontsize=15)
            plt.title(f'Quartic Smoothing Spline on {grb_name}', fontsize=18)
            
            st.pyplot(fig)

            # --- UI Features for QSS ---
            fn = f"{grb_name}_qss_plot.pdf"
            img = io.BytesIO()
            plt.savefig(img, format='pdf', bbox_inches='tight')
            st.download_button(label="📥 Download High-Res Plot (PDF)", data=img, file_name=fn, mime="application/pdf")

            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Residual Std Dev", f"{sigma_resid:.6f}")
            col2.metric("Execution Time", f"{end_time - start_time:.2f} s")
            col3.metric("Smoothing Factor (s)", f"{N * conf_s_mult:.1f}")

            with st.expander("📉 Residual Analysis"):
                fig_res, ax_res = plt.subplots(figsize=(10, 2))
                ax_res.scatter(log_ts_norm.flatten(), resid_norm, alpha=0.6, color='green')
                ax_res.axhline(0, color='r', linestyle='--')
                ax_res.set_title("Normalized Residuals")
                st.pyplot(fig_res)

            # --- 8. Export Data ---
            combined_df = trimmed_data.copy(deep=True)
            new_rows = []
            for i in range(len(expanded)):
                logt_pt = expanded[i][0]
                t_lin = 10 ** logt_pt
                
                # Synthetic linear errors
                pos_t_lin = 10 ** (logt_pt + sampled_time_errs[i])
                neg_t_lin = 10 ** (logt_pt - sampled_time_errs[i])
                
                flux_lin = 10 ** recon_denorm_log[i]
                pos_f_lin = 10 ** (recon_denorm_log[i] + sampled_flux_errs[i])
                neg_f_lin = 10 ** (recon_denorm_log[i] - sampled_flux_errs[i])
                
                new_rows.append({
                    "t": t_lin,
                    "pos_t_err": abs(pos_t_lin - t_lin),
                    "neg_t_err": abs(t_lin - neg_t_lin),
                    "flux": flux_lin,
                    "pos_flux_err": abs(pos_f_lin - flux_lin),
                    "neg_flux_err": abs(flux_lin - neg_f_lin)
                })
            
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([combined_df, new_df], ignore_index=True)
            
            st.success(f"Reconstruction Complete for {grb_name}")
            csv_buffer = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Combined Data (CSV)", data=csv_buffer, 
                             file_name=f"{grb_name}_quartic_spline.csv", mime="text/csv")
            
            st.markdown("---")
            with st.expander("📚 Cite this Implementation"):
                st.code("""@article{GRBReconstruction2024,
  title={Spline-Based Approaches for GRB Light Curve Reconstruction},
  author={Research Team},
  journal={Astrophysical Data Analysis},
  year={2024}
}""", language="tex")
    
    train_qss()
