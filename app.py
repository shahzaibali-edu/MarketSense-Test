import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os
import logging
import warnings

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MarketSense Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# USER CONFIGURATION
# ==========================================
# 1. Market File Paths (Updated for relative paths for GitHub/Streamlit)
MARKET_CONFIG = {
    "Liaquat Bazar":  "Liaquat Bazar.csv",
    "Chiltan Market": "Chiltan Market.csv",
    "Khezi":          "Khezi.csv",
    "NATO Market":    "NATO Market.csv",
    "Jail Road":      "Jail Road.csv"
}

# 2. Market Coordinates (Quetta Specific)
MARKET_COORDS = {
    "Liaquat Bazar":  {"lat": 30.192269147109428, "lon": 67.01483853926425},
    "Chiltan Market": {"lat": 30.211448062218803, "lon": 67.03869845865138},
    "Khezi":          {"lat": 30.230811105583122, "lon": 66.95533627290733},
    "NATO Market":    {"lat": 30.17421295100368,  "lon": 66.99944955090552},
    "Jail Road":      {"lat": 30.209756098461632, "lon": 66.99708899462671}
}

# Default Weather Location (Quetta)
LOCATION_LAT = 30.1798
LOCATION_LON = 66.9750

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ==========================================
# CORE CLASSES
# ==========================================

class WeatherService:
    def get_weather_data(self, start_date, end_date):
        """Fetches daily rain sum from Open-Meteo."""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LOCATION_LAT,
            "longitude": LOCATION_LON,
            "start_date": start_str,
            "end_date": end_str,
            "daily": "rain_sum",
            "timezone": "auto"
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if 'daily' not in data:
                return self._generate_dry_weather(start_date, end_date)

            dates = pd.to_datetime(data['daily']['time'])
            rain_sum = np.array(data['daily']['rain_sum'])
            
            # If rain > 2.0mm, it's a "Rainy Day" (1), else 0
            rain_binary = (pd.Series(rain_sum).fillna(0) > 2.0).astype(int)

            return pd.DataFrame({'ds': dates, 'rain': rain_binary})

        except Exception as e:
            st.warning(f"Weather API failed: {e}. Using defaults.")
            return self._generate_dry_weather(start_date, end_date)

    def _generate_dry_weather(self, start, end):
        dates = pd.date_range(start, end)
        return pd.DataFrame({'ds': dates, 'rain': 0})

class DataHandler:
    def __init__(self):
        self.weather_service = WeatherService()

    # Cache this to prevent reloading on every interaction
    @st.cache_data(ttl=600) 
    def load_local_file(_self, market_name):
        file_path = MARKET_CONFIG.get(market_name)

        if not file_path or not os.path.exists(file_path):
             return None

        try:
            try:
                df = pd.read_csv(file_path)
                if len(df.columns) < 2:
                    df = pd.read_csv(file_path, sep=';')
            except Exception:
                df = pd.read_csv(file_path, sep=None, engine='python')

            df.columns = df.columns.str.strip().str.lower()
            
            date_col = next((c for c in df.columns if 'date' in c or 'time' in c or 'ds' in c), None)
            if not date_col: raise ValueError(f"Could not find 'date' column.")

            try:
                df['ds'] = pd.to_datetime(df[date_col], dayfirst=True)
            except Exception:
                df['ds'] = pd.to_datetime(df[date_col], errors='coerce')

            df['ds'] = df['ds'].dt.normalize()
            df = df.dropna(subset=['ds'])

            y_col = next((c for c in df.columns if 'footfall' in c or 'count' in c or 'visit' in c or 'y' == c), None)
            if not y_col: raise ValueError(f"Could not find 'footfall' column.")

            df['y'] = pd.to_numeric(df[y_col], errors='coerce')
            df['y'] = df['y'].clip(lower=0)
            df = df.dropna(subset=['y'])

            trans_col = next((c for c in df.columns if 'trans' in c or 'sale' in c), None)
            if trans_col:
                df['transactions'] = pd.to_numeric(df[trans_col], errors='coerce').fillna(0)
            else:
                df['transactions'] = (df['y'] * 0.1).astype(int)

            df = df.sort_values('ds').reset_index(drop=True)
            df['market'] = market_name
            df['road_closure'] = 0

            # Fetch weather (uncached inside the cached function is fine)
            weather_df = _self.weather_service.get_weather_data(
                df['ds'].min(), df['ds'].max()
            )

            df = pd.merge(df, weather_df, on='ds', how='left').fillna(0)
            return df[['ds', 'y', 'transactions', 'rain', 'road_closure', 'market']]

        except Exception as e:
            st.error(f"CSV Load Error ({market_name}): {str(e)}")
            return None

class MarketAnalyzer:
    def __init__(self):
        self.model = None

    def _fallback_forecast(self, df, forecast_days=30):
        df['day_idx'] = np.arange(len(df))
        df['dow'] = df['ds'].dt.dayofweek
        X = df[['day_idx']].values
        y = df['y'].values
        reg = LinearRegression().fit(X, y)
        trend = reg.predict(X)
        residuals = y - trend
        dow_seasonality = df.copy()
        dow_seasonality['resid'] = residuals
        seasonal_map = dow_seasonality.groupby('dow')['resid'].mean().to_dict()
        last_idx = df['day_idx'].iloc[-1]
        future_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1), periods=forecast_days)
        future_indices = np.arange(last_idx + 1, last_idx + 1 + forecast_days)
        future_dow = future_dates.dayofweek
        future_trend = reg.predict(future_indices.reshape(-1, 1))
        future_seasonality = np.array([seasonal_map.get(d, 0) for d in future_dow])
        future_yhat = future_trend + future_seasonality
        history_yhat = trend + np.array([seasonal_map.get(d, 0) for d in df['dow']])
        full_dates = pd.concat([df['ds'], pd.Series(future_dates)]).reset_index(drop=True)
        full_yhat = np.concatenate([history_yhat, future_yhat])
        std_resid = np.std(residuals)
        result = pd.DataFrame({'ds': full_dates, 'yhat': full_yhat, 'yhat_lower': full_yhat - 1.96*std_resid, 'yhat_upper': full_yhat + 1.96*std_resid})
        cols_to_keep = [c for c in ['ds', 'y', 'transactions', 'rain', 'road_closure'] if c in df.columns]
        result = pd.merge(result, df[cols_to_keep], on='ds', how='left')

        result = result.sort_values('ds').reset_index(drop=True)
        result['y'] = result['y'].astype(float)
        last_real_date = df['ds'].max()
        result.loc[result['ds'] > last_real_date, 'y'] = np.nan
        return result

    # Cache the training as it is expensive
    @st.cache_data(show_spinner=True)
    def train_predict(_self, df, forecast_days=30):
        try:
            from prophet import Prophet
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.5
            )

            if 'rain' in df.columns: model.add_regressor('rain')
            if 'road_closure' in df.columns: model.add_regressor('road_closure')

            model.fit(df)

            future = model.make_future_dataframe(periods=forecast_days)
            future['rain'] = 0
            if 'rain' in df.columns:
                 future.iloc[:len(df), list(future.columns).index('rain')] = df['rain']
            future['road_closure'] = 0

            forecast = model.predict(future)

            # --- HYBRID REACTIVE OVERRIDE ---
            ema_series = df['y'].ewm(span=3, adjust=False).mean()
            n_hist = len(df)
            reactive_yhat = forecast['yhat'].copy()
            reactive_yhat.iloc[:n_hist] = (forecast['yhat'].iloc[:n_hist] * 0.3) + (ema_series * 0.7)

            last_prophet_val = forecast['yhat'].iloc[n_hist-1]
            last_reactive_val = reactive_yhat.iloc[n_hist-1]
            gap_offset = last_reactive_val - last_prophet_val

            reactive_yhat.iloc[n_hist:] = forecast['yhat'].iloc[n_hist:] + gap_offset
            forecast['yhat'] = reactive_yhat
            forecast['yhat_lower'] = forecast['yhat_lower'] + gap_offset
            forecast['yhat_upper'] = forecast['yhat_upper'] + gap_offset

            result = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')

            result = result.sort_values('ds').reset_index(drop=True)
            result['y'] = result['y'].astype(float)
            last_real_date = df['ds'].max()
            result.loc[result['ds'] > last_real_date, 'y'] = np.nan

            return result

        except Exception as e:
            st.warning(f"Prophet Error: {e}. Switching to Fallback.")
            return _self._fallback_forecast(df, forecast_days)

    def detect_anomalies(self, result_df):
        result_df['is_anomaly'] = False
        result_df['anomaly_score'] = 0.0
        mask = result_df['y'].notna()
        anomaly_mask = mask & (result_df['y'] < result_df['yhat_lower'])
        threshold_mask = mask & (result_df['y'] < (result_df['yhat'] * 0.75))
        final_mask = anomaly_mask | threshold_mask

        result_df.loc[final_mask, 'is_anomaly'] = True
        safe_yhat = result_df['yhat'].replace(0, 1)
        result_df.loc[final_mask, 'anomaly_score'] = ((result_df['yhat_lower'] - result_df['y']) / safe_yhat) * 100
        return result_df

    def calculate_pulse_score(self, result_df, lookback=7):
        recent = result_df[result_df['y'].notna()].tail(lookback)
        if len(recent) == 0: return 100, "Insufficient Data"

        base_score = 100
        reasons = []

        anomaly_count = recent['is_anomaly'].sum()
        if anomaly_count > 0:
            days_in_view = len(recent)
            density_penalty = (anomaly_count / days_in_view) * 350
            fixed_penalty = anomaly_count * 2
            total_penalty = density_penalty + fixed_penalty
            base_score -= total_penalty
            reasons.append(f"{anomaly_count} Anomalies Detected")

        if len(recent) >= 6:
            curr_avg = recent['y'].iloc[-3:].mean()
            prev_avg = recent['y'].iloc[-6:-3].mean()
            if curr_avg < prev_avg:
                drop_pct = (prev_avg - curr_avg) / prev_avg
                base_score -= (drop_pct * 100)
                reasons.append(f"Negative Trend (-{int(drop_pct*100)}%)")

        final_score = max(0, int(base_score))
        reason_str = "Market Healthy" if final_score >= 90 else "Stable"
        if reasons:
            reason_str = ", ".join(reasons)

        return final_score, reason_str

class PolicyEngine:
    @staticmethod
    def suggest_action(pulse_score, recent_df, weather_rain_recent):
        avg_footfall = recent_df['y'].mean()
        if 'transactions' in recent_df.columns:
            avg_trans = recent_df['transactions'].mean()
        else:
            avg_trans = 0

        conversion = avg_trans / avg_footfall if avg_footfall > 0 else 0
        suggestions = []

        if pulse_score < 50:
            suggestions.append(("Emergency Grant", "Score Critical (<50). Immediate liquidity needed."))
        if recent_df['is_anomaly'].iloc[-1]:
            if weather_rain_recent == 1:
                return [("No Action (Weather)", "Drop likely due to rain. Wait for weather to clear.")]
            else:
                suggestions.append(("Security/Access Check", "Unexplained drop. Check for access blocks."))
        if conversion < 0.08:
            suggestions.append(("Inventory Restock / Promo", "Footfall high, Sales low. Improve offer."))
        if not suggestions:
            suggestions.append(("Maintain", "Market healthy."))
        return suggestions

    @staticmethod
    def simulate_recovery(forecast_df, action_type, start_date_idx):
        simulated_yhat = forecast_df['yhat'].copy()
        n = len(simulated_yhat)
        horizon = n - start_date_idx
        if horizon <= 0: return simulated_yhat

        history_start = max(0, start_date_idx - 20)
        history_end = max(1, start_date_idx - 2)
        target_level = forecast_df['y'].iloc[history_start:history_end].mean()
        if pd.isna(target_level): target_level = forecast_df['y'].mean()

        current_forecast_level = simulated_yhat.iloc[start_date_idx]
        gap = target_level - current_forecast_level if current_forecast_level < target_level else current_forecast_level * 0.1

        if action_type == "Micro-loan (Linear Recovery)":
            ramp = np.linspace(0, 1.0, min(14, horizon))
            if horizon > 14: ramp = np.concatenate([ramp, np.full(horizon-14, 1.0)])
            simulated_yhat.iloc[start_date_idx:] += (gap * ramp)
        elif action_type == "Marketing Push (Spike)":
            curve = (np.exp(-np.arange(horizon) / 4) * 1.5)
            simulated_yhat.iloc[start_date_idx:] += (gap * curve) + (gap * 0.8)
        elif action_type == "Infrastructure Fix (Step)":
            simulated_yhat.iloc[start_date_idx:] += gap

        return simulated_yhat


# ==========================================
# STREAMLIT UI LOGIC
# ==========================================

handler = DataHandler()
analyzer = MarketAnalyzer()
policy = PolicyEngine()

# --- Sidebar Controls ---
st.sidebar.title("Configuration")
selected_market = st.sidebar.selectbox("Select Market / Location", list(MARKET_CONFIG.keys()))
lookback_days = st.sidebar.slider("Lookback Days", 7, 90, 30)
st.sidebar.markdown("### 🛠️ Policy Simulation")
sim_action = st.sidebar.selectbox("Simulate Intervention", ["None", "Micro-loan (Linear Recovery)", "Marketing Push (Spike)", "Infrastructure Fix (Step)"])

# --- Main Dashboard ---
st.title("📊 MarketSense / EchoPulse Dashboard")

# Create Tabs
tab1, tab2 = st.tabs(["Forecast & Simulation", "City Pulse Map"])

with tab1:
    # 1. Load Data
    df = handler.load_local_file(selected_market)

    if df is None:
        st.error(f"Could not find file for '{selected_market}'. Please ensure the CSV file is uploaded to the repository.")
    else:
        df_clean = df.sort_values('ds').reset_index(drop=True)

        # 2. Train & Forecast
        with st.spinner("Analyzing market data..."):
            res_df = analyzer.train_predict(df_clean, forecast_days=30)
            res_df = analyzer.detect_anomalies(res_df)

        # 3. Metrics
        current_date_idx = len(df_clean) - 1
        lookback_slice = res_df.iloc[max(0, current_date_idx - lookback_days) : current_date_idx + 1]
        score, reason = analyzer.calculate_pulse_score(res_df, lookback=lookback_days)

        # 4. Suggestions
        last_rain_val = df_clean['rain'].iloc[-1] if 'rain' in df_clean.columns else 0
        suggestions = policy.suggest_action(score, lookback_slice, last_rain_val)
        suggested_action_text, suggested_reason_text = suggestions[0]

        # 5. Simulation
        sim_yhat = None
        if sim_action != "None":
            sim_yhat = policy.simulate_recovery(res_df, sim_action, current_date_idx)

        # --- DISPLAY CARDS ---
        col1, col2, col3 = st.columns(3)

        # Color Logic
        score_color = "#22c55e" if score > 80 else "#f97316" if score > 50 else "#ef4444"
        weather_text = "🌧️ Rain" if last_rain_val == 1 else "☀️ Clear"
        weather_bg = "#3b82f6" if last_rain_val == 1 else "#f59e0b"

        with col1:
            st.markdown(f"""
            <div style="background-color: {score_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
                <h2 style="margin:0; font-size: 40px; color: white;">{score}</h2>
                <p style="margin:0; opacity: 0.9">Pulse Score</p>
                <hr style="margin: 10px 0; border-color: rgba(255,255,255,0.3);">
                <p style="margin:0; font-size: 14px;">{reason}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background-color: {weather_bg}; padding: 20px; border-radius: 10px; text-align: center; color: white; height: 100%;">
                <h3 style="margin:0; font-size: 24px; color: white;">{weather_text}</h3>
                <p style="margin:10px 0 0 0; font-size: 14px; opacity: 0.9">Latest Conditions</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: white; text-align: center; height: 100%; color: #333;">
                <strong style="font-size: 16px; display: block; margin-bottom: 5px;">💡 {suggested_action_text}</strong>
                <p style="margin: 0; font-size: 13px; color: #666;">{suggested_reason_text}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- PLOTTING ---
        fig = go.Figure()

        # Observed
        fig.add_trace(go.Scatter(
            x=res_df['ds'], y=res_df['y'],
            mode='lines', name='Observed Footfall',
            connectgaps=False, line=dict(color='white', width=2)
        ))

        # Expected
        fig.add_trace(go.Scatter(
            x=res_df['ds'], y=res_df['yhat'],
            mode='lines', name='Expected (Baseline)', line=dict(color='blue', dash='dash')
        ))

        # Confidence
        fig.add_trace(go.Scatter(x=res_df['ds'], y=res_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=res_df['ds'], y=res_df['yhat_lower'], mode='lines', line=dict(width=0),
                                 fill='tonexty', fillcolor='rgba(0, 0, 255, 0.1)', name='95% Confidence'))

        # Anomalies
        anomalies = res_df[res_df['is_anomaly'] == True]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ds'], y=anomalies['y'], mode='markers',
                                     name='Anomaly Detected', marker=dict(color='red', size=10, symbol='x')))

        # Simulation
        if sim_yhat is not None:
            fig.add_trace(go.Scatter(x=res_df['ds'].iloc[current_date_idx:], y=sim_yhat.iloc[current_date_idx:],
                                     mode='lines', name=f'Simulated: {sim_action}', line=dict(color='green', width=3)))

        # Zoom
        start_display = res_df['ds'].iloc[current_date_idx] - timedelta(days=lookback_days)
        fig.update_xaxes(range=[start_display, res_df['ds'].iloc[-1]])

        fig.update_layout(
            title=f"MarketSense: {selected_market}",
            hovermode="x unified",
            height=500,
            margin=dict(l=20, r=20, t=40, b=80),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top")
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🗺️ City Pulse Map")
    
    @st.cache_data(ttl=600)
    def generate_map_data(lookback):
        market_data = []
        for name, path in MARKET_CONFIG.items():
            df = handler.load_local_file(name)
            status = "No Data"
            score = 0
            color = "gray"

            if df is not None and not df.empty:
                res_df = analyzer.train_predict(df, forecast_days=5)
                res_df = analyzer.detect_anomalies(res_df)
                score, _ = analyzer.calculate_pulse_score(res_df, lookback=lookback)

                if score > 80: color = "#22c55e" # Green
                elif score > 50: color = "#f97316" # Orange
                else: color = "#ef4444" # Red
                status = f"Score: {score}"

            coords = MARKET_COORDS.get(name, {"lat": LOCATION_LAT, "lon": LOCATION_LON})
            market_data.append({
                "Name": name, "Lat": coords["lat"], "Lon": coords["lon"],
                "Score": score, "Color": color, "Status": status
            })
        return pd.DataFrame(market_data)

    if st.button("Refresh Map") or True: # Auto-run on load
        with st.spinner("Generating detailed map..."):
            map_df = generate_map_data(lookback_days)
            
            fig_map = px.scatter_mapbox(
                map_df, lat="Lat", lon="Lon", hover_name="Name",
                hover_data={"Status": True, "Lat": False, "Lon": False, "Color": False, "Score": False},
                color="Color", color_discrete_map="identity",
                zoom=12, height=600, center={"lat": 30.1798, "lon": 66.9750}
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":0,"l":0,"b":0},
                showlegend=False
            )
            fig_map.update_traces(marker=dict(size=45, opacity=0.6))
            st.plotly_chart(fig_map, use_container_width=True)
