import streamlit as st
import pandas as pd
# SHAP is optional for deployment
try:
    import shap
except ImportError:
    shap = None
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from datetime import datetime
import random
import joblib
import os

st.set_page_config(
    page_title="Smart Solar Battery Health Monitoring",
    page_icon="🔋",
    layout="wide"
)

# LOAD ML MODEL
model = joblib.load("battery_health_model.pkl")

# HEADER
st.title("🔋 Smart Solar Battery Health Monitoring")
st.caption("Real-Time Monitoring • Machine Learning Prediction • Battery Analytics")

st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.success(
        "🟢 SYSTEM ONLINE\n\n"
        "Monitoring system is active."
    )

with c2:
    st.info(
        "🤖 ML MODEL ACTIVE\n\n"
        "Random Forest prediction enabled."
    )

with c3:
    st.info(
        "📊 MONITORING ACTIVE\n\n"
        "Battery parameters are being monitored."
    )
# SIDEBAR
st.sidebar.title("⚙️ Control Panel")

# INPUT MODE

mode = st.sidebar.radio(
    "Input Mode",
    [
        "🎲 Simulation Mode",
        "✍️ Manual Input Mode",
        "🔌 Hardware Mode"
    ]
)


# SIMULATION MODE

if mode == "🎲 Simulation Mode":

    if st.sidebar.button("🔄 Generate New Reading"):
        st.rerun()

    voltage = round(
        random.uniform(10.5, 13.0),
        2
    )

    current = round(
        random.uniform(-2.0, 2.0),
        2
    )

    temperature = round(
        random.uniform(20, 50),
        1
    )

    soc = random.randint(
        5,
        100
    )


# MANUAL INPUT MODE

elif mode == "✍️ Manual Input Mode":

    st.sidebar.subheader(
        "Battery Parameters"
    )

    voltage = st.sidebar.number_input(
        "Battery Voltage (V)",
        min_value=0.0,
        max_value=20.0,
        value=12.5,
        step=0.1
    )

    current = st.sidebar.number_input(
        "Battery Current (A)",
        min_value=-10.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    temperature = st.sidebar.number_input(
        "Battery Temperature (°C)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.5
    )

    soc = st.sidebar.slider(
        "State of Charge (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    # =========================================================
    # V28.3 - INPUT DATA VALIDATION
    # =========================================================

    input_valid = True

    if voltage <= 0:
        st.sidebar.error(
            "⚠️ Battery voltage must be greater than 0 V."
        )
        input_valid = False

    if current < 0:
        st.sidebar.warning(
            "⚠️ Negative current indicates charging/reverse current."
        )

    if temperature < 0 or temperature > 60:
        st.sidebar.error(
            "⚠️ Battery temperature is outside the safe input range."
        )
        input_valid = False

    if soc < 0 or soc > 100:
        st.sidebar.error(
            "⚠️ State of Charge must be between 0% and 100%."
        )
        input_valid = False# HARDWARE MODE

else:

    st.sidebar.subheader(
        "🔌 Hardware Connection"
    )

    st.sidebar.info(
        "Waiting for ESP32 sensor data."
    )

    # Simulated live hardware data
    hardware_data = {
        "voltage": round(random.uniform(10.5, 13.0), 2),
        "current": round(random.uniform(-2.0, 2.0), 2),
        "temperature": round(random.uniform(20.0, 50.0), 1),
        "soc": random.randint(5, 100)
    }

    # Read values from hardware data
    voltage = hardware_data["voltage"]
    current = hardware_data["current"]
    temperature = hardware_data["temperature"]
    soc = hardware_data["soc"]

    # Hardware connection status
    hardware_status = "Demo Mode"

    st.sidebar.success(
        f"🟢 Hardware Status: {hardware_status}"
    )

    st.sidebar.caption(
        "ESP32 live sensor integration will be added next."
    )

# =========================================================
# V29.2 - SAFE ML INPUT PREPARATION
# =========================================================

feature_names = [
    "Voltage",
    "Current",
    "Temperature",
    "SOC"
]

input_values = [
    voltage,
    current,
    temperature,
    soc
]

input_data = pd.DataFrame(
    [input_values],
    columns=feature_names
)

# Ensure the model receives the exact feature order
input_data = input_data[
    feature_names
]
# =========================================================
# V29.1 - SAFE ML PREDICTION
# =========================================================

prediction = "Prediction Error"
confidence = 0.0

try:

    prediction_result = model.predict(input_data)

    if len(prediction_result) > 0:

        prediction = prediction_result[0]

        try:

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = float(
                max(probabilities) * 100
            )

        except Exception:

            confidence = 0.0

    else:

        st.warning(
            "⚠️ ML model returned no prediction."
        )

except Exception as e:

    st.error(
        f"⚠️ ML prediction error: {e}"
    )
# =========================================================
# V31 - BATTERY HEALTH SCORE
# =========================================================

health_scores = {
    "Healthy": 90,
    "Needs Attention": 60,
    "Critical": 25,
    "Prediction Error": 0
}

health_score = health_scores.get(prediction, 0)
# =========================================================
# V31.4 - SMART BATTERY ALERTS
# =========================================================

alerts = []

if temperature >= 42:
    alerts.append("🌡️ High temperature detected.")

if voltage < 11.5:
    alerts.append("🔋 Low battery voltage detected.")

if soc < 20:
    alerts.append("⚠️ Battery SOC is critically low.")

if not alerts:
    alerts.append("✅ No critical battery alerts detected.")
# =========================================================
# V31.4 - DISPLAY SMART ALERTS
# =========================================================

st.subheader("🚨 Smart Alerts")

for alert in alerts:
    if "No critical" in alert:
        st.success(alert)
    else:
        st.warning(alert)
# =========================================================
# V31 - SMART BATTERY RECOMMENDATION
# =========================================================

if prediction == "Healthy":
    recommendation = (
        "Battery is operating normally. "
        "Continue regular monitoring."
    )

elif prediction == "Needs Attention":
    recommendation = (
        "Battery needs attention. "
        "Check charging conditions and monitor it closely."
    )

elif prediction == "Critical":
    recommendation = (
        "Battery condition is critical. "
        "Inspect the system before continuing normal operation."
    )

else:
    recommendation = (
        "Unable to generate a battery recommendation."
    )
# =========================================================
# V31 - DISPLAY SMART RECOMMENDATION
# =========================================================

st.info(
    f"💡 **Smart Recommendation:** {recommendation}"
)

# =========================================================
# V31 - DISPLAY BATTERY HEALTH SCORE
# =========================================================

st.subheader("🔋 Battery Health Analysis")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Battery Health",
        prediction
    )

with col2:
    st.metric(
        "Health Score",
        f"{health_score}%"
    )
# =========================================================
# V29.5 - CONSISTENT HEALTH ASSESSMENT
# =========================================================

# HEALTH SCORE

health_score = 100

if voltage < 12.0:
    health_score -= 15

if voltage < 11.5:
    health_score -= 15

if soc < 30:
    health_score -= 10

if soc < 20:
    health_score -= 10

if temperature > 35:
    health_score -= 10

if temperature > 42:
    health_score -= 15

health_score = max(
    0,
    min(100, health_score)
)

# Convert prediction safely to text
prediction_text = str(prediction).strip()

# HEALTH CONDITION

if (
    prediction_text == "Critical"
    or health_score < 50
):

    condition = "Critical"

    health_status = "🔴 CRITICAL"

    condition_message = (
        "Battery condition is critical. "
        "Check the battery immediately."
    )

elif (
    prediction_text == "Needs Attention"
    or health_score < 80
):

    condition = "Needs Attention"

    health_status = "🟡 WARNING"

    condition_message = (
        "Battery needs attention. "
        "Monitor voltage, temperature and charge level."
    )

else:

    condition = "Healthy"

    health_status = "🟢 GOOD"

    condition_message = (
        "Battery is operating in a healthy condition."
    )
# PREDICTION FEEDBACK

st.info(
    f"🤖 ML Prediction: {prediction}"
)

st.caption(
    f"💡 {condition_message}"
)
# RECOMMENDATION

if condition == "Healthy":

    recommendation = (
        "Battery is operating normally. "
        "Continue regular monitoring and maintenance."
    )

elif condition == "Needs Attention":

    recommendation = (
        "Battery should be monitored closely. "
        "Check voltage, temperature and charge level."
    )

else:

    recommendation = (
        "Battery requires immediate attention. "
        "Check the battery condition and consider servicing or replacement."
    )
# =========================================================
# V25 - EXPLAINABLE AI (XAI)
# =========================================================

st.header("🧠 Explainable AI")

st.write(
    "The Random Forest model uses these factors to make battery health predictions."
)

feature_names = [
    "Voltage",
    "Current",
    "Temperature",
    "SOC"
]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="ML Feature Importance"
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.caption(
    "Higher importance means the feature has a greater influence on the "
    "Random Forest model's overall decisions."
)
# =========================================================
# V25.2 - INDIVIDUAL PREDICTION EXPLANATION
# =========================================================

st.subheader("🔍 Why did the model make this prediction?")

explanation = None

try:

    if shap is None:
        raise ImportError(
            "SHAP is not installed. Add 'shap==0.52.0' to requirements.txt."
        )

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(input_data)

    values = shap_values.values

    # Handle SHAP output for different versions/shapes
    if values.ndim == 3:

        class_index = list(model.classes_).index(prediction)

        explanation_values = values[0, :, class_index]

    elif values.ndim == 2:

        explanation_values = values[0]

    else:

        explanation_values = values.flatten()

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Impact": explanation_values
    })

    explanation["Absolute Impact"] = (
        explanation["Impact"].abs()
    )

    explanation = explanation.sort_values(
        by="Absolute Impact",
        ascending=False
    )

    fig = px.bar(
        explanation,
        x="Impact",
        y="Feature",
        orientation="h",
        title=f"Why the model predicted: {prediction}"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

except Exception as e:

    st.error(
        f"⚠️ Individual XAI explanation unavailable: {e}"
    )
# =========================================================
# V25.3 - HUMAN-READABLE AI EXPLANATION
# =========================================================

st.subheader("💡 AI Explanation")

try:

    top_features = explanation.head(2)

    feature_1 = top_features.iloc[0]["Feature"]
    feature_2 = top_features.iloc[1]["Feature"]

    impact_1 = top_features.iloc[0]["Impact"]
    impact_2 = top_features.iloc[1]["Impact"]

    direction_1 = "increased" if impact_1 > 0 else "decreased"
    direction_2 = "increased" if impact_2 > 0 else "decreased"

    st.info(
        f"🔎 The model predicted **{prediction}**. "
        f"The strongest factors were **{feature_1}** and **{feature_2}**. "
        f"{feature_1} {direction_1} the model's tendency toward this prediction, "
        f"while {feature_2} {direction_2} it."
    )

except Exception as e:

    st.error(
        f"⚠️ AI Explanation Error: {e}"
    )

# OPERATING STATUS

if current > 0.1:

    operating_status = "☀️ CHARGING"

elif current < -0.1:

    operating_status = "🔌 DISCHARGING"

else:

    operating_status = "⏸ IDLE"
# CURRENT BATTERY PARAMETERS

st.header("📊 Current Battery Parameters")

p1, p2, p3, p4 = st.columns(4)

with p1:
    st.metric(
        "⚡ Voltage",
        f"{voltage} V"
    )

with p2:
    st.metric(
        "🔌 Current",
        f"{current} A"
    )

with p3:
    st.metric(
        "🌡 Temperature",
        f"{temperature} °C"
    )

with p4:
    st.metric(
        "🔋 State of Charge",
        f"{soc}%"
    )
# =========================================================
# V26.1 - BATTERY OVERVIEW
# =========================================================

st.header("🔋 Battery Overview")

o1, o2, o3 = st.columns(3)

with o1:
    st.metric(
        "❤️ Health Score",
        f"{health_score}%"
    )

with o2:
    st.metric(
        "🤖 ML Prediction",
        prediction
    )

with o3:
    st.metric(
        "⚙️ Operating Status",
        operating_status
    )
# =========================================================
# V26.2 - VISUAL BATTERY HEALTH STATUS
# =========================================================

st.subheader("🔋 Battery Health Status")

if condition == "Healthy":

    st.success(
        f"🟢 HEALTHY — Battery health score: {health_score}%"
    )

elif condition == "Needs Attention":

    st.warning(
        f"🟡 NEEDS ATTENTION — Battery health score: {health_score}%"
    )

else:

    st.error(
        f"🔴 CRITICAL — Battery health score: {health_score}%"
    )

st.progress(
    health_score / 100
)
# =========================================================
# V26.3 - SMART DASHBOARD INSIGHT
# =========================================================

st.subheader("💡 Smart Dashboard Insight")

if condition == "Healthy":

    if soc >= 70 and temperature <= 35:

        insight = (
            "Battery parameters are currently in a healthy range. "
            "Continue regular monitoring and maintenance."
        )

    else:

        insight = (
            "Battery is currently healthy, but some parameters "
            "should continue to be monitored."
        )

elif condition == "Needs Attention":

    insight = (
        "Some battery parameters require attention. "
        "Monitor voltage, SOC and temperature closely."
    )

else:

    insight = (
        "Battery parameters indicate a critical condition. "
        "Inspection and corrective action are recommended."
    )

st.info(
    f"🔎 {insight}"
)

# LIVE BATTERY GAUGES

st.header("🎛️ Live Battery Gauges")

g1, g2 = st.columns(2)

with g1:

    voltage_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=voltage,
            title={"text": "Battery Voltage (V)"},
            gauge={
                "axis": {"range": [10, 14]},
                "threshold": {
                    "line": {"width": 4},
                    "value": 12.0
                }
            }
        )
    )

    voltage_gauge.update_layout(height=300)

    st.plotly_chart(
        voltage_gauge,
        width="stretch"
    )


with g2:

    soc_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=soc,
            title={"text": "State of Charge (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"width": 4},
                    "value": 50
                }
            }
        )
    )

    soc_gauge.update_layout(height=300)

    st.plotly_chart(
        soc_gauge,
        width="stretch"
    )


g3, g4 = st.columns(2)


with g3:

    current_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current,
            title={"text": "Battery Current (A)"},
            gauge={
                "axis": {"range": [-2, 2]},
                "threshold": {
                    "line": {"width": 4},
                    "value": 0
                }
            }
        )
    )

    current_gauge.update_layout(height=300)

    st.plotly_chart(
        current_gauge,
        width="stretch"
    )


with g4:

    temperature_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=temperature,
            title={"text": "Battery Temperature (°C)"},
            gauge={
                "axis": {"range": [15, 55]},
                "threshold": {
                    "line": {"width": 4},
                    "value": 35
                }
            }
        )
    )

    temperature_gauge.update_layout(height=300)

    st.plotly_chart(
        temperature_gauge,
        width="stretch"
    )
# BATTERY HEALTH STATUS

st.header("❤️ Battery Health Status")

health_col1, health_col2 = st.columns([1, 2])

with health_col1:

    health_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={"text": "Battery Health Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "threshold": {
                    "line": {"width": 5},
                    "value": health_score
                }
            }
        )
    )

    health_gauge.update_layout(height=350)

    st.plotly_chart(
        health_gauge,
        width="stretch"
    )


with health_col2:

    st.subheader("Overall Battery Condition")

    if condition == "Healthy":

        st.success("🟢 GOOD")
        st.write("Battery condition is healthy.")
        st.write("Normal operation can continue.")

    elif condition == "Needs Attention":

        st.warning("🟡 WARNING")
        st.write("Battery needs attention.")
        st.write("Monitor the battery parameters closely.")

    else:

        st.error("🔴 CRITICAL")
        st.write("Battery condition is critical.")
        st.write("Inspection or service is recommended.")

    st.metric(
        "ML Prediction",
        prediction
    )

    st.metric(
        "ML Confidence",
        f"{confidence:.1f}%"
    )

    st.metric(
        "Operating Status",
        operating_status
    )
# =========================================================
# RECOMMENDATION
# =========================================================

st.header("💡 Battery Health Recommendation")

if condition == "Healthy":

    st.success(
        f"🟢 {recommendation}"
    )

elif condition == "Needs Attention":

    st.warning(
        f"🟡 {recommendation}"
    )

else:

    st.error(
        f"🔴 {recommendation}"
    )


# =========================================================
# SMART ANALYSIS & ALERTS
# =========================================================

st.header("⚠️ Smart Analysis & Alerts")

alerts = []


# Voltage alert

if voltage < 11.5:

    alerts.append(
        ("🔴 CRITICAL",
         "Battery voltage is critically low.")
    )

elif voltage < 12.0:

    alerts.append(
        ("🟡 WARNING",
         "Battery voltage is below the recommended level.")
    )

else:

    alerts.append(
        ("🟢 NORMAL",
         "Battery voltage is within the normal range.")
    )


# SOC alert

if soc < 20:

    alerts.append(
        ("🔴 CRITICAL",
         "Battery charge is critically low.")
    )

elif soc < 30:

    alerts.append(
        ("🟡 WARNING",
         "Battery charge is below 30%.")
    )

else:

    alerts.append(
        ("🟢 NORMAL",
         "Battery charge level is satisfactory.")
    )


# Temperature alert

if temperature > 42:

    alerts.append(
        ("🔴 CRITICAL",
         "Battery temperature is dangerously high.")
    )

elif temperature > 35:

    alerts.append(
        ("🟡 WARNING",
         "Battery temperature is above the normal range.")
    )

else:

    alerts.append(
        ("🟢 NORMAL",
         "Battery temperature is within the normal range.")
    )


# Health alert

if health_score < 50:

    alerts.append(
        ("🔴 CRITICAL",
         "Overall battery health requires immediate attention.")
    )

elif health_score < 80:

    alerts.append(
        ("🟡 WARNING",
         "Battery health should be monitored closely.")
    )

else:

    alerts.append(
        ("🟢 NORMAL",
         "Overall battery health is good.")
    )


# Display alerts

for level, message in alerts:

    if level == "🔴 CRITICAL":

        st.error(
            f"{level} — {message}"
        )

    elif level == "🟡 WARNING":

        st.warning(
            f"{level} — {message}"
        )

    else:

        st.success(
            f"{level} — {message}"
        )


# Overall analysis

st.subheader("🔎 Overall Analysis")

critical_count = sum(
    1
    for level, message in alerts
    if level == "🔴 CRITICAL"
)

warning_count = sum(
    1
    for level, message in alerts
    if level == "🟡 WARNING"
)

if critical_count > 0:

    st.error(
        f"🚨 {critical_count} critical issue(s) detected. "
        "Battery inspection is recommended."
    )

elif warning_count > 0:

    st.warning(
        f"⚠️ {warning_count} warning(s) detected. "
        "Continue monitoring the battery."
    )

else:

    st.success(
        "✅ All monitored battery parameters are currently normal."
    )
# =========================================================
# V29.3 - SAFE BATTERY HISTORY MANAGEMENT
# =========================================================

new_data = pd.DataFrame([{
    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Voltage": voltage,
    "Current": current,
    "Temperature": temperature,
    "SOC": soc,
    "Health Score": health_score,
    "Prediction": prediction,
    "Condition": condition,
    "Status": operating_status
}])

history_file = "battery_history.csv"

required_history_columns = [
    "Time",
    "Voltage",
    "Current",
    "Temperature",
    "SOC",
    "Health Score",
    "Prediction",
    "Condition",
    "Status"
]

try:

    if os.path.exists(history_file):

        history = pd.read_csv(history_file)

        missing_columns = [
            column
            for column in required_history_columns
            if column not in history.columns
        ]

        if missing_columns:

            st.warning(
                "⚠️ Battery history file is missing required "
                f"columns: {', '.join(missing_columns)}"
            )

            history = new_data

        else:

            history = pd.concat(
                [history, new_data],
                ignore_index=True
            )

    else:

        history = new_data

    history = history[
        required_history_columns
    ]

    history = history.tail(100)

    history.to_csv(
        history_file,
        index=False
    )

except Exception as e:

    st.warning(
        f"⚠️ Battery history could not be saved: {e}"
    )

    history = new_data
# =========================================================
# V31.5 - BATTERY ANALYTICS
# =========================================================

st.subheader("📊 Battery Analytics")

if len(history) > 1:

    st.markdown("### 🔋 Battery Parameter Trends")

    st.line_chart(
        history.set_index("Time")[
            ["Voltage", "Current", "Temperature", "SOC"]
        ]
    )

else:

    st.info(
        "📊 Collect more battery readings to display trends."
    )
# =========================================================
# V31.6 - HEALTH SCORE TREND
# =========================================================

if len(history) > 1:

    st.markdown("### ❤️ Battery Health Score Trend")

    st.line_chart(
        history.set_index("Time")[
            ["Health Score"]
        ]
    )
# =========================================================
# V31.7 - RECENT BATTERY READINGS
# =========================================================

st.markdown("### 🕒 Recent Battery Readings")

if len(history) > 0:

    recent_history = history.tail(10).iloc[::-1]

    st.dataframe(
        recent_history,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No battery readings available yet.")
# =========================================================
# V31.8 - BATTERY CONDITION DISTRIBUTION
# =========================================================

st.markdown("### 🧠 Battery Condition Distribution")

if len(history) > 0:

    condition_counts = (
        history["Prediction"]
        .value_counts()
        .rename_axis("Battery Condition")
        .reset_index(name="Readings")
    )

    st.bar_chart(
        condition_counts.set_index("Battery Condition")
    )

else:

    st.info("No prediction history available yet.")
# =========================================================
# V28.1 - BATTERY PERFORMANCE SUMMARY
# =========================================================

st.header("📈 Battery Performance Summary")

if not history.empty:

    total_readings = len(history)

    healthy_count = (
        history["Condition"] == "Healthy"
    ).sum()

    attention_count = (
        history["Condition"] == "Needs Attention"
    ).sum()

    critical_count = (
        history["Condition"] == "Critical"
    ).sum()

    average_voltage = history["Voltage"].mean()
    average_temperature = history["Temperature"].mean()
    average_soc = history["SOC"].mean()

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "📊 Total Readings",
            total_readings
        )

    with s2:
        st.metric(
            "🟢 Healthy",
            healthy_count
        )

    with s3:
        st.metric(
            "🔴 Critical",
            critical_count
        )

    s4, s5, s6 = st.columns(3)

    with s4:
        st.metric(
            "⚡ Avg Voltage",
            f"{average_voltage:.2f} V"
        )

    with s5:
        st.metric(
            "🌡 Avg Temperature",
            f"{average_temperature:.1f} °C"
        )

    with s6:
        st.metric(
            "🔋 Avg SOC",
            f"{average_soc:.1f}%"
        )

    st.caption(
        f"🟡 Needs Attention readings: {attention_count}"
    )

else:

    st.info(
        "ℹ️ No battery history available yet."
    )
# =========================================================
# V28.2 - BATTERY TREND ANALYSIS
# =========================================================

st.header("📈 Battery Trend Analysis")

if len(history) >= 2:

    chart_data = history.copy()

    chart_data["Time"] = pd.to_datetime(
        chart_data["Time"]
    )

    st.subheader("⚡ Voltage Trend")

    voltage_chart = px.line(
        chart_data,
        x="Time",
        y="Voltage",
        markers=True,
        title="Battery Voltage Over Time"
    )

    st.plotly_chart(
        voltage_chart,
        width="stretch"
    )

    st.subheader("🌡 Temperature Trend")

    temperature_chart = px.line(
        chart_data,
        x="Time",
        y="Temperature",
        markers=True,
        title="Battery Temperature Over Time"
    )

    st.plotly_chart(
        temperature_chart,
        width="stretch"
    )

    st.subheader("🔋 SOC Trend")

    soc_chart = px.line(
        chart_data,
        x="Time",
        y="SOC",
        markers=True,
        title="State of Charge Over Time"
    )

    st.plotly_chart(
        soc_chart,
        width="stretch"
    )

    st.subheader("❤️ Health Score Trend")

    health_chart = px.line(
        chart_data,
        x="Time",
        y="Health Score",
        markers=True,
        title="Battery Health Score Over Time"
    )

    st.plotly_chart(
        health_chart,
        width="stretch"
    )

else:

    st.info(
        "ℹ️ At least 2 battery readings are required "
        "to display trend charts."
    )
# =========================================================
# BATTERY HISTORY & ANALYTICS
# =========================================================

st.header("📈 Battery History & Analytics")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Voltage",
    "🔌 Current",
    "🌡 Temperature",
    "❤️ Health Score",
    "📊 Condition",
    "📋 Data"
])


with tab1:

    fig = px.line(
        history,
        x="Time",
        y="Voltage",
        title="Battery Voltage History",
        markers=True
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with tab2:

    fig = px.line(
        history,
        x="Time",
        y="Current",
        title="Battery Current History",
        markers=True
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with tab3:

    fig = px.line(
        history,
        x="Time",
        y="Temperature",
        title="Battery Temperature History",
        markers=True
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with tab4:

    fig = px.line(
        history,
        x="Time",
        y="Health Score",
        title="Battery Health Score History",
        markers=True
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with tab5:

    condition_counts = (
        history["Condition"]
        .value_counts()
        .reset_index()
    )

    condition_counts.columns = [
        "Condition",
        "Count"
    ]

    fig = px.pie(
        condition_counts,
        names="Condition",
        values="Count",
        title="Battery Condition Distribution",
        hole=0.45
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with tab6:

    st.dataframe(
        history,
        width="stretch"
    )


# =========================================================
# BATTERY PERFORMANCE SUMMARY
# =========================================================

st.header("📊 Battery Performance Summary")

s1, s2, s3, s4, s5 = st.columns(5)


with s1:

    st.metric(
        "Avg Voltage",
        f"{history['Voltage'].mean():.2f} V"
    )


with s2:

    st.metric(
        "Avg Current",
        f"{history['Current'].mean():.2f} A"
    )


with s3:

    st.metric(
        "Avg Temperature",
        f"{history['Temperature'].mean():.1f} °C"
    )


with s4:

    st.metric(
        "Avg SOC",
        f"{history['SOC'].mean():.1f}%"
    )


with s5:

    st.metric(
        "Avg Health",
        f"{history['Health Score'].mean():.1f}%"
    )


# =========================================================
# DOWNLOAD REPORT
# =========================================================

st.header("📥 Battery Report")

csv_data = history.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Battery History (CSV)",
    data=csv_data,
    file_name="battery_health_report.csv",
    mime="text/csv",
    width="stretch"
)
# =========================================================
# V29.4 - SAFE PROFESSIONAL BATTERY REPORT
# =========================================================

report_time = datetime.now().strftime(
    "%d-%m-%Y %H:%M:%S"
)

report_id = datetime.now().strftime(
    "BAT-%Y%m%d-%H%M%S"
)

st.subheader("📄 Professional Battery Report")

# Safe XAI factor extraction
try:

    if (
        "explanation" in locals()
        and not explanation.empty
        and "Feature" in explanation.columns
    ):

        top_xai_factors = ", ".join(
            explanation.head(2)["Feature"].astype(str).tolist()
        )

    else:

        top_xai_factors = (
            "XAI information unavailable"
        )

except Exception:

    top_xai_factors = (
        "XAI information unavailable"
    )

# Safe report generation
try:

    report_text = f"""
SMART SOLAR BATTERY HEALTH MONITORING SYSTEM
=============================================

BATTERY HEALTH REPORT

Report ID      : {report_id}
Generated On   : {report_time}

Current Battery Parameters
--------------------------
Voltage        : {voltage} V
Current        : {current} A
Temperature    : {temperature} °C
State of Charge: {soc} %

Battery Assessment
------------------
ML Prediction  : {prediction}
Condition      : {condition}
Health Score   : {health_score}%

System Status
-------------
Operating Status: {operating_status}

Recommendation
--------------
{recommendation}

AI Explanation
--------------
The Random Forest machine learning model analyzed
Voltage, Current, Temperature and SOC to determine
the battery health condition.

Top XAI Factors
---------------
{top_xai_factors}

This report is generated by the
Smart Solar Battery Health Monitoring System.
"""

    st.download_button(
        label="📄 Download Professional Battery Report",
        data=report_text,
        file_name="smart_battery_report.txt",
        mime="text/plain",
        width="stretch"
    )

except Exception as e:

    st.error(
        f"⚠️ Professional report could not be generated: {e}"
    )
# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Smart Solar Battery Health Monitoring System | "
    "B.Tech Data Science Project Prototype"
)

st.info(
    "ℹ️ The current prototype uses simulated or manually entered "
    "battery parameters. Live ESP32 sensor integration can replace "
    "these inputs in the final stage."
)
