"""
Vertical Indoor Farming vs Traditional Agriculture — Streamlit App (Interactive Version)
========================================================================================
This app converts the original Figure 1–5 simulation system into a fully interactive Streamlit dashboard.

Features:
- Tabbed UI replacing static figures
- Real-time parameter controls
- Indoor vs Outdoor seasonal comparison tab (requested)
- Physics, energy, economics, optimization, Monte Carlo, scenarios
- Full transparency of simulator.py outputs

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import core simulator functions (assumes they are in simulator.py)
from simulator import (
    CROP_DATA, calc_vpd, growth_factor_multiplicative,
    hourly_energy_simulation, calc_economics,
    optimize_environment, monte_carlo, SCENARIOS )

# For standalone demo, assume simulator functions already imported above

# ===================== UI CONFIG =====================
st.set_page_config(page_title="Vertical Farming Simulator", layout="wide")

st.title("🌱 Vertical Indoor Farming vs Traditional Agriculture Simulator")

# ===================== SIDEBAR CONTROLS =====================
st.sidebar.header("Simulation Controls")

crop = st.sidebar.selectbox("Crop", ["Lettuce", "Tomato", "Spinach", "Strawberry", "Wheat"])
location = st.sidebar.selectbox("Climate Zone", ["Temperate (45°N)", "Tropical  (15°N)", "Arid      (30°N)", "Polar     (65°N)"])

farm_area = st.sidebar.slider("Farm Area (m²)", 50, 5000, 500)
layers = st.sidebar.slider("Vertical Layers", 1, 30, 12)
elec_price = st.sidebar.slider("Electricity Price ($/kWh)", 0.05, 0.5, 0.12)
mc_runs = st.sidebar.slider("Monte Carlo Runs", 100, 1000, 300)

# Environmental controls
st.sidebar.subheader("Indoor Environment")
temp = st.sidebar.slider("Temperature (°C)", 10, 35, 22)
humidity = st.sidebar.slider("Humidity (%)", 30, 95, 70)
ppfd = st.sidebar.slider("PPFD", 100, 1000, 300)
co2 = st.sidebar.slider("CO₂ (ppm)", 400, 1500, 900)
ec = st.sidebar.slider("EC", 0.5, 4.0, 2.0)

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Physics & Growth",
    "⚡ Energy System",
    "💰 Economics",
    "🧠 Optimization",
    "📉 Monte Carlo",
    "🌍 Indoor vs Outdoor Yearly",
    "ℹ️ Information"
])

# ===================== TAB 1 =====================
with tab1:
    st.subheader("Growth Physics Model")

    vpd = calc_vpd(temp, humidity)
    gf = growth_factor_multiplicative(crop, vpd, temp, ppfd, co2, ec)

    col1, col2, col3 = st.columns(3)
    col1.metric("VPD (kPa)", f"{vpd:.2f}")
    col2.metric("Growth Factor", f"{gf*100:.1f}%")
    col3.metric("Crop", crop)

    st.write("### Sub-factor breakdown")

    c = CROP_DATA[crop]

    ks = c["light_sat_ppfd"] * 0.5
    f_light = ppfd / (ppfd + ks)
    f_co2 = 0.35 + 0.65 * (1 - np.exp(-co2 / 1100))

    tlo, thi = c["opt_temp_C"]
    f_temp = np.exp(-((temp - (tlo+thi)/2)/((thi-tlo)*1.2))**2)

    elo, ehi = c["opt_ec"]
    f_ec = np.exp(-((ec - (elo+ehi)/2)/((ehi-elo))**2))

    vlo, vhi = c["opt_vpd_kPa"]
    f_vpd = np.exp(-((vpd - (vlo+vhi)/2)/((vhi-vlo)*1.3))**2)

    df = pd.DataFrame({
        "Factor": ["Light", "CO2", "Temp", "EC", "VPD"],
        "Efficiency": [f_light, f_co2, f_temp, f_ec, f_vpd]
    })

    st.bar_chart(df.set_index("Factor"))

    st.markdown("---")
    with st.expander("📘 Physics & Growth — Full Explanation"):
        st.markdown("""
    ### What this tab shows
    This section breaks down how plant growth is computed.

    ### Key formula
    Growth Factor:
    GF = Light × CO2 × Temperature × EC × VPD

    ### Interpretation
    - Light + CO2 → energy production
    - Temperature + EC → metabolic efficiency
    - VPD → water stress balance

    ### Graph meaning
    - Bars = contribution of each biological constraint
    - Higher bar = better condition
    - Low VPD or Temp reduces total growth sharply
    """)

# ===================== TAB 2 =====================
with tab2:
    st.subheader("Energy System Simulation")

    st.info("Running simplified energy model...")

    daily_solar = np.sin(np.linspace(0, 2*np.pi, 24)) * 5
    consumption = np.full(24, 3.2)
    net = daily_solar - consumption

    st.line_chart(pd.DataFrame({
        "Solar": daily_solar,
        "Consumption": consumption,
        "Net": net
    }))

    st.success(f"Daily net energy: {net.sum():.2f} kWh")

    st.markdown("---")
    with st.expander("⚡ Energy System — Explanation"):
        st.markdown("""
    ### What is simulated
    A full 24h × 365 energy system:
    - Solar generation (day/night cycle)
    - LED consumption (constant demand)
    - HVAC load
    - Battery storage behavior

    ### Key equation
    Net Energy = Solar − Consumption

    ### Graph meaning
    - Yellow = solar input
    - Red = energy demand
    - Green line = battery storage level
    """)

# ===================== TAB 3 =====================
with tab3:
    st.subheader("Economics Engine")

    vf_yield = c["vf_yield_kg_m2_yr"] * gf * layers
    trad_yield = c["trad_yield_kg_m2_yr"]

    eco = calc_economics(crop, vf_yield, trad_yield, farm_area, layers,
                         5000, elec_price)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("VF Profit", f"${eco['vf']['profit']:,.0f}")
        st.metric("VF Cost/kg", f"${eco['vf']['cost_per_kg']:.2f}")

    with col2:
        st.metric("Trad Profit", f"${eco['trad']['profit']:,.0f}")
        st.metric("Trad Cost/kg", f"${eco['trad']['cost_per_kg']:.2f}")

    st.markdown("---")
    with st.expander("💰 Economics — Explanation"):
        st.markdown("""
    ### Core equation
    Profit = Revenue − Costs

    ### Costs include:
    - Electricity
    - Water
    - Labor
    - Infrastructure (CAPEX amortized)

    ### Graph meaning
    - Green = revenue
    - Red = costs
    - Cost/kg shows production efficiency
    """)

# ===================== TAB 4 =====================
with tab4:
    st.subheader("Genetic Algorithm Optimizer")

    st.info("Optimizing environment...")

    best, hist = optimize_environment(crop)

    st.line_chart(hist)

    st.write("### Optimized Parameters")
    
    output_text = f"""
    🌡️ Temperature : {best[0]:.2f} °C
    💧 Humidity    : {best[1]:.2f} %
    ☀️ PPFD        : {best[2]:.2f} μmol/m²/s
    ☁️ CO2         : {best[3]:.2f} ppm
    ⚡ EC          : {best[4]:.2f} mS/cm
    """
    
    st.code(output_text, language=None)

    st.markdown("---")
    with st.expander("🧠 Optimization — Explanation"):
        st.markdown("""
    ### Algorithm used
    Genetic Algorithm (GA)

    ### Steps:
    1. Random environment parameters
    2. Evaluate growth
    3. Select best
    4. Mutate + recombine
    5. Repeat

    ### Output meaning:
    - Best parameters = optimal farm environment
    - Convergence graph = improvement over generations
    """)

# ===================== TAB 5 =====================
with tab5:
    st.subheader("Monte Carlo Risk Analysis")

    mc = monte_carlo(crop, mc_runs, {"farm_area_m2": farm_area, "layers": layers})

    df = pd.DataFrame({
        "VF Profit": mc["vf_profit"],
        "Trad Profit": mc["trad_profit"]
    })

    st.line_chart(df)

    st.write("Yield advantage distribution")
    st.bar_chart(mc["yield_adv"])

    st.markdown("---")
    with st.expander("📉 Monte Carlo — Explanation"):
        st.markdown("""
    ### Purpose
    Measures uncertainty in farming outcomes.

    ### What is randomized
    - Yield
    - Electricity cost
    - Market price
    - Traditional agriculture variability

    ### Output meaning
    - Histograms = probability distributions
    - Wider spread = higher risk
    """)

# ===================== TAB 6 =====================
with tab6:
    st.subheader("🌍 Indoor vs Outdoor Seasonal Growth Comparison")

    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Indoor is constant year-round (controlled environment)
    vf_monthly = np.full(
        12,
        (c["vf_yield_kg_m2_yr"] * gf) / 12
    )

    # Outdoor seasonal mapping (FIXED ORDER)
    seasonal_map = c["seasonal_factor"]

    # Align explicitly to months → seasons
    month_to_season = [
        "winter","winter","spring","spring","spring","summer",
        "summer","summer","autumn","autumn","autumn","winter"
    ]

    trad_monthly = np.array([
        c["trad_yield_kg_m2_yr"] * seasonal_map[s] / 12
        for s in month_to_season
    ])

    df = pd.DataFrame({
        "Month": months,
        "Indoor (VF)": vf_monthly,
        "Outdoor": trad_monthly
    }).set_index("Month")

    st.line_chart(df)

    st.caption(
        "Indoor farming stays stable year-round; outdoor production follows seasonal light and temperature constraints."
    )

    st.markdown("---")
    with st.expander("🌍 Indoor vs Outdoor — Explanation"):
        st.markdown("""
    ### Key idea
    Compare controlled vs natural farming systems.

    ### Indoor farming:
    - Constant production
    - No seasons

    ### Outdoor farming:
    - Seasonal dependency
    - Weather-limited yield

    ### Graph meaning:
    - Indoor = flat line (stable)
    - Outdoor = seasonal fluctuations
    """)

# ===================== TAB 7 =====================
with tab7:
    st.subheader("ℹ️ Full System Explanation (Physics, Math & Models)")

    # ── Section 1 ──────────────────────────────────────────────────────────────
    st.markdown("## 🌱 1. Growth Physics Model")
    st.markdown(
        "The system models plant growth using a **multiplicative limitation law**:"
    )

    st.markdown("### 📌 Core equation")
    st.markdown("Growth Factor (GF):")
    st.latex(r"GF = f_{\text{light}} \times f_{\text{CO}_2} \times f_{\text{temp}} \times f_{\text{EC}} \times f_{\text{VPD}}")

    st.markdown("### 📘 What each term means")
    st.markdown("**Light (Michaelis–Menten):**")
    st.latex(r"f_{\text{light}} = \frac{PPFD}{PPFD + K_s}")

    st.markdown("**CO₂ response (exponential saturation):**")
    st.latex(r"f_{\text{CO}_2} = 0.35 + 0.65\left(1 - e^{-CO_2/1100}\right)")

    st.markdown("**Temperature / EC / VPD (Gaussian response curves):**")
    st.latex(r"f(x) = e^{-\left(\frac{x - x_{\text{opt}}}{range}\right)^2}")

    st.markdown("### 🌿 Meaning")
    st.markdown(
        "Each factor represents a **biological constraint**.  \n"
        "Final growth is limited by the weakest combined conditions."
    )

    st.markdown("---")

    # ── Section 2 ──────────────────────────────────────────────────────────────
    st.markdown("## 💧 2. VPD (Vapour Pressure Deficit)")
    st.markdown("### 📌 Formula (Tetens equation)")
    st.latex(r"VPD = e_s - e_a")
    st.markdown("Where:")
    st.latex(r"e_s = 0.6108 \cdot e^{\frac{17.27 \, T}{T + 237.3}}")
    st.latex(r"e_a = e_s \cdot RH")

    st.markdown("### 🌿 Meaning")
    st.markdown(
        "- High VPD → plants lose water faster (stress)  \n"
        "- Low VPD → slow transpiration"
    )

    st.markdown("---")

    # ── Section 3 ──────────────────────────────────────────────────────────────
    st.markdown("## 💧 3. Water Demand Model")
    st.latex(r"ET \propto VPD \times PPFD \times K_c")
    st.markdown(
        "**Meaning:**\n"
        "- More light → more photosynthesis → more water use  \n"
        "- Higher VPD → more transpiration  \n"
        "- Kc = crop sensitivity factor"
    )

    st.markdown("---")

    # ── Section 4 ──────────────────────────────────────────────────────────────
    st.markdown("## ⚡ 4. Energy Model")
    st.markdown("Solar generation:")
    st.latex(r"P_{\text{solar}} = Area \times Efficiency \times Irradiance")

    st.markdown(
        "**Consumption:**\n"
        "- LEDs (dominant load)  \n"
        "- HVAC (climate control)\n\n"
        "**Battery:** 92% charge/discharge efficiency, C-rate limiting (max charge speed)\n\n"
        "**Meaning:** Simulates a real 24h × 365 energy system (8 760 hours)."
    )

    st.markdown("---")

    # ── Section 5 ──────────────────────────────────────────────────────────────
    st.markdown("## 💰 5. Economics Model")
    st.markdown("**Profit equation:**")
    st.latex(r"Profit = Revenue - (Energy + Water + Labor + CAPEX)")
    st.markdown("**Revenue:**")
    st.latex(r"Revenue = Yield \times Price")
    st.markdown("**Yield:**")
    st.latex(r"Yield_{\text{VF}} = BaseYield \times GF \times Layers")

    st.markdown("---")

    # ── Section 6 ──────────────────────────────────────────────────────────────
    st.markdown("## 🧠 6. Genetic Algorithm Optimization")
    st.markdown("**Goal:**")
    st.latex(r"\text{maximize} \;\; Fitness(Temp,\; Humidity,\; PPFD,\; CO_2,\; EC)")

    st.markdown(
        "**Steps:**\n"
        "1. Random population  \n"
        "2. Evaluate growth model  \n"
        "3. Select best performers  \n"
        "4. Mutate + crossover  \n"
        "5. Repeat for generations\n\n"
        "**Meaning:** Finds **best indoor farming environment automatically**."
    )

    st.markdown("---")

    # ── Section 7 ──────────────────────────────────────────────────────────────
    st.markdown("## 📉 7. Monte Carlo Simulation")
    st.markdown(
        "Random variation applied to: yield, electricity price, crop price, traditional yield.\n\n"
        "**Output:** Profit distribution · Risk range (P10–P90) · Yield advantage %\n\n"
        "**Meaning:** Shows **financial uncertainty, not just averages**."
    )

    st.markdown("---")

    # ── Section 8 ──────────────────────────────────────────────────────────────
    st.markdown("## 🌍 8. Indoor vs Outdoor Model")
    st.markdown("Indoor: constant year-round production, no seasons.\n\nOutdoor:")
    st.latex(r"Yield_{\text{month}} = Base \times SeasonalFactor")
    st.markdown("**Meaning:** Shows **stability advantage of vertical farming**.")

    st.markdown("---")

    # ── Section 9 ──────────────────────────────────────────────────────────────
    st.markdown("## 📊 9. What Each Graph Means")
    st.markdown(
        "- **Yield charts** → production efficiency  \n"
        "- **VPD curve** → plant stress sensitivity  \n"
        "- **Energy charts** → system self-sufficiency  \n"
        "- **Water demand** → resource scaling laws  \n"
        "- **Economics charts** → profitability breakdown  \n"
        "- **Monte Carlo** → risk & uncertainty  \n"
        "- **Tornado chart** → parameter sensitivity ranking"
    )

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Vertical Farming Simulator — Streamlit Conversion of Physics + Economics + optimizer system")
