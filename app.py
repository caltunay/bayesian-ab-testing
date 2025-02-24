import streamlit as st
from streamlit_option_menu import option_menu
from utils import plot_sample_size_required, results_ab_test
import time

# Initialize session state for tab tracking
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Sample Size Calculator"

if "calculating" not in st.session_state:
    st.session_state.calculating = False

# Main content header
st.title("Bayesian Framework A/B Testing")

# Disclaimer at the top of the page
st.markdown(
    """
    <div style="background-color: #3364ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <strong>Disclaimer:</strong> Once calcualte button is clicked, please do not change tabs or sliders. 
        Streamlit will rerun the app during any interaction. Please wait for the calculation to finish.
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar navigation using option menu
with st.sidebar:
    selected_tab = option_menu(
        menu_title="Bayesian Framework A/B Testing",  
        options=["Sample Size Calculator", "Post-Hoc Analysis"],
        icons=["calculator", "bar-chart"], 
        menu_icon="cast",
        default_index=0
    )

# Update session state based on selection
st.session_state.active_tab = selected_tab

# Display corresponding content based on the active tab
if st.session_state.active_tab == "Sample Size Calculator":
    st.header("Sample Size Calculator")

    with st.sidebar:
        baseline_cr = st.slider("Baseline Conversion Rate (%)", 1., 99., 10., step=0.1, disabled=st.session_state.calculating)
        mde = st.slider("Practical Significance (%)", 1., 50., 1., step=0.1, disabled=st.session_state.calculating)
        power_threshold = st.slider("Power Threshold (%)", 50, 100, 80, disabled=st.session_state.calculating)

        # Button to trigger calculation
        calculate_button = st.button('Calculate', disabled=st.session_state.calculating)

    if calculate_button:
        st.session_state.calculating = True  # Disable sliders during calculation
        plot_sample_size_required(baseline_cr / 100, mde / 100, power_threshold)
        st.success("Calculation completed!")
        st.session_state.calculating = False
        st.cache_data.clear()

elif st.session_state.active_tab == "Post-Hoc Analysis":
    st.header("Post-Hoc Analysis")

    with st.sidebar:
        # group a
        st.subheader('Variant A')
        variant_a_conversions = st.number_input('Conversions (Variant A)', min_value=0, step=1)
        variant_a_samples = st.number_input('Samples (Variant A)', min_value=1, step=1)
        
        # group b
        st.subheader('Variant B')
        variant_b_conversions = st.number_input('Conversions (Variant B)', min_value=0, step=1)
        variant_b_samples = st.number_input('Samples (Variant b)', min_value=1, step=1)
        
        # rope
        st.subheader('ROPE')
        rope_min = st.number_input('ROPE, Lower bound', min_value=-1., max_value=0., value=-.01, step=.01)
        rope_max = st.number_input('ROPE, Upper bound', min_value=0., max_value=1., value=.01, step=.01)
        
        # calculate button
        calculate_ab_button = st.button('Calculate')

    if calculate_ab_button:
        st.session_state.calculating = True  
        rope = (rope_min, rope_max)
        df_results = results_ab_test(variant_a_conversions,
            variant_a_samples,
            variant_b_conversions,
            variant_b_samples,
            rope
        )

        # results table
        st.subheader('A/B Test Results')
        st.write(df_results)

        st.success("Calculation completed!")
        st.session_state.calculating = False
        st.cache_data.clear()
            