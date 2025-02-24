import streamlit as st 

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az

import pymc as pm

import json
import time

def calculate_expected_loss(variant_a_simulation, variant_b_simulation, variant_b_won, min_difference_delta=0):
    # calculate losses for both control and treatment simulations
    loss_variant_a = [max((j - min_difference_delta) - i, 0) for i, j in zip(variant_a_simulation, variant_b_simulation)]
    loss_variant_b = [max(i - (j - min_difference_delta), 0) for i, j in zip(variant_a_simulation, variant_b_simulation)]

    # Apply the treatment_won flag to calculate total loss
    all_loss_variant_a = [int(i) * j for i, j in zip(variant_b_won, loss_variant_a)]
    all_loss_variant_b = [(1 - int(i)) * j for i, j in zip(variant_b_won, loss_variant_b)]

    # Calculate expected loss as the average of all losses
    expected_loss_variant_a = np.mean(all_loss_variant_a)
    expected_loss_variant_b = np.mean(all_loss_variant_b)
    return expected_loss_variant_a, expected_loss_variant_b

def experiment_simulations(variant_a_cr, variant_b_cr, n=100, prior_alpha=1, prior_beta=1, epsilon=0.001, variant_sample_size=1000, min_simulations_per_experiment=0):
    result = pd.DataFrame()  # Initialise result dataframe

    # Loop through simulations
    for simulation in range(n):
        sample_size, variant_a_conversions, variant_b_conversions = 0, 0, 0  # Initialise variables
        records = []

        # Generate binomial simulations for both variants
        variant_a_simulations = np.random.binomial(n=1, p=variant_a_cr, size=variant_sample_size)
        variant_b_simulations = np.random.binomial(n=1, p=variant_b_cr, size=variant_sample_size)

        # Loop through each sample in the simulation
        for i in range(variant_sample_size):
            sample_size += 1
            variant_a_conversions += variant_a_simulations[i]  # Count conversions for variant A
            variant_b_conversions += variant_b_simulations[i]  # Count conversions for variant B

            # Generate beta distributions for both variants
            variant_a_pdfs = np.random.beta(prior_alpha + variant_a_conversions, prior_beta + (sample_size - variant_a_conversions), size=100)
            variant_b_pdfs = np.random.beta(prior_alpha + variant_b_conversions, prior_beta + (sample_size - variant_b_conversions), size=100)

            # Determine if treatment (variant B) has higher values than control (variant A)
            variant_b_pdf_higher = [b > a for a, b in zip(variant_a_pdfs, variant_b_pdfs)]  # Compare variant A and B distributions

            # Calculate the expected losses for both variants
            expected_loss_variant_a, expected_loss_variant_b = calculate_expected_loss(variant_a_pdfs, variant_b_pdfs, variant_b_pdf_higher)

            # Prepare the record to store the results
            record = {
                'simulation': simulation + 1,
                'sample': sample_size,
                'variant_b_cr': variant_b_conversions / sample_size,  # Conversion rate for variant B
                'variant_a_cr': variant_a_conversions / sample_size,  # Conversion rate for variant A
                'variant_b_expected_loss': expected_loss_variant_b,  # Expected loss for variant B
                'variant_a_expected_loss': expected_loss_variant_a  # Expected loss for variant A
            }

            # Determine the winner based on expected losses and threshold
            if simulation >= min_simulations_per_experiment:
                if expected_loss_variant_b <= epsilon:
                    record['winner'] = 'variant_b'  # Variant B wins
                elif expected_loss_variant_a <= epsilon:
                    record['winner'] = 'variant_a'  # Variant A wins
                else:
                    record['winner'] = 'inconclusive'  # No clear winner

            # Append the record to the list
            records.append(record)

        # Convert the records to a DataFrame and concatenate with the previous results
        simulation_results = pd.DataFrame.from_records(records)
        result = pd.concat([result, simulation_results])  # Append results for all simulations

    # Return the concatenated result after all simulations are completed
    return result

def plot_sample_size_required(baseline_cr=.1, mde=.01, power_threshold=80, num_simulations=5):
    # Set up the plot with seaborn style
    sns.set(style="whitegrid")
    plt.figure(figsize=(11, 6))

    # List to store first hit power values
    first_hit_powers = []
    x_max_values = []  
    simulation_status = st.empty()

    with st.spinner(f'Running simulations'):
        # Loop to run the simulation multiple times and plot each result
        for _ in range(num_simulations):
            time.sleep(1)

            if _ + 1 == num_simulations:
                simulation_status.write(f'Ran {num_simulations} simulations!')
            else:
                simulation_status.write(f'{_+1} out of {num_simulations}')
            
            # Simulate the experiment data
            df = experiment_simulations(baseline_cr, baseline_cr * (mde+1))

            # Filter out inconclusive simulations
            simulations = df[df['winner'] != 'inconclusive']

            # Group by simulation and sample to get the minimum sample size for each simulation
            grouped = simulations[['simulation', 'sample']].groupby('simulation', as_index=False).min()
            results = simulations.merge(grouped, on=['simulation', 'sample'])

            # Count the number of simulations for each sample size
            hist = results[['simulation', 'sample']].groupby('sample', as_index=False).count()
            records = hist.set_index('sample').to_dict()['simulation']

            # Initialization
            conclusive_simulations = 0
            plotting_conclusions = []
            x = []
            first_hit_power = None  

            # Iterate dynamically over sample sizes
            for i in range(1, 1_000_000):
                if i in records:
                    conclusive_simulations += records[i]

                plotting_conclusions.append(conclusive_simulations)
                x.append(i)

                # Capture the first time the power_threshold% is hit
                if conclusive_simulations >= power_threshold and first_hit_power is None:
                    first_hit_power = i  

                # Stop when sample size reaches double the first power_threshold% hit
                if first_hit_power and i >= first_hit_power * 2:
                    break  

            # Store the first hit power for later analysis
            if first_hit_power is not None:
                first_hit_powers.append(first_hit_power)
                x_max_values.append(first_hit_power * 2)

            # Define x-axis limit (up to double the first power_threshold% hit)
            x_max = min(max(x_max_values), 1_000_000)

            # Plot the cumulative conclusive simulations line for this simulation
            plt.plot(x, plotting_conclusions, color='blue', alpha=0.3)  # Set alpha for transparency

    # Plot the horizontal dashed line for the power threshold
    plt.axhline(y=power_threshold, color='red', linestyle='--', label=f'{power_threshold}% threshold')

    # Set plot limits and labels
    plt.xlim([1, x_max])
    plt.ylim([0, power_threshold + 10])
    plt.title(f'Conclusive Simulations for {int(baseline_cr*100)}% Conversion Rate and {int(mde*100)}% Practical Significance')
    plt.xlabel('Sample Size')
    plt.ylabel('Proportion of Conclusive Simulations')

    # Show legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    st.pyplot(plt)

    # After plotting, calculate and print statistics for the first hit powers
    if first_hit_powers:
        mean_first_hit_power = int(np.mean(first_hit_powers))
        min_first_hit_power = np.min(first_hit_powers)
        max_first_hit_power = np.max(first_hit_powers)

        stats_data = {
        'Statistic': ['Mean First Hit Power', 'Min First Hit Power', 'Max First Hit Power'],
        'Value': [mean_first_hit_power, min_first_hit_power, max_first_hit_power]
    }
        # Convert to dataframe
        stats_df = pd.DataFrame(stats_data)

        # Display the title and simulation information
        st.write(f"First hit power statistics over {num_simulations} simulations:")
        st.write(f"With baseline conversion: {int(100 * baseline_cr)}%, and Practical Significance {int(100 * mde)}%")

        # Display the statistics in a table format
        st.table(stats_df)

def calculate_ab_test(variant_a_conversions, variant_a_samples, variant_b_conversions, variant_b_samples):
    with pm.Model() as model:
        # Assign weak priors
        p_a = pm.Beta('p_a', alpha=1, beta=1)
        p_b = pm.Beta('p_b', alpha=1, beta=1)

        # Deterministic delta function to calculate difference in p_a and p_b
        delta = pm.Deterministic('delta', p_a - p_b)

        # Observed data modeled as binomial distribution
        obs_a = pm.Binomial('obs_a', n=variant_a_samples, p=p_a, observed=variant_a_conversions)
        obs_b = pm.Binomial('obs_b', n=variant_b_samples, p=p_b, observed=variant_b_conversions)

        # Perform MCMC sampling
        trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)
        return trace

def results_ab_test(variant_a_conversions, variant_a_samples, variant_b_conversions, variant_b_samples, rope):
    # Ensure conversions are smaller than or equal to samples
    if variant_a_conversions > variant_a_samples:
        raise ValueError("Conversions for Variant A cannot be greater than sample size.")
    if variant_b_conversions > variant_b_samples:
        raise ValueError("Conversions for Variant B cannot be greater than sample size.")
    
    # Calculate the Bayesian posterior using the model
    trace = calculate_ab_test(variant_a_conversions, variant_a_samples, variant_b_conversions, variant_b_samples)
    
    # Extract the posterior samples for p_a and p_b
    samples_a = trace.posterior['p_a'].values
    samples_b = trace.posterior['p_b'].values

    # Calculate the difference between p_b and p_a
    diff = samples_b - samples_a

    # Create the plot explicitly using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_posterior(diff, rope=rope, rope_color='red', ax=ax)
    ax.set_title('Difference Posterior')

    # Calculate expected losses
    # expected_loss_a = np.mean(np.maximum(samples_b - samples_a, 0))  # Probability A is worse than B
    # expected_loss_b = np.mean(np.maximum(samples_a - samples_b, 0))  # Probability B is worse than A

    prob_b_better = (samples_b > samples_a).mean() # Probability that B is better than A
    prob_a_better = (samples_a > samples_b).mean() #  Probability that A is better than B

    # Create a DataFrame with the results
    data = {
        'Samples': [variant_a_samples, variant_b_samples],
        'Conversions': [variant_a_conversions, variant_b_conversions],
        # 'Expected Loss': [expected_loss_a, expected_loss_b],
        'Probability to be Best': [f"{prob_a_better * 100:.1f}%", f"{prob_b_better * 100:.1f}%"]
    }

    # Create the DataFrame with variants as index
    df = pd.DataFrame(data, index=['Variant A', 'Variant B'])

    st.pyplot(fig)
    
    # Return the DataFrame for display and the figure for plotting
    return df