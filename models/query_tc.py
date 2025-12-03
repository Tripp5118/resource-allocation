#!/usr/bin/env python
'''
Generate CTE and K properties from Thermo-Calc for a composition space.
Simplified version - only calculates CTE and K at room temperature.
System: Fe-Co-Cr-Ni-V
Database: TCHEA8
'''

import numpy as np
import pandas as pd
from tc_python import TCPython, CompositionUnit
from itertools import compress
import concurrent.futures
import os
import time

def writeToTracker(calcName, text):
    '''Write to tracking file.'''
    no_write = True
    while no_write:
        try:
            with open(f'{calcName}-tracker.txt', 'a') as f:
                f.write(text)
            no_write = False
        except:
            pass

def calculate_properties(param):
    '''
    Calculate CTE and K for a batch of compositions.

    Args:
        param: Dictionary containing:
            - INDICES: List of row indices
            - COMP: DataFrame with compositions
            - ACT_EL: List of active elements

    Returns:
        String indicating completion status
    '''
    indices = param["INDICES"]
    comp_df = param["COMP"].copy()  # Make a copy to avoid issues
    elements = param["ACT_EL"]

    with TCPython() as session:
        # Setup property calculation
        # Database: TCHEA8
        # Freeze-in temperature: 500°C = 773.15 K
        # Evaluation temperature: 25°C = 298.15 K
        # Reference temperature for CTE: 25°C = 298.15 K

        prop_calc = (
            session
            .select_database_and_elements('TCHEA8', elements)  # Changed to TCHEA8
            .get_system()
            .with_property_model_calculation('Equilibrium with Freeze-in Temperature')
            .set_argument('Freeze-in-temperature', 773.15)  # 500°C
            .set_argument('Minimization strategy', 'Global minimization only')
            .set_argument('Reference temperature for technical CTE', 298.15)  # 25°C
            .set_temperature(298.15)  # 25°C evaluation temperature
            .set_composition_unit(CompositionUnit.MOLE_FRACTION)
        )

        for i in indices:
            # Get composition and active elements
            comp = comp_df.loc[i][elements]
            active_el = list(compress(elements, list(comp > 0)))

            try:
                # Skip unary compositions
                if len(active_el) == 1:
                    print(f'Skipping unary composition at index {i}')
                    comp_df.at[i, 'CTE (1/K)'] = np.nan
                    comp_df.at[i, 'K (W/(mK))'] = np.nan
                    continue

                # Set composition (all but last element)
                for j in range(len(active_el) - 1):
                    prop_calc = prop_calc.set_composition(active_el[j], comp[active_el[j]])

                # Calculate properties
                prop_result = prop_calc.calculate(timeout_in_minutes=1.5)

                # Extract CTE and K
                comp_df.at[i, 'CTE (1/K)'] = prop_result.get_value_of('Technical thermal expansion (1/K)')
                comp_df.at[i, 'K (W/(mK))'] = prop_result.get_value_of('Thermal conductivity (W/(mK))')

                print(f'Completed index {i}')

            except Exception as e:
                print(f'Exception at index {i}:')
                print(e)
                comp_df.at[i, 'CTE (1/K)'] = np.nan
                comp_df.at[i, 'K (W/(mK))'] = np.nan

            finally:
                # Save progress after each batch
                comp_df.to_csv(f'CalcFiles/Results_Set_ReCalc{param["INDICES"][0]}.csv', index=False)
                continue

    writeToTracker('CTE_K', f"Batch starting at index {param['INDICES'][0]} completed\n")
    return 'Calculation Completed'


if __name__ == '__main__':
    ##########################################################################
    # CONFIGURATION
    ##########################################################################
    input_csv = 'models/failed_predictions.csv'  # Your input CSV file
    elements = ['Fe', 'Co', 'Cr', 'Ni', 'V']  # Fe-Co-Cr-Ni-V system
    batch_size = 100  # Number of compositions per batch
    n_workers = 8  # Number of parallel workers
    ##########################################################################

    # Load composition data
    results_df = pd.read_csv(input_csv)

    # Verify required columns exist
    missing = [e for e in elements if e not in results_df.columns]
    if missing:
        raise ValueError(f"Missing element columns in CSV: {missing}")

    print(f"Loaded {len(results_df)} compositions from {input_csv}")
    print(f"System: {'-'.join(elements)}")
    print(f"Database: TCHEA8")

    # Create output directory
    if not os.path.exists("CalcFiles"):
        os.mkdir("CalcFiles")

    writeToTracker('CTE_K', "*****Start Generating Calculation Sets*****\n")
    writeToTracker('CTE_K', f"System: {'-'.join(elements)}\n")
    writeToTracker('CTE_K', f"Database: TCHEA8\n")

    # Group compositions by active element set to minimize TC reinitializations
    print("Organizing compositions by element sets...")
    tic = time.time()

    Els = []
    for row in range(results_df.shape[0]):
        comp = results_df.iloc[row][elements]
        active_el = list(compress(elements, list(comp > 0)))
        if active_el not in Els:
            Els.append(active_el)
        if row % 1000 == 0:
            toc = time.time()
            print(f'{round(row / results_df.shape[0] * 100, 1)}% done gathering systems in {round(toc - tic, 1)}s')

    # Reorganize dataframe by element groups
    print("Reorganizing dataframe...")
    results_df2 = pd.DataFrame()
    for El_i in Els:
        cond = (np.all(results_df[El_i] > 0, axis=1)) & \
               (np.sum(results_df[El_i], 1) > 1 - 1e-9) & \
               (np.sum(results_df[El_i], 1) < 1 + 1e-9)
        results_df2 = pd.concat([results_df2, results_df[cond]], axis=0)
        toc = time.time()
        print(f'{round(Els.index(El_i) / len(Els) * 100, 1)}% done rearranging in {round(toc - tic, 1)}s')

    results_df = results_df2.reset_index(drop=True)

    # Create calculation batches
    print("Creating calculation batches...")
    indices = results_df.index
    prev_active_el = []
    parameters = []
    count = 0
    new_calc_dict = {"INDICES": [], "COMP": [], "ACT_EL": []}

    for i in indices:
        comp = results_df.loc[i][elements]
        active_el = list(compress(elements, list(comp > 0)))

        # Start new batch if element set changes or batch size reached
        if (active_el != prev_active_el) or (count == batch_size):
            if new_calc_dict["INDICES"]:
                new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
                new_calc_dict["ACT_EL"] = prev_active_el

                # Check if already completed
                if not os.path.exists(f"CalcFiles/Results_Set_ReCalc_{new_calc_dict['INDICES'][0]}.csv"):
                    parameters.append(new_calc_dict)
                    writeToTracker('CTE_K', f"Batch added: Start Index {new_calc_dict['INDICES'][0]}\n")
                else:
                    writeToTracker('CTE_K', f"Batch already completed: Start Index {new_calc_dict['INDICES'][0]}\n")

                new_calc_dict = {"INDICES": [], "COMP": [], "ACT_EL": []}
            count = 0

        new_calc_dict["INDICES"].append(i)
        prev_active_el = active_el
        count += 1

    # Add the last batch
    if new_calc_dict["INDICES"]:
        new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
        new_calc_dict["ACT_EL"] = prev_active_el
        if not os.path.exists(f"CalcFiles/Results_Set_ReCalc_{new_calc_dict['INDICES'][0]}.csv"):
            parameters.append(new_calc_dict)
            writeToTracker('CTE_K', f"Batch added: Start Index {new_calc_dict['INDICES'][0]}\n")
        else:
            writeToTracker('CTE_K', f"Batch already completed: Start Index {new_calc_dict['INDICES'][0]}\n")

    writeToTracker('CTE_K', "*****Calculation Sets Generated*****\n")
    print(f"Created {len(parameters)} calculation batches")

    # Run calculations in parallel
    print(f"Starting calculations with {n_workers} workers...")
    completed_calculations = []

    with concurrent.futures.ProcessPoolExecutor(n_workers) as executor:
        for result_from_process in zip(parameters, executor.map(calculate_properties, parameters)):
            params, results = result_from_process
            if results == "Calculation Completed":
                completed_calculations.append('Completed')
                print(f"Completed batch starting at index {params['INDICES'][0]}")

    writeToTracker('CTE_K',
                   f"**********Calculations Finished - {len(parameters) - len(completed_calculations)} Incomplete**********\n")

    print(f"\nAll calculations complete!")
    print(f"Successfully completed: {len(completed_calculations)}/{len(parameters)} batches")