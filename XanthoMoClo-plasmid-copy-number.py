"""Author: Audrey Ory
Date: 2024-11-18
Commentary:
Script to analyze qPCR data to quantify plasmid copy number.

This script creates a joint probability density function of the plasmid copies
and genome copies to create a joint estimate of the plasmid copy number per genome given the data.
Further, by fitting a gaussian to the joint estimate, this script derives a 95% confidence interval for the estimates.
"""
import glob
from scipy.optimize import root_scalar, curve_fit
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict
from scipy.integrate import quad
from QLD.PoissonJoint import PoissonJoint


"""
matrix_dir: path to a directory continaing your qPCR results. Label columns 1 through the total # of dilutions you did. Each row is a replicate dilution series. 
ouutput_dir: a path to a folder where you wish to deposit your output files.
"""

matrix_dir = '/Users/audreyory/Desktop/Springer Lab/qPCR/copy number qPCR/matrices'
all_files = os.listdir(matrix_dir)

output_dir = '//Users/audreyory/Desktop/Springer Lab/qPCR/copy number qPCR/matrices'
os.makedirs(output_dir, exist_ok=True)

"""
Fill in your strain names and plasmid names below. Note that the script assumes you name each matrix file '{strain} {plasmid} genome matrix.csv' and '{strain} {plasmid} plasmid matrix.csv'
"""

strains = ['RFI']
plasmids = ['o7']


collect = []
for strain in strains:
    for plasmid in plasmids: 

        # NaN values are assigned 0 and real CT values are assigned 1 

        genome_df = pd.read_csv(f"{matrix_dir}/{strain} {plasmid} genome matrix.csv").replace({np.nan: 0})
        genome_df[genome_df > 0] = 1
        
        plasmid_df = pd.read_csv(f"{matrix_dir}/{strain} {plasmid} plasmid matrix.csv").replace({np.nan: 0})
        plasmid_df[plasmid_df > 0] = 1
       
        # generating the genome and plasmid probability distributions 
        # given the data, how probable is it that a certain x # of DNA copies entered the dilution series? 

        X = np.logspace(-1, 2, 1000)
        genome_Y = PoissonJoint(X, genome_df, 2)
        plasmid_Y = PoissonJoint(X, plasmid_df, 2)

        joint_values = defaultdict(list)

        # generating the joint probability distribution for genome and plasmid 

        for i, x in enumerate(X): 
            joint_Y_array = plasmid_Y[i] * genome_Y
            joint_x_array = x/X 

            for jx, jy in zip(joint_x_array, joint_Y_array):
                jx = round(jx, 6)
                joint_values[jx].append(jy)

        # making an array of all the unique values for plasmid x/genome x 
    
        unique_joint_x = sorted(joint_values.keys())
        
        # summing up all the probabilities that have the same plasmid x/genome x value 
        
        summed_joint_Y = []
        for jx in unique_joint_x: 
            summed_joint_Y.append(sum(joint_values[jx]))

        # fit a gaussian to your data 

        # Transform x to log10 scale for fitting
        log10_x = np.log10(unique_joint_x)

        # Define the Gaussian function in log space
        def gaussian(log_x, a, b, c):
            """
            Gaussian function in log space.
            log_x: log10-transformed x-values
            a: amplitude (peak height)
            b: mean (center of the Gaussian in log10 space)
            c: standard deviation (spread in log10 space)
            """
            return a * np.exp(-((log_x - b) ** 2) / (2 * c ** 2))

        # Provide initial guesses for the parameters
        initial_guess = [max(summed_joint_Y), np.mean(log10_x), 1]

        # Perform the curve fitting
        params, covariance = curve_fit(gaussian, log10_x, summed_joint_Y, p0=initial_guess, maxfev=10000)

        # Extract the fitted parameters
        a_fit, b_fit, c_fit = params

        def fitted_gaussian(x):
            # Parameters from the fit
            a = a_fit
            b = b_fit
            c = c_fit
            # Transform x to log space, then calculate Gaussian
            log_x = np.log10(x)
            return a * np.exp(-((log_x - b) ** 2) / (2 * c ** 2))
        
        # visualize the fit 
        plt.plot(unique_joint_x, summed_joint_Y)
        x_values = np.logspace(-1, 2, 1000)
        y_values = fitted_gaussian(x_values)
        
        fit_df = pd.DataFrame({'x': x_values, 'y': y_values})
        fit_output_file = os.path.join(output_dir, f"{strain}_{plasmid}_joint_distribution_fit.csv")
        fit_df.to_csv(fit_output_file, index=False)

        plt.plot(x_values, y_values, linestyle='dotted', color='r')
        ax = plt.gca()
        ax.set_xscale("log")
        plt.savefig(f"{strain} {plasmid} copy number prob.png")
        plt.clf()

        # finding the x at which probability is at a maximum and finding the index 
        copy_number_best = [x for x,y in zip(unique_joint_x, summed_joint_Y) if y == max(summed_joint_Y)]
        best_index = unique_joint_x.index(copy_number_best)

        # finding the x at which my Gaussian reaches a maximum and finding the index 
        gaussian_max = x_values[np.argmax(y_values)]  # x corresponding to the max y value
        gaussian_min = x_values[np.argmin(y_values)]

        print(copy_number_best, gaussian_max)
       
        # finding the area under the Gaussain 


        lower_limit = min(x_values)
        upper_limit = max(x_values)
        area, error = quad(fitted_gaussian, lower_limit, upper_limit)
        print(area, lower_limit, upper_limit)
        target_area = area*0.95
        y_max = fitted_gaussian(gaussian_max)
        y_min = fitted_gaussian(gaussian_min)

        current_y = y_max
        step_size = 0.00001 * y_max 
        cumulative_area = 0
        x_lower, x_upper = None, None
        print(f"y max is {y_max}")


        # Updated find_x function with dynamic handling for very small y values
        def find_x(y_value, side):

            # Validate that the y_value is within the range of the Gaussian
            if y_value > y_max:
                raise ValueError(f"y_value {y_value} exceeds the peak y value {y_max}.")
            if y_value < 0:
                raise ValueError(f"y_value {y_value} is below zero, which is not valid for this Gaussian.")

            if side == "left":
                # Dynamically adjust the bracket if necessary
                try:
                    return root_scalar(lambda x: fitted_gaussian(x) - y_value, bracket=[lower_limit, gaussian_max]).root
                except ValueError:
                    raise ValueError(f"No root found for 'left' side with current_y={y_value}. Expand the bracket.")
            elif side == "right":
                try:
                    return root_scalar(lambda x: fitted_gaussian(x) - y_value, bracket=[gaussian_max, upper_limit]).root
                except ValueError:
                    raise ValueError(f"No root found for 'right' side with current_y={y_value}. Expand the bracket.")

        while cumulative_area < target_area: 
            # Decrease y
            current_y = max(current_y - step_size, y_min)

            # Find x values where Gaussian equals current_y
            try:
                x_lower = find_x(current_y, "left")
                x_upper = find_x(current_y, "right")
                last_valid_x_lower = x_lower 
            except ValueError as e:
                print(current_y)
                print(cumulative_area, target_area)
                break  # Stop the loop if no valid roots are found

            # Calculate the cumulative area between x_lower and x_upper
            if x_lower is not None and x_upper is not None:
                cumulative_area, _ = quad(fitted_gaussian, x_lower, x_upper)
                print(f"cumulative_area: {cumulative_area}, target_area: {target_area}, current_y: {current_y}, step size: {step_size}, delta y: {current_y - step_size}")

            if cumulative_area >= target_area:
                break


        collect.append({"strain": strain,
                        "plasmid": plasmid,
                        "distribution copy number estimate": copy_number_best,
                        "gaussian fit copy number estimate": gaussian_max,
                        "minimum x": x_lower,
                        "maximum x": x_upper})
        

df = pd.DataFrame(collect)
df.to_csv(f'{output_dir}/copy number results.csv', index=False)
