import numpy as np
import pandas as pd
import os
import logging as log
from photutils.aperture import CircularAperture, aperture_photometry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Configure logging
log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

class ExtrapolationPhotometry:
    def __init__(self, target_data, comparison_data, check_data, min_radius, max_radius, dir_manager, subfolder, subsubfolder, logger=None):
        """
        Initialize the ExtrapolationPhotometry class.

        Args:
            target_data (dict): Dictionary containing target data for each file.
            comparison_data (dict): Dictionary containing comparison data for each file.
            check_data (dict): Dictionary containing check data for each file.
            min_radius (int): Minimum radius to use for photometry.
            max_radius (int): Maximum radius to use for photometry.
            dir_manager (DirectoryManager): DirectoryManager instance for file organization.
            subfolder (str): Subfolder within the output directory.
            subsubfolder (str): Subsubfolder within the subfolder.
            logger (logging.Logger): Logger instance for logging.
        """
        
        self.target_data = target_data
        self.comparison_data = comparison_data
        self.check_data = check_data
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.radii =  np.arange(min_radius, max_radius + 1, 1)
        self.dir_manager = dir_manager
        self.subfolder = subfolder
        self.subsubfolder = subsubfolder
        
        if logger is None:
            self.logger = log.getLogger('ExtrapolationPhotometry')
        else:
            self.logger = logger
            
        self.dirs = self.dir_manager.create_output_structure(subfolder, subsubfolder)
        
        self.logger.info(f"ExtrapolationPhotometry initialized with radius range: {min_radius} to {max_radius}")
        self.logger.info(f"Output directories: {self.dirs}")
    
    def __linear_fit(self, x, a, b):
        """
        Linear fitting function for curve fitting.
        
        Args:
            x (float): Independent variable
            a (float): Slope parameter
            b (float): Intercept parameter
            
        Returns:
            float: Result of linear function a*x + b
        """
        return a * x + b
    
    @staticmethod
    def __safe_log10(value, min_value=1e-10):
        """
        Calculate log10 safely with minimum value protection.
        
        Args:
            value (float): Input value
            min_value (float, optional): Minimum allowed value. Defaults to 1e-10.
            
        Returns:
            float: log10 of the input value (or min_value if value <= 0)
        """
        return np.log10(max(value, min_value))
    
    def __calculate_differential_magnitude(self, flux1, flux2):
        """
        Calculate magnitude difference between two fluxes.
        
        Args:
            flux1 (float): First flux value
            flux2 (float): Second flux value
            
        Returns:
            float: Magnitude difference (-2.5 * log10(flux1/flux2))
        """
        return -2.5 * self.__safe_log10(flux1 / flux2)

    def perform_photometry(self):
        """
        Perform aperture photometry on all stars for the configured radius range.
        
        Returns:
            dict: Dictionary containing photometry results for each file.
        """
        
        self.logger.info("Starting photometry...")
        
        if not all([self.target_data, self.comparison_data, self.check_data]):
            self.logger.error("Missing data for photometry.")
            return None
        
        results = {}
        
        # Perform aperture photometry on all stars for the configured radius range
        for file_name in self.target_data.keys():
            self.logger.info(f"Processing file: {file_name}")
            
            try:
                target_flux = self.__perform_aperture_photometry(self.target_data[file_name])
                comparison_flux = self.__perform_aperture_photometry(self.comparison_data[file_name])
                check_flux = self.__perform_aperture_photometry(self.check_data[file_name])
                
                # Store the photometry results
                results[file_name] = {
                    "target": target_flux,
                    "comparison": comparison_flux,
                    "check": check_flux
                }
                
                csv_path = self.dir_manager.get_photometry_file_path(
                    self.subfolder,
                    self.subsubfolder,
                    file_name
                )
                
                # Save the photometry results to a CSV
                df = pd.DataFrame(
                    {
                        "radius": self.radii,
                        "target_flux": target_flux,
                        "comparison_flux": comparison_flux,
                        "check_flux": check_flux
                    }
                )
                
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Photometry results saved: {csv_path}")
                
            except Exception as e:
                self.logger.info(f"Error processing {file_name} : {e}")
        
        return results
    
    def __perform_aperture_photometry(self, data):
        """
        Perform aperture photometry on a single star image with robust error handling.
        
        Args:
            data (numpy.ndarray): 2D array containing star image data
            
        Returns:
            numpy.ndarray: Array of flux values for each aperture radius
            None: If photometry fails
        """
        try:
            # Validate input data
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                self.logger.error("Invalid input data - must be 2D numpy array")
                return None
                
            height, width = data.shape
            center = (width // 2, height // 2)
            
            # Verify center is within image bounds
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                self.logger.error(f"Center point {center} outside image bounds {data.shape}")
                return None
                
            fluxes = []
            valid_radii = []
            
            for r in self.radii:
                try:
                    # Check if aperture stays within image bounds
                    if (center[0] - r < 0 or center[0] + r >= width or 
                        center[1] - r < 0 or center[1] + r >= height):
                        self.logger.warning(f"Skipping radius {r} - aperture extends outside image")
                        continue
                        
                    aperture = CircularAperture([center], r=r)
                    phot_table = aperture_photometry(data, aperture)
                    flux = phot_table['aperture_sum'][0]
                    
                    # Check for invalid flux values
                    if np.isnan(flux) or np.isinf(flux):
                        self.logger.warning(f"Invalid flux value {flux} for radius {r}")
                        continue
                        
                    fluxes.append(flux)
                    valid_radii.append(r)
                    
                except Exception as e:
                    self.logger.warning(f"Photometry failed for radius {r}: {str(e)}")
                    continue
                    
            if len(fluxes) != len(self.radii):
                self.logger.warning(f"Only got {len(fluxes)}/{len(self.radii)} successful measurements")
                
            if not fluxes:
                self.logger.error("No valid flux measurements obtained")
                return None
                
            return np.array(fluxes) if fluxes else None
            
        except Exception as e:
            self.logger.error(f"Photometry failed completely: {str(e)}")
            return None
    
    def extrapolate_to_zero(self, photometry_results):
        """
        Extrapolate flux measurements to zero radius using linear regression.
        
        Args:
            photometry_results (dict): Results from perform_photometry()
            
        Returns:
            dict: Dictionary containing extrapolation results for each file.
        """
        self.logger.info("Starting flux extrapolation...")

        extrapolation_results = {}

        # Perform flux regression
        for file_name, results in photometry_results.items():
            try:
                # Ensure that the arrays are of the same dtype
                target_flux = np.asarray(results["target"], dtype=np.float64)
                comparison_flux = np.asarray(results["comparison"], dtype=np.float64)
                check_flux = np.asarray(results["check"], dtype=np.float64)
                
                # Calculate the area of the circular photometry
                radii = np.asarray(self.radii, dtype=float)
                areas = np.pi * np.square(radii)
                areas = np.asarray(areas, dtype=np.float64)

                self.logger.info(f"areas shape: {areas.shape}")
                self.logger.info(f"target flux shape: {target_flux.shape}")
                self.logger.info(f"comparison flux shape: {comparison_flux.shape}")
                self.logger.info(f"check flux shape: {check_flux.shape}")

                # Ensure that arrays are of the same length
                if not (len(target_flux) == len(comparison_flux) == len(check_flux)):
                    self.logger.error(f"Inconsistent array lengths for {file_name}")
                    self.logger.error(f"Target flux length: {len(target_flux)}")
                    self.logger.error(f"Comparison flux length: {len(comparison_flux)}")
                    self.logger.error(f"Check flux length: {len(check_flux)}")
                    continue

                if len(radii) != len(target_flux):
                    self.logger.error(f"Mismatch: radii length ({len(radii)}) vs flux length ({len(target_flux)})")
                    continue

                # Check for NaN or Inf values again
                if np.isnan(areas).any() or np.isnan(target_flux).any() or np.isnan(comparison_flux).any() or np.isnan(check_flux).any():
                    self.logger.error(f"NaN detected in input data for {file_name}")
                    self.logger.error(f"areas: {areas}")
                    self.logger.error(f"target_flux: {target_flux}")
                    self.logger.error(f"comparison_flux: {comparison_flux}")
                    self.logger.error(f"check_flux: {check_flux}")
                    continue

                if np.isinf(areas).any() or np.isinf(target_flux).any() or np.isinf(comparison_flux).any() or np.isinf(check_flux).any():
                    self.logger.error(f"Infinite values detected in input data for {file_name}")
                    continue

                self.logger.info(f"Final array lengths: {len(target_flux)}, {len(comparison_flux)}, {len(check_flux)}, {len(areas)}")

                if len(areas) != len(target_flux):
                    self.logger.error(f"Final mismatch in lengths: areas ({len(areas)}) vs target_flux ({len(target_flux)})")
                    continue
                
                if not (len(areas) == len(target_flux) == len(comparison_flux) == len(check_flux)):
                    self.logger.error(f"Final length mismatch - Areas: {len(areas)}, "
                                    f"Target: {len(target_flux)}, "
                                    f"Comparison: {len(comparison_flux)}, "
                                    f"Check: {len(check_flux)}")
                    continue
                # Ensure that the variables are have the same length
                min_length = min(len(radii), len(target_flux), len(comparison_flux), len(check_flux))
                radii = radii[:min_length]
                target_flux = target_flux[:min_length]
                comparison_flux = comparison_flux[:min_length]
                check_flux = check_flux[:min_length]
                areas = np.pi * np.square(radii[:min_length])
                                            
                self.logger.debug(f"Final array contents - Areas: {areas}")
                self.logger.debug(f"Target Flux: {target_flux}")
                self.logger.debug(f"Comparison Flux: {comparison_flux}")
                self.logger.debug(f"Check Flux: {check_flux}")
                
                self.logger.info(f"Passing to curve_fit - Areas: {areas}, Target Flux: {target_flux}")

                # Try to perform the curve fit for extrapolation
                try:
                    target_popt, _ = curve_fit(self.__linear_fit, areas, target_flux)
                    comparison_popt, _ = curve_fit(self.__linear_fit, areas, comparison_flux)
                    check_popt, _ = curve_fit(self.__linear_fit, areas, check_flux)
                except RuntimeError as e:
                    self.logger.error(f"Curve fit failed for {file_name}: {e}")
                    self.logger.error(f"Areas: {areas}")
                    self.logger.error(f"Target Flux: {target_flux}")
                    self.logger.error(f"Comparison Flux: {comparison_flux}")
                    self.logger.error(f"Check Flux: {check_flux}")
                    continue

                # Store the extrapolation results
                target_intercept = target_popt
                comparison_intercept = comparison_popt
                check_intercept = check_popt

                # Calculate magnitude differences
                mag_diff_tar_comp = -2.5 * np.log10(max(target_intercept[1], 1e-10) / max(comparison_intercept[1], 1e-10))
                mag_diff_comp_check = -2.5 * np.log10(max(comparison_intercept[1], 1e-10) / max(check_intercept[1], 1e-10))
                mag_diff_tar_check = -2.5 * np.log10(max(target_intercept[1], 1e-10) / max(check_intercept[1], 1e-10))

                # Store the extrapolation results
                frame_id = os.path.splitext(os.path.basename(file_name))[0]

                extrapolation_results[file_name] = {
                    "frame_id": frame_id,
                    "target_intercept": target_intercept,
                    "comparison_intercept": comparison_intercept,
                    "check_intercept": check_intercept,
                    "magnitude_difference_target_comparison": mag_diff_tar_comp,
                    "magnitude_difference_comparison_check": mag_diff_comp_check,
                    "magnitude_difference_target_check": mag_diff_tar_check
                }

                df = pd.DataFrame(
                    {
                        "frame_id": frame_id,
                        "radius": radii,
                        "target_flux": target_flux,
                        "comparison_flux": comparison_flux,
                        "check_flux": check_flux,
                        "target_intercept": target_intercept,
                        "comparison_intercept": comparison_intercept,
                        "check_intercept": check_intercept,
                        "magnitude_difference_target_comparison": mag_diff_tar_comp,
                        "magnitude_difference_comparison_check": mag_diff_comp_check,
                        "magnitude_difference_target_check": mag_diff_tar_check
                    }
                )

                # Save the extrapolation results to a CSV file
                csv_path = self.dir_manager.get_extrapolation_file_path(
                    self.subfolder,
                    self.subsubfolder,
                    file_name
                )

                df.to_csv(csv_path, index=False)
                self.logger.info(f"Flux extrapolation results saved: {csv_path}")

                self.__create_extrapolation_plot(
                    file_name,
                    areas,
                    target_flux,
                    comparison_flux,
                    check_flux,
                    target_intercept,
                    comparison_intercept,
                    check_intercept
                )

            except Exception as e:
                self.logger.error(f"Error extrapolation for {file_name} : {e}")

        self.logger.info("Extrapolation completed.")
        return extrapolation_results
    
    def __create_extrapolation_plot(self, file_name, areas, target_flux, comparison_flux, check_flux, target_intercept, comparison_intercept, check_intercept):
        """
        Create plot showing flux vs. aperture area with linear fits.
        
        Args:
            file_name (str): Name of the file being processed
            areas (numpy.ndarray): Array of aperture areas
            target_flux (numpy.ndarray): Target star flux values
            comparison_flux (numpy.ndarray): Comparison star flux values
            check_flux (numpy.ndarray): Check star flux values
            target_intercept (list): [slope, intercept] for target star fit
            comparison_intercept (list): [slope, intercept] for comparison star fit
            check_intercept (list): [slope, intercept] for check star fit
        """
        try:
            self.logger.info(f"Creating extrapolation plot for: {file_name}")
            
            # Define the smoothing parameters
            x_smooth =  np.linspace(0, np.max(areas), 100)
            
            # Perform the extrapolation with linear fitting
            target_fit = self.__linear_fit(x_smooth, target_intercept[0], target_intercept[1])
            comparison_fit = self.__linear_fit(x_smooth, comparison_intercept[0], comparison_intercept[1])
            check_fit = self.__linear_fit(x_smooth, check_intercept[0], check_intercept[1])
            
            # Create the plot
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=["Target", "Comparison", "Check"],
                shared_xaxes=True,
                shared_yaxes=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=areas,
                    y=target_flux,
                    mode='markers',
                    name='Target Data',
                    marker=dict(color='blue')
                ),
                row=1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=target_fit,
                    mode='lines',
                    name='Target Fit',
                    line=dict(color='blue')
                ),
                row=1,
                col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=areas,
                    y=comparison_flux,
                    mode='markers',
                    name='Comparison Data',
                    marker=dict(color='red')
                ),
                row=1,
                col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=areas,
                    y=comparison_fit,
                    mode='lines',
                    name='Comparison Fit',
                    line=dict(color='red')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=areas,
                    y=check_flux,
                    mode='markers',
                    name='Check Data',
                    marker=dict(color='green')
                ),
                row=1,
                col=3
            )
            
            fig.add_trace(
                go.Scatter(
                    x=areas,
                    y=check_fit,
                    mode='lines',
                    name='Check Fit',
                    line=dict(color='green')
                )
            )
            
            fig.update_layout(
                title_text=f"Extrapolation to Zero Radius - {os.path.basename(file_name)}",
                height=1000,
                width=2000,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Aperture Area", row=1, col=1)
            fig.update_xaxes(title_text="Aperture Area", row=1, col=2)
            fig.update_xaxes(title_text="Aperture Area", row=1, col=3)
            
            fig.update_yaxes(title_text="Flux", row=1, col=1)
            fig.update_yaxes(title_text="Flux", row=1, col=2)
            fig.update_yaxes(title_text="Flux", row=1, col=3)
            
            fig.add_annotation(
                x=0,
                y=target_intercept[1],
                text=f"Intercept: {target_intercept[1]:.2f}",
                showarrow=True,
                arrowhead=2,
                row=1,
                col=1
            )
            fig.add_annotation(
                x=0,
                y=comparison_intercept[1],
                text=f"Intercept: {comparison_intercept[1]:.2f}",
                showarrow=True,
                arrowhead=2,
                row=1,
                col=2
            )
            fig.add_annotation(
                x=0,
                y=check_intercept[1],
                text=f"Intercept: {check_intercept[1]:.2f}",
                showarrow=True,
                arrowhead=2,
                row=1,
                col=3
            )
            
            plot_path = self.dir_manager.get_extrapolation_plot_path(
                self.subfolder,
                self.subsubfolder,
                file_name
            )
            fig.write_image(plot_path)
            self.logger.info(f"Saved extrapolation plot to {plot_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating extrapolation plot for: {file_name}: {e}")

    def compute_magnitude_differences(self, extrapolation_results):
        """
        Compute magnitude differences between stars from extrapolation results.
        
        Args:
            extrapolation_results (dict): Results from extrapolate_to_zero()
            
        Returns:
            pd.DataFrame: DataFrame
        """
        
        self.logger.info("Computing magnitude differences...")
        data = []

        # Compute magnitude differences
        for file_name, results in extrapolation_results.items():
            try:
                frame_id = results["frame_id"]
                target_flux = results["target_intercept"][1]
                comparison_flux = results["comparison_intercept"][1]
                check_flux = results["check_intercept"][1]

                # Ensure positive flux values to avoid NaN
                target_comp_diff = -2.5 * self.__safe_log10(target_flux / max(comparison_flux, 1e-10))
                comp_check_diff = -2.5 * self.__safe_log10(comparison_flux / max(check_flux, 1e-10))
                target_check_diff = -2.5 * self.__safe_log10(target_flux / max(check_flux, 1e-10))

                # Check for outliers and stabilize extreme values
                if abs(target_comp_diff) > 5:
                    target_comp_diff = np.clip(target_comp_diff, -5, 5)
                if abs(comp_check_diff) > 5:
                    comp_check_diff = np.clip(comp_check_diff, -5, 5)
                if abs(target_check_diff) > 5:
                    target_check_diff = np.clip(target_check_diff, -5, 5)

                self.logger.info(f"Computed magnitude differences for: {file_name}")

                # Append to data
                data.append({
                    "filename": file_name,
                    "frame_id": frame_id,
                    "target_comparison_diff": target_comp_diff,
                    "comparison_check_diff": comp_check_diff,
                    "target_check_diff": target_check_diff
                })

            except Exception as e:
                self.logger.error(f"Error computing magnitude difference for {file_name}: {e}")
        
        # Save to CSV
        df = pd.DataFrame(data)
        csv_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_magnitude_differences.csv"
        csv_path = self.dir_manager.get_magnitude_diff_file_path(
            self.subfolder,
            self.subsubfolder,
            csv_filename
        )

        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved magnitude differences to {csv_path}")
        return df

    def visualize_magnitude_differences(self, magnitude_df):
        """
        Create plot of magnitude differences over time.
        
        Args:
            magnitude_df (pd.DataFrame): Results from compute_magnitude_differences()
            
        Returns:
            plotly.graph_objects.Figure: The generated figure
        """
        self.logger.info("Visualizing magnitude differences...")
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=magnitude_df.index,
                y=magnitude_df["target_comparison_diff"],
                mode='markers+lines',
                name='Target - Comparison',
                marker=dict(color='blue')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=magnitude_df.index,
                y=magnitude_df["comparison_check_diff"],
                mode='markers+lines',
                name='Check - Comparison',
                marker=dict(color='red')
            )
        )

        fig.update_layout(
            title_text="Magnitude Differences",
            xaxis_title="Frame Index",
            yaxis_title="Magnitude Difference",
            height=600,
            width=1000
        )

        plot_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_magnitude_differences.png"
        plot_path = self.dir_manager.get_magnitude_diff_file_path(
            self.subfolder,
            self.subsubfolder,
            plot_filename
        )

        fig.write_image(plot_path)
        self.logger.info(f"Saved magnitude differences plot to {plot_path}")
        return fig
    
    def perform_flux_regression(self, photometry_results, reference_radius=None):
        """
        Perform linear regression on flux vs. aperture areas and predict flux at reference radius.
        
        Args:
            photometry_results (dict): Results from perform_photometry()
            reference_radius (float, optional): Radius to predict flux at. Defaults to median radius.
            
        Returns:
            dict: Dictionary containing regression results for each file, with keys:
                - "frame_id": frame identifier
                - flux predictions at reference radius
                - magnitude differences between stars
        """
        self.logger.info("Starting flux regression...")
        
        regression_results = {}
        
        for file_name, results in photometry_results.items():
            try:
                self.logger.info(f"Performing regression for: {file_name}")
                
                target_flux = results["target"]
                comparison_flux = results["comparison"]
                check_flux = results["check"]
                
                # Convert radii to areas
                areas = np.pi * np.square(self.radii)
                
                # Define the reference radius
                if reference_radius is None:
                    reference_radius = np.median(self.radii)  # Default to median radius
                    
                reference_area = np.pi * (reference_radius ** 2)

                # Fit linear regression model
                target_model = LinearRegression().fit(areas.reshape(-1, 1), target_flux)
                comparison_model = LinearRegression().fit(areas.reshape(-1, 1), comparison_flux)
                check_model = LinearRegression().fit(areas.reshape(-1, 1), check_flux)

                # Predict flux at the reference radius
                target_pred = target_model.predict([[reference_area]])[0]
                comparison_pred = comparison_model.predict([[reference_area]])[0]
                check_pred = check_model.predict([[reference_area]])[0]

                # Compute magnitude differences
                mag_diff_tar_comp = self.__calculate_differential_magnitude(target_pred, comparison_pred)
                mag_diff_comp_check = self.__calculate_differential_magnitude(comparison_pred, check_pred)

                # Extract frame ID from filename
                frame_id = os.path.splitext(os.path.basename(file_name))[0]

                regression_results[file_name] = {
                    "frame_id": frame_id,
                    "target_flux": target_pred,
                    "comparison_flux": comparison_pred,
                    "check_flux": check_pred,
                    "magnitude_difference_target_comparison": mag_diff_tar_comp,
                    "magnitude_difference_comparison_check": mag_diff_comp_check
                }

                # Save to CSV
                df = pd.DataFrame({
                    "frame_id": [frame_id],
                    "reference_radius": [reference_radius],
                    "target_flux": [target_pred],
                    "comparison_flux": [comparison_pred],
                    "check_flux": [check_pred],
                    "magnitude_difference_target_comparison": [mag_diff_tar_comp],
                    "magnitude_difference_target_check": [mag_diff_comp_check]
                })
                
                csv_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_flux_regression_magnitude_differences.csv"

                
                csv_path = self.dir_manager.get_extrapolation_file_path(
                    self.subfolder,
                    self.subsubfolder,
                    csv_filename,
                )

                df.to_csv(csv_path, index=False)
                self.logger.info(f"Flux regression results saved: {csv_path}")

            except Exception as e:
                self.logger.error(f"Error in regression for {file_name}: {e}")

        self.logger.info("Flux regression completed.")
        return regression_results
