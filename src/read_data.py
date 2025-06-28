import os
import numpy as np
import logging as log
from glob import glob
from astropy.io import fits
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataReader:
    def __init__(self, base_path, data_path, logger=None):
        """
        Initialize the DataReader class.

        Args:
            base_path (str): Base directory path.
            data_path (str): Relative path to the data directory.
        """
        self.base_path = base_path
        self.data_path = data_path
        self.files_path = {}  # Store paths of .fits files
        self.data = {}  # Store data from .fits files

        # Set up logging
        log.basicConfig(level=log.INFO, format="%(levelname)s: %(message)s")

        # Set up logging
        if logger is None:
            self.logger = log.getLogger("DataReader")
        else:
            self.logger = logger
        
        self.logger.info(f"DataReader initialized with base path: {base_path} and data path: {data_path}")


    def read_files(self):
        """
        Recursively finds all .fits files in the specified directory and stores their paths.
        
        Returns:
            dict: A dictionary mapping file names to their full paths.
        """
        # Combine base path and data path
        folder_path = os.path.join(self.base_path, self.data_path)
        
        # Check if the directory exists
        if not os.path.exists(folder_path):
            self.logger.error(f"Directory '{folder_path}' does not exist.")
            return {}

        # Find all .fits files
        fits_files = glob(os.path.join(folder_path, "**", "*.fits"), recursive=True)

        if not fits_files:
            self.logger.warning("No .fits files found in the directory.")
            return {}

        # Store file paths
        self.files_path = {os.path.basename(f): f for f in fits_files}
        self.logger.info(f"Found {len(self.files_path)} .fits files.")

        return self.files_path

    def read_data(self, files_path=None):
        """
        Reads data from the found .fits files and stores them in a dictionary.
        
        Returns:
            dict: A dictionary mapping file names to their data.
        """
        # Check if we have file paths
        if files_path is None:
            files_path = self.files_path
            
        if not files_path:
            self.logger.warning("No .fits files found. Please run 'read_files()' first.")
            return {}

        # Read data from.fits files and store in the dictionary
        for file, path in files_path.items():
            try:
                with fits.open(path, memmap=False) as hdul:
                    if hdul[0].data is None:
                        self.logger.warning(f"File '{file}' contains no data.")
                    else:
                        self.data[file] = hdul[0].data
            except Exception as e:
                self.logger.error(f"Error reading '{file}': {e}")

        self.logger.info(f"Successfully read {len(self.data)} .fits files.")
        return self.data

    def visualize_fits(self, files=None, num_cols=3, title="FITS Files Visualization"):
        """
        Visualizes the data from the found.fits files using Plotly.
        
        Args:
            files (list, optional): Names of the.fits files to visualize. Defaults to None.
            num_cols (int, optional): Number of columns in the visualization grid. Defaults to 3.
            title (str, optional): Title of the visualization. Defaults to "FITS Files Visualization".

        Returns:
            Plotly Figure: The visualization of the.fits files.
        """
        if not self.data:
            self.logger.warning("No data available to visualize. Please run 'read_data()' first.")
            return None
        
        if files is None:
            files = list(self.data.keys())
        else:
            files = [f for f in files if f in self.data]
        
        if not files:
            self.logger.warning("No valid files to visualized.")
            return None
        
        # Define the number of rows and columns
        num_files = len(files)
        num_rows = (num_files + num_cols - 1) // num_cols
        
        # Create the plotly figure
        fig = make_subplots(
            rows = num_rows,
            cols = num_cols,
            subplot_titles = files,
            horizontal_spacing = 0.05,
            vertical_spacing = 0.1
        )
        
        for i, file in enumerate(files):
            row = i //num_cols + 1
            col = i % num_cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z = self.data[file],
                    colorscale="viridis",
                    colorbar=dict(title="Intensity") if col == num_cols else dict(showticklabels=False),
                    showscale=(col == num_cols)
                ),
                row=row,
                col=col
            )
        
        fig.update_layout(
            title=title,
            height = 1200 * num_rows,
            width = 1500 * num_cols,
            template="plotly_dark"
        )
        
        self.logger.info(f"Created visualization for {len(files)} FITS files.")
        return fig
    
    def visualize_single_fits(self, file_name, title=None):
        """
        Visualizes the data from a specific.fits file using Plotly.
        
        Args:
            file_name (str): Name of the.fits file to visualize.
            title (str, optional): Title of the visualization. Defaults to None.
        
        Returns:
            Plotly Figure: The visualization of the specified.fits file.
        """
        if file_name not in self.data:
            self.logger.warning(f"File '{file_name}' not found.")
            return None
        
        if title is None:
            title = f"Visualization of {file_name}"
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z = self.data[file_name],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            )
        )
        
        fig.update_layout(
            title=title,
            height = 900,
            width = 1200,
            template="plotly_dark"
        )
        
        self.logger.info(f"Created visualization for file '{file_name}'.")
        return fig