import numpy as np
import pandas as pd
import cv2
from astropy.io import fits
from astropy.visualization import simple_norm
from photutils.aperture import CircularAperture, aperture_photometry
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging as log
import os

class ObjectDetection:
    def __init__(self, dataset, reference_image=None, star_chart=None, logger=None):
        """
        Initialize the ObjectDetection class.

        Args:
            dataset (dict): Dictionary of FITS data from DataReader.
            reference_image (str): Path to the reference image.
            star_chart (str): Path to the star chart image.
            logger (logging.Logger, optional): Logger instance. If None, a new logger will be created.
        """
        self.dataset = dataset
        self.reference_image = reference_image
        self.star_chart = star_chart
        
        # Initialize empty dictionaries for detected stars
        self.target = {}
        self.comparison = {}
        self.check = {}
        
        # Store reference positions
        self.target_refs = []
        self.comparison_refs = []
        self.check_refs = []
        
        # Store star positions
        self.target_pos = None
        self.comparison_pos = None
        self.check_pos = None

        # Set up logging
        if logger is None:
            self.logger = log.getLogger("ObjectDetection")
        else:
            self.logger = logger
        
        self.logger.info("ObjectDetection initialized.")
    
    def show_reference_and_chart(self):
        """
        Showing the reference and chart of the object given
        
        Returns:
            plotly.Figure: The plotly figure with the reference and chart.
        """
        self.logger.info("Showing reference image and star chart.")
        
        ref_image = self.__read_image(self.reference_image)
        chart_image = self.__read_image(self.star_chart)
        
        if ref_image is None or chart_image is None:
            self.logger.error("Failed to read reference image or star chart.")
            return None
        
        # Make plot of reference image and star chart with subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Reference Image", "Star Chart"),
            column_widths=[0.7, 0.3],
            horizontal_spacing=0.05
        )
        
        fig.add_trace(
            go.Heatmap(
                z=ref_image[::-1],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=chart_image[::-1],
                colorscale="gray",
                colorbar=dict(title="Intensity")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Reference Image & Star Chart",
            template="plotly_dark",
            width=1200,
            height=600
        )
        
        self.logger.info("Reference image and star chart displayed.")
        return fig
    
    def __read_image(self, image_path):
        """
        Read reference image and star chart using cv2
        
        Returns:
            numpy.ndarray: Image data as a numpy array. If the image cannot be read, returns None.
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.logger.error(f"Failed to read image at {image_path}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"Error reading image at {image_path}: {e}")
            return None
        
    def set_estimated_positions(self, estimated_target, estimated_comparison, estimated_check):
        """
        Set initial estimated positions for target, comparison, and check stars.

        Args:
            estimated_target (tuple): Estimated (x, y) position of the target star.
            estimated_comparison (tuple): Estimated (x, y) position of the comparison star.
            estimated_check (tuple): Estimated (x, y) position of the check star.
        """
        self.estimated_target = estimated_target
        self.estimated_comparison = estimated_comparison
        self.estimated_check = estimated_check

        self.logger.info(f"Estimated positions set. Target: {estimated_target}, Comparison: {estimated_comparison}, Check: {estimated_check}")
    
    def set_reference_stars(self, target_refs, comparison_refs, check_refs):
        """
        Set position of reference stars for target, comparison and check stars.
        
        Args:
            target_refs (list): List of (x, y) positions of reference stars for target star
            comparison_refs (list): List of (x, y) positions of reference stars for comparison star
            check_refs (list): List of (x, y) positions of reference stars for check star
        """
        self.target_refs = target_refs
        self.comparison_refs = comparison_refs
        self.check_refs = check_refs

        # Calculate reference centroids
        ref_target_centroid = self.__calculate_position(target_refs)
        ref_comparison_centroid = self.__calculate_position(comparison_refs)
        ref_check_centroid = self.__calculate_position(check_refs)

        self.logger.info(f"Reference centroids calculated: Target={ref_target_centroid}, Comparison={ref_comparison_centroid}, Check={ref_check_centroid}")

        # Compute offset between estimated and reference centroids, if estimated positions exist
        if hasattr(self, 'estimated_target') and self.estimated_target:
            offset_target = (ref_target_centroid[0] - self.estimated_target[0], ref_target_centroid[1] - self.estimated_target[1])
            self.target_pos = self.__apply_offset(self.estimated_target, offset_target)
        else:
            self.target_pos = ref_target_centroid

        if hasattr(self, 'estimated_comparison') and self.estimated_comparison:
            offset_comparison = (ref_comparison_centroid[0] - self.estimated_comparison[0], ref_comparison_centroid[1] - self.estimated_comparison[1])
            self.comparison_pos = self.__apply_offset(self.estimated_comparison, offset_comparison)
        else:
            self.comparison_pos = ref_comparison_centroid

        if hasattr(self, 'estimated_check') and self.estimated_check:
            offset_check = (ref_check_centroid[0] - self.estimated_check[0], ref_check_centroid[1] - self.estimated_check[1])
            self.check_pos = self.__apply_offset(self.estimated_check, offset_check)
        else:
            self.check_pos = ref_check_centroid

        self.logger.info(f"Final adjusted positions -> Target: {self.target_pos}, Comparison: {self.comparison_pos}, Check: {self.check_pos}")

    def __apply_offset(self, estimated_pos, offset):
        """
        Apply offset to an estimated position to calculate the corrected position.

        Args:
            estimated_pos (tuple): Estimated (x, y) position.
            offset (tuple): (dx, dy) offset to apply.

        Returns:
            tuple: Corrected (x, y) position.
        """
        corrected_x = int(estimated_pos[0] + offset[0])
        corrected_y = int(estimated_pos[1] + offset[1])
        return (corrected_x, corrected_y)


    def __calculate_position(self, positions):
        """
        Calculates the position of the objects based on the given positions

        Args:
            positions (list): List of (x, y) positions.
            
        Returns:
            tuple: The average (x, y) position.
        """
        if not positions:
            return None
        
        x_avg = sum(p[0] for p in positions) / len(positions)
        y_avg = sum(p[1] for p in positions) / len(positions)
    
        return (int(x_avg), int(y_avg))

    def refine_position(self, image, position, window_size=10):
        """
        Refine object position to ensure it is at the center of the detected frame.

        Args:
            image (np.array): 2D image array where the object is located.
            position (tuple): Estimated (x, y) position of the object.
            window_size (int): Size of the area around the position to analyze.

        Returns:
            tuple: Refined (x, y) position.
        """
        
        # Create a sub-image centered around the position
        x, y = position
        half_win = window_size // 2
            
        # Ensure the sub-image is within the image boundaries
        x_min, x_max = max(0, x - half_win), min(image.shape[1], x + half_win)
        y_min, y_max = max(0, y - half_win), min(image.shape[0], y + half_win)
            
        # Get the sub-image and find the centroid
        sub_image = image[y_min:y_max, x_min:x_max]
            
        # Check if the sub-image is empty
        if sub_image.size == 0:
            self.logger.warning(f"Sub-image is empty for position {position}")
            return position
            
        # Find the centroid of the sub-image and adjust the position accordingly
        centroid = np.array(np.unravel_index(np.argmax(sub_image), sub_image.shape))
        refined_x = x_min + centroid[1]
        refined_y = y_min + centroid[0]
            
        return (refined_x, refined_y)

    def align_and_detect_stars(self, crop_size=100):
        """
        Align and detect stars with position refinement.
        
        Args:
            crop_size (int): Size of the cropped image around the star.
        
        Returns:
            tuple: Dictionaries of target, comparison, and check star crops
        """

        self.logger.info("Starting star alignment and detection...")
        
        # Ensure that the images are the same size
        if crop_size % 2 != 0:
            crop_size += 1
            self.logger.info(f"Adjusted crop size to {crop_size} to ensure it's even.")
        
        # Get the target, comparison, and check images
        for file_name, data in self.dataset.items():
            self.logger.info(f"Processing file: {file_name}")
            
            # First, align the positions
            target_pos = self.target_pos
            comparison_pos = self.comparison_pos
            check_pos = self.check_pos
            
            # Then, refine the positions to ensure stars are centered
            target_crop = self.__crop_centered(data, target_pos, crop_size)
            refined_target_pos = self.refine_position(target_crop, 
                                                        (target_crop.shape[1]//2, target_crop.shape[0]//2))
            
            comparison_crop = self.__crop_centered(data, comparison_pos, crop_size)
            refined_comparison_pos = self.refine_position(comparison_crop, 
                                                            (comparison_crop.shape[1]//2, comparison_crop.shape[0]//2))
            
            check_crop = self.__crop_centered(data, check_pos, crop_size)
            refined_check_pos = self.refine_position(check_crop, 
                                                        (check_crop.shape[1]//2, check_crop.shape[0]//2))
            
            # Store the refined crops and positions
            self.target[file_name] = target_crop
            self.comparison[file_name] = comparison_crop
            self.check[file_name] = check_crop
            
            self.logger.info(f"Processed {file_name}: Refined Target={refined_target_pos}, Refined Comparison={refined_comparison_pos}, Refined Check={refined_check_pos}")
            
        self.logger.info("Star alignment and detection complete.")
        return self.target, self.comparison, self.check
    
    def __crop_centered(self, data, position, crop_size):
        """
        Crop the target to ensure that the target on the center of window
        
        Args:
            data (np.array): 2D image array.
            position (tuple): (x, y) position of the center of the crop.
            crop_size (int): Size of the cropped image.
            
        Returns:
            np.array: Cropped image
        """
        
        if position is None:
            return None
        
        # Ensure the crop size is odd
        x, y = position
        height, width = data.shape
        half_crop = crop_size // 2
        
        # Ensure the crop is within the image boundaries
        x_min = max(0, x - half_crop)
        x_max = min(width, x + half_crop)
        y_min = max(0, y - half_crop)
        y_max = min(height, y + half_crop)
        
        # Crop the data
        cropped = data[y_min:y_max, x_min:x_max]
        
        # Ensure the cropped size matches the requested size
        if cropped.shape != (crop_size, crop_size):
            self.logger.warning(f"Crop at {position} had to be adjusted to fit image boundaries.")
            self.logger.warning(f"Actual crop size: {cropped.shape}")
            self.logger.warning(f"Requested crop size: ({crop_size}, {crop_size})")
        
            padded = np.zeros((crop_size, crop_size), dtype=data.dtype)
            
            pad_y = (crop_size - cropped.shape[0]) // 2
            pad_x = (crop_size - cropped.shape[1]) // 2
            
            padded[pad_y:pad_y + cropped.shape[0], pad_x:pad_x + cropped.shape[1]] = cropped
            
            return padded
        
        return cropped
    
    def show_detected_stars(self):
        """
        Display the detected stars.
        
        Returns:
            go.Figure: Plotly figure with detected stars
        """
        if not self.target or not self.comparison or not self.check:
            self.logger.warning("No detected stars to display. Run align_and_detect_stars() first.")
            return None
        
        # Get the list of detected stars
        file_names = list(self.target.keys())
        
        if not file_names:
            self.logger.warning("No files to display.")
            return None
        
        first_file = file_names[0]
        last_file = file_names[-1]
        
        # Create the plotly figure
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                f"Target Star (First: {first_file})",
                f"Target Star (Last: {last_file})",
                f"Comparison Star (First: {first_file})",
                f"Comparison Star (Last: {last_file})",
                f"Check Star (First: {first_file})",
                f"Check Star (Last: {last_file})"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
    
        fig.add_trace(
            go.Heatmap(
                z=self.target[first_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
                go.Heatmap(
                z=self.target[last_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=self.comparison[first_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=self.comparison[last_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=self.check[first_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=self.check[last_file],
                colorscale="viridis",
                colorbar=dict(title="Intensity")
                ),
            row=3, col=2
            )
        
        fig.update_layout(
            title="Detected Stars (First and Last Frames)",
            template="plotly_dark",
            width = 1000,
            height = 1000
        )
    
        self.logger.info("Detected stars displayed.")
        return fig