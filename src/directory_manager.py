import os
import logging as log
from datetime import datetime

class DirectoryManager:
    def __init__(self, parent_folder="data_output"):
        """
        Initializes the directory manager
        
        Args:
            parent_folder (str, optional): The parent folder for the output directory. Defaults to "data_output".
        """
        self.parent_folder = parent_folder
        self.date_str = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(self.parent_folder, self.date_str)
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = None  # Initialized in __setup_logging()

    def __setup_logging(self, log_dir):
        """
        Sets up logging for the DirectoryManager in the specified log directory
        
        Args:
            log_dir (str): The directory to store the log files
        """
        log_file = os.path.join(log_dir, "main.log")
        os.makedirs(log_dir, exist_ok=True)
        log.basicConfig(
            level=log.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                log.FileHandler(log_file),
                log.StreamHandler()
            ]
        )
        self.logger = log.getLogger("DirectoryManager")
        self.logger.info(f"Initialized DirectoryManager with output root: {self.output_dir}")

    def create_output_structure(self, subfolder, subsubfolder):
        """
        Create output structure for better directory structure
        
        Args:
            subfolder (str): The subfolder within the output directory
            subsubfolder (str): The subsubfolder within the output directory
        """
        base_path = os.path.join(self.output_dir, subfolder, subsubfolder)
        extrapolation_path = os.path.join(base_path, "extrapolation_files")
        photometry_path = os.path.join(base_path, "photometry_files")
        photometry_extrapolation_path = os.path.join(photometry_path, "extrapolation")
        log_path = os.path.join(base_path, "logs")

        os.makedirs(extrapolation_path, exist_ok=True)
        os.makedirs(photometry_path, exist_ok=True)
        os.makedirs(photometry_extrapolation_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        self.__setup_logging(log_path)
        self.logger.info(f"Created directory structure for {subfolder}/{subsubfolder}")

        return {
            "base_path": base_path,
            "extrapolation_path": extrapolation_path,
            "photometry_path": photometry_path,
            "photometry_extrapolation_path": photometry_extrapolation_path,
            "log_path": log_path,
            "timestamp_dir": self.output_dir,
        }

    def get_photometry_file_path(self, subfolder, subsubfolder, filename):
        """
        Get the path to the photometry file
        """
        return os.path.join(
            self.output_dir, subfolder, subsubfolder,
            "photometry_files",
            f"concentric-photometry_{filename}.csv"
        )

    def get_extrapolation_file_path(self, subfolder, subsubfolder, filename):
        """
        Get the path to the extrapolation file
        """
        return os.path.join(
            self.output_dir, subfolder, subsubfolder,
            "extrapolation_files",
            f"concentric-extrapolation_{filename}.csv"
        )

    def get_extrapolation_plot_path(self, subfolder, subsubfolder, filename):
        """
        Get the path to the extrapolation plot file
        """
        plot_filename = f"{os.path.splitext(filename)[0]}_extrapolation.png"
        return os.path.join(
            self.output_dir, subfolder, subsubfolder,
            "extrapolation_files",
            plot_filename
        )

    def get_magnitude_diff_file_path(self, subfolder, subsubfolder, filename):
        """
        Get the path to the different magnitude file
        """
        return os.path.join(
            self.output_dir, subfolder, subsubfolder,
            "photometry_files",
            "extrapolation",
            filename
        )
