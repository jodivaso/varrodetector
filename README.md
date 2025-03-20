# VarroDetector

<p>VarroDetector is an open-source tool designed to identify and count Varroa mites in images of sticky sheets taken with smartphones.
The software runs in low-range computers (no GPU is needed). Neither installation nor internet connection is needed, just double-click
on the executable.</p>

<p>The detection process is based on a YOLOv11 nano model specifically trained on hundreds of images to identify Varroa mites.</p>

<kbd>
<img src="readme_video.gif" alt="VarroDetector example"/>
</kbd>

## Features

- **String Detection**: Automatically identifies and crops images based on guide strings
- **AI-powered Detection**: Uses YOLOv11 nano learning model to identify Varroa mites
- **Confidence Threshold**: Adjustable detection sensitivity per image or globally
- **Region of Interest (ROI)**: Define specific areas for mite counting
- **Subfolder Support**: Processes nested folder structures
- **Comprehensive Statistics**: Per-image, subfolder, and total counts
- **Raw File Support**: Processes both JPG and DNG camera files

### Execution Options

#### Option 1: Executable File (Recommended)
1. Download the [VarroDetector executable file for windows](https://unirioja-my.sharepoint.com/:u:/g/personal/jodivaso_unirioja_es/Eb0Jq31RbwpAjirbJkOjRVQBsu2onCeP1FL0neXk8dRHYw?e=EYUUie) (more platforms coming soon).
2. Run the executable file

*Note:* This file is self-contained, so the application takes a few seconds to start since the contents must be unzipped on the fly.

#### Option 2: Run from Source
1. Clone this repository
```bash
git clone https://github.com/jodivaso/VarroDetector.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python varroa_mite_gui.py
```

### User's manual

The program is very easy to use; however, you can [click here](https://unirioja-my.sharepoint.com/:b:/g/personal/jodivaso_unirioja_es/EcD0rAZJ49pHrSW40yprr2sBtFBxz5tAsLZVexBZqLI4cA?e=tb5JJv) 
to download the user's manual in PDF format.

### Controls

- **Zoom**: Mouse wheel
- **Pan**: Middle mouse button
- **Add mite manually**: Left click and drag
- **Delete mite**: Right click on detection box
- **Hide detections**: Press and hold 'h' key
- **View different images**: Click on image names in the list

### Working with Regions of Interest (ROI)

1. Select an image
2. Click "Edit ROI"
3. Left click to add points around your area of interest
4. Double click to complete the polygon (will be drawn in yellow)
5. Statistics will update to count only mites within the ROI
6. Right click to delete the current ROI

### Saving Results

When you click "Save Results", a new folder named "results" will be created in your input folder containing:

- **images/**: All processed images with visible detection boxes
- **labels/**: YOLO format text files with detection coordinates
- **statistics.csv**: Detailed counts for each image
- **statistics_subfolders.csv**: Summary statistics by subfolder

## Output Format

### statistics.csv
- **filename**: Relative path to the image
- **threshold**: Confidence threshold used for detection
- **varroa_count**: Number of mites detected in this image
- **subfolder_count**: Total mites in this image's subfolder
- **total_count**: Total mites across all images

### statistics_subfolders.csv
- **folder_name**: Subfolder path
- **num_varroa_mites_folder**: Total mites detected in this subfolder
- **threshold**: Confidence threshold(s) used
- **num_images**: Number of images in this subfolder
- **name_images**: List of image filenames


## Acknowledgments

This research has been funded by: 
- Grant INICIA2023/02 by La Rioja Government (Spain)
- MCIU/AEI/10.13039/501100011033 (grant PID2023-148475OB-I00)
- The EU Horizon Europe (grant 101082073)
- The DGA-FSE (grant A07_23R)

## License
This software uses a YOLOv11 nano model; thus, it is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

<img src="ur_logo.png" alt="University of La Rioja" width="200"/>&nbsp;&nbsp;&nbsp;&nbsp;<img src="unizar_logo.png" alt="University of Zaragoza" width="200"/>&nbsp;&nbsp;&nbsp;&nbsp;<img src="beeguards_logo.png" alt="BeeGuards" width="120" />
