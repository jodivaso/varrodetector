import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import os
import threading
import shutil
from ultralytics import YOLO
import math
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import euclidean
import rawpy


# Set appearance mode and color theme
ctk.set_appearance_mode("light")  # Changed to light mode
ctk.set_default_color_theme("blue")

# Custom color scheme
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#4b5563',    # Gray
    'accent': '#3b82f6',       # Light blue
    'success': '#22c55e',      # Green
    'warning': '#f59e0b',      # Yellow
    'error': '#ef4444',        # Red
    'background': '#ffffff',   # White
    'surface': '#f3f4f6',     # Light gray
    'text': '#1f2937',        # Dark gray
}


def adjust_dynamic_brightness(rgb_image, target_brightness=150):
    """
    Adjusts the brightness of an image dynamically to meet a target brightness level.

    :param rgb_image: The processed RGB image as a NumPy array.
    :param target_brightness: The target brightness level (0-255).
    :return: The brightness-adjusted image.
    """
    # Calculate the mean brightness of the image
    current_brightness = np.mean(rgb_image)
    brightness_factor = target_brightness / current_brightness if current_brightness > 0 else 1

    # Scale the brightness and clip values
    adjusted_image = np.clip(rgb_image * brightness_factor, 0, 255).astype(np.uint8)
    return adjusted_image


def process_dng(file_path):
    """Process a DNG file and return the RGB image."""
    try:
        with rawpy.imread(file_path) as raw:
            rgb_image = raw.postprocess(
                gamma=(2.0, 4.5),
                no_auto_bright=False,
                output_bps=16,
                use_camera_wb=True,
                user_sat=0.9,
                highlight_mode=1
            )
        rgb_image = (rgb_image / 256).astype(np.uint8)
        rgb_image = adjust_dynamic_brightness(rgb_image, target_brightness=150)
        # Convert from RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image
    except Exception as e:
        print(f"Error processing DNG file: {str(e)}")
        return None

def load_image_safely(image_path):
    """Load image using cv2 and convert to PIL Image to avoid truncation errors"""
    # Read image using cv2
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    return Image.fromarray(img_rgb)

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


def get_resource_path(relative_path):
    """Obtiene la ruta absoluta de un recurso, sea en desarrollo o en el ejecutable."""
    if hasattr(sys, '_MEIPASS'):
        # En el ejecutable, los recursos están en sys._MEIPASS
        base_path = sys._MEIPASS
    else:
        # En desarrollo, los recursos están en el directorio actual
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang) - 180 if abs(ang) > 180 else abs(ang)  # ang + 360 if ang < 0 else ang


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        print('lines do not intersect')
        return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def slope(x1, y1, x2, y2):
    ###finding slope
    if x2 != x1:
        return ((y2 - y1) / (x2 - x1))
    else:
        return 'NA'


def drawLine(image, x1, y1, x2, y2, color=(0, 255, 0)):
    m = slope(x1, y1, x2, y2)
    h, w = image.shape[:2]
    if m != 'NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px = 0
        py = -(x1 - 0) * m + y1
        ##ending point
        qx = w
        qy = -(x2 - w) * m + y2
    else:
        ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px, py = x1, 0
        qx, qy = x1, h
    cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, 3)


def crop_green_lines_from_array(img):
    """Process green lines detection from a numpy array image."""
    height, width, channels = img.shape
    img_small = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
    greenChannel = img_small[:, :, 1]
    b, g, r = cv2.split(img_small)

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    ## mask of green
    mask1 = cv2.inRange(hsv, (30, 25, 40), (140, 255, 255))

    kernel = np.ones((2, 2), np.uint8)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

    dst = cv2.Canny(mask1, 50, 200, None, 3)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, minLineLength=min(height, width) * 0.1 * 0.7, maxLineGap=200)

    if linesP is None or len(linesP) < 2:
        print("No green strings detected in the image")
        return None
    else:
        distance_list = []

        for i in range(0, len(linesP)):
            l = linesP[i][0]
            start_point = (l[0], l[1])
            end_point = (l[2], l[3])
            line_length = euclidean(start_point, end_point)
            distance_list.append(line_length)

        np_distance_list = np.array(distance_list)
        ids = (-np_distance_list).argsort()
        linesP_sorted_by_distance = [linesP[i][0] for i in ids]

        x1 = linesP_sorted_by_distance[0][0]
        y1 = linesP_sorted_by_distance[0][1]
        x2 = linesP_sorted_by_distance[0][2]
        y2 = linesP_sorted_by_distance[0][3]

        for i in range(1, len(linesP_sorted_by_distance)):
            l = linesP_sorted_by_distance[i]
            start_point = (l[0], l[1])
            end_point = (l[2], l[3])
            x_inter, y_inter = findIntersection(x1, y1, x2, y2, l[0], l[1], l[2], l[3])
            if x_inter is not None:
                angle = getAngle(start_point, (x_inter, y_inter), (x1, y1))
                if angle > 80 and angle < 95:
                    mask = np.zeros(img_small.shape[:2], dtype="uint8")
                    drawLine(mask, l[0], l[1], l[2], l[3], 255)
                    drawLine(mask, x1, y1, x2, y2, 255)
                    print("Detected green strings in the image.", "Detected angle:", angle)
                    mask = cv2.bitwise_not(mask)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    final_mask = np.zeros(img_small.shape, dtype="uint8")
                    cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255),
                                     thickness=cv2.FILLED)
                    final_mask = cv2.resize(final_mask, (width, height))

                    result = cv2.bitwise_and(img, final_mask)
                    return result

    print("No green strings detected in the image.")
    return None


def crop_green_lines(initial_img_path):
    initial_img = cv2.imread(initial_img_path)

    height, width, channels = initial_img.shape
    img = cv2.resize(initial_img, (0, 0), fx=0.1, fy=0.1)
    greenChannel = img[:, :, 1]
    b, g, r = cv2.split(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green
    mask1 = cv2.inRange(hsv, (30, 25, 40), (140, 255, 255))

    kernel = np.ones((2, 2), np.uint8)

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    # mask1 = cv2.dilate(mask1,kernel,iterations = 6)

    dst = cv2.Canny(mask1, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    # cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 10, minLineLength=min(height, width) * 0.1 * 0.7, maxLineGap=200)

    if linesP is None or len(linesP) < 2:
        print("No green strings detected in file", os.path.basename(initial_img_path))
        return None
    else:
        distance_list = []

        for i in range(0, len(linesP)):
            l = linesP[i][0]
            start_point = (l[0], l[1])
            end_point = (l[2], l[3])
            line_length = euclidean(start_point, end_point)
            distance_list.append(line_length)

        np_distance_list = np.array(distance_list)
        ids = (-np_distance_list).argsort()  # Para que sea en orden descedendiente
        linesP_sorted_by_distance = [linesP[i][0] for i in ids]

        x1 = linesP_sorted_by_distance[0][0]
        y1 = linesP_sorted_by_distance[0][1]
        x2 = linesP_sorted_by_distance[0][2]
        y2 = linesP_sorted_by_distance[0][3]

        for i in range(1, len(linesP_sorted_by_distance)):
            l = linesP_sorted_by_distance[i]
            start_point = (l[0], l[1])
            end_point = (l[2], l[3])
            x_inter, y_inter = findIntersection(x1, y1, x2, y2, l[0], l[1], l[2], l[3])
            if x_inter is not None:
                angle = getAngle(start_point, (x_inter, y_inter), (x1, y1))
                if angle > 80 and angle < 95:
                    mask = np.zeros(img.shape[:2], dtype="uint8")
                    drawLine(mask, l[0], l[1], l[2], l[3], 255)
                    drawLine(mask, x1, y1, x2, y2, 255)
                    print("Detected green strings in file", os.path.basename(initial_img_path) + ".", "Detected angle:", angle)
                    mask = cv2.bitwise_not(mask)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    # Rellenamos el contorno más grande.

                    final_mask = np.zeros(img.shape, dtype="uint8")
                    cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255),
                                     thickness=cv2.FILLED)
                    final_mask = cv2.resize(final_mask, (width, height))

                    result = cv2.bitwise_and(initial_img, final_mask)
                    return result

    print("No green strings detected in file", os.path.basename(initial_img_path) + ".")
    return None  # Si llego aquí es que no había ningún ángulo de 90.


class ModernTiledImageViewer(ctk.CTkFrame):
    def __init__(self, parent, GUI):
        super().__init__(parent)
        self.parent= parent
        self.GUI = GUI # We are passing the whole GUI and this is somehow weird, but we need it to update the statistics.

        # Storage for all boxes with their confidence scores
        self.all_boxes = []  # Store all boxes (detections and user-drawn) with confidence scores
        self.confidence_threshold = 0.1

        # Create a canvas with modern styling
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"][1],
            cursor="arrow"
        )

        # Modern scrollbars
        self.v_scroll = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.h_scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=self.canvas.xview)

        # Grid layout
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )

        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Add after other initializations
        self.init_roi_variables()

        # Initialize variables
        self.original_image = None
        self.image_path = None
        self.tile_cache = {}
        self.tile_size = 1024
        self.scale = 1.0
        self.current_box = None
        self.drag_start = None
        self.initial_scale = None

        # Box interaction variables
        self.active_box = None
        self.resize_handle = None
        self.box_drag_mode = None
        self.hover_box = None
        self.corner_radius = 5
        self.boxes_hidden = False
        self.hide_key_pressed = False
        self.edge_sensitivity = 5
        self.is_drawing_new = False

        # Bind keyboard events to root window
        self.root = self.winfo_toplevel()
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        # Create loading label
        self.loading_label = ctk.CTkLabel(self, text="Loading image...")

        # Create thread pool
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Bind events
        self._bind_events()

    def init_roi_variables(self):
        self.roi_points = []  # Store points for current ROI being drawn
        self.roi_polygons = {}  # Store completed ROIs for each image {image_name: [points]}
        self.drawing_roi = False  # Flag to indicate if we're currently drawing ROI
        self.current_roi_line = None  # Store temporary line while drawing
        self.hover_roi = False  # Track if mouse is over ROI

    def start_roi_drawing(self):
        """Start ROI drawing mode"""
        if not self.GUI.current_image:
            return

        self.drawing_roi = True
        self.roi_points = []
        self.canvas.config(cursor="cross")

        # Bind ROI-specific events
        self.canvas.bind("<Button-1>", self.add_roi_point)
        self.canvas.bind("<Motion>", self.update_roi_preview)
        self.canvas.bind("<Button-3>", self.delete_roi)
        self.canvas.bind("<Double-Button-1>", self.complete_roi)

    def stop_roi_drawing(self):
        """Stop ROI drawing mode"""
        self.drawing_roi = False
        self.canvas.config(cursor="arrow")
        self.roi_points = []
        if self.current_roi_line:
            self.canvas.delete(self.current_roi_line)
            self.current_roi_line = None

        # Restore original bindings
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.delete_box)

    def add_roi_point(self, event):
        """Add a point to the current ROI"""
        if not self.drawing_roi:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Add point
        self.roi_points.append((canvas_x, canvas_y))

        # Draw point
        point_radius = 3
        self.canvas.create_oval(
            canvas_x - point_radius, canvas_y - point_radius,
            canvas_x + point_radius, canvas_y + point_radius,
            fill="yellow", tags="roi_point"
        )

        # Draw line between points
        if len(self.roi_points) > 1:
            prev_x, prev_y = self.roi_points[-2]
            self.canvas.create_line(
                prev_x, prev_y, canvas_x, canvas_y,
                fill="yellow", width=2, tags="roi_line"
            )

    def update_roi_preview(self, event):
        """Update preview line while drawing ROI"""
        if not self.drawing_roi or not self.roi_points:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Delete previous preview line
        if self.current_roi_line:
            self.canvas.delete(self.current_roi_line)

        # Draw new preview line from last point to current mouse position
        last_x, last_y = self.roi_points[-1]
        self.current_roi_line = self.canvas.create_line(
            last_x, last_y, canvas_x, canvas_y,
            fill="yellow", width=2, dash=(4, 4), tags="roi_preview"
        )

    def complete_roi(self, event):
        """Complete ROI polygon on double-click"""
        if not self.drawing_roi or len(self.roi_points) < 3:
            return

        # Close the polygon
        first_x, first_y = self.roi_points[0]
        last_x, last_y = self.roi_points[-1]
        self.canvas.create_line(
            last_x, last_y, first_x, first_y,
            fill="yellow", width=2, tags="roi_line"
        )

        # Store ROI for current image
        if self.GUI.current_image:
            # Convert to image coordinates
            image_points = [(x / self.scale, y / self.scale) for x, y in self.roi_points]
            self.roi_polygons[self.GUI.current_image] = image_points

        # Clean up
        self.stop_roi_drawing()
        self.draw_roi()  # Redraw with hover effects

        # Reset button text and color
        self.GUI.roi_button.configure(
            text="Edit ROI",
            fg_color=COLORS['secondary']
        )

        # Update statistics
        self.GUI.update_box_statistics()

    def draw_roi(self):
        """Draw stored ROI for current image"""
        self.canvas.delete("roi_line", "roi_point", "roi_preview")

        if not self.GUI.current_image or not self.drawing_roi and \
                self.GUI.current_image not in self.roi_polygons:
            return

        points = self.roi_polygons.get(self.GUI.current_image, [])
        if not points:
            return

        # Scale points to current zoom level
        scaled_points = [(x * self.scale, y * self.scale) for x, y in points]

        # Draw polygon
        fill_color = "yellow" if self.hover_roi else ""

        # Draw filled polygon with stipple for semi-transparency
        self.canvas.create_polygon(
            *[coord for point in scaled_points for coord in point],
            outline="yellow",
            fill=fill_color,
            stipple="gray50",  # Semi-transparent fill
            width=2,
            tags="roi_line"
        )

        # Draw points at vertices for better visibility
        for x, y in scaled_points:
            self.canvas.create_oval(
                x - 3, y - 3,
                x + 3, y + 3,
                fill="yellow",
                outline="white",
                tags="roi_line"
            )

    def delete_roi(self, event=None):
        """Delete ROI for current image"""
        if self.GUI.current_image in self.roi_polygons:
            del self.roi_polygons[self.GUI.current_image]
            self.canvas.delete("roi_line", "roi_point", "roi_preview")
            self.GUI.update_box_statistics()

    def point_in_polygon(self, x, y, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_boxes_in_roi(self, threshold=None):
        """Get boxes that are within the current ROI using specified threshold"""
        if not self.GUI.current_image or self.GUI.current_image not in self.roi_polygons:
            # If no threshold provided, use current viewer threshold
            if threshold is None:
                return [box for box in self.all_boxes if box[4] >= self.confidence_threshold]
            return [box for box in self.all_boxes if box[4] >= threshold]

        roi_points = self.roi_polygons[self.GUI.current_image]
        if not roi_points:
            if threshold is None:
                return [box for box in self.all_boxes if box[4] >= self.confidence_threshold]
            return [box for box in self.all_boxes if box[4] >= threshold]

        # Filter boxes by threshold first
        threshold_to_use = threshold if threshold is not None else self.confidence_threshold
        threshold_boxes = [box for box in self.all_boxes if box[4] >= threshold_to_use]

        roi_boxes = []
        for box in threshold_boxes:
            # Get box center point
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Check if center is in ROI
            if self.point_in_polygon(center_x, center_y, roi_points):
                roi_boxes.append(box)

        return roi_boxes

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", self.delete_box)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)


    # Add these new methods to ModernTiledImageViewer:
    def set_confidence_threshold(self, value):
        """Update confidence threshold and redraw boxes without reloading image"""
        self.confidence_threshold = value
        self.draw_all_boxes()

    def get_visible_boxes(self):
        """Return only boxes that meet the confidence threshold"""
        return [box for box in self.all_boxes if box[4] >= self.confidence_threshold]

    def get_box_at_position(self, x, y):
        """Return box and hit area ('edge', 'corner', 'inside', or None) at given position"""
        for box in self.get_visible_boxes():  # Only check visible boxes
            scaled_box = [coord * self.scale for coord in box[:4]]  # Only scale coordinates

            # Check corners first
            corners = [
                (scaled_box[0], scaled_box[1]),
                (scaled_box[2], scaled_box[1]),
                (scaled_box[2], scaled_box[3]),
                (scaled_box[0], scaled_box[3])
            ]

            for corner in corners:
                if abs(x - corner[0]) <= self.corner_radius and abs(y - corner[1]) <= self.corner_radius:
                    return box, 'corner', corners.index(corner)

            # Check edges
            if abs(x - scaled_box[0]) <= self.edge_sensitivity and scaled_box[1] <= y <= scaled_box[3]:
                return box, 'edge', 'left'
            if abs(x - scaled_box[2]) <= self.edge_sensitivity and scaled_box[1] <= y <= scaled_box[3]:
                return box, 'edge', 'right'
            if abs(y - scaled_box[1]) <= self.edge_sensitivity and scaled_box[0] <= x <= scaled_box[2]:
                return box, 'edge', 'top'
            if abs(y - scaled_box[3]) <= self.edge_sensitivity and scaled_box[0] <= x <= scaled_box[2]:
                return box, 'edge', 'bottom'

            # Check inside
            if (scaled_box[0] <= x <= scaled_box[2] and
                    scaled_box[1] <= y <= scaled_box[3]):
                return box, 'inside', None

        return None, None, None

    def update_cursor(self, hit_area, edge_type=None):
        """Update cursor based on hit area"""
        if hit_area == 'inside':
            self.canvas.configure(cursor="fleur")
        elif hit_area == 'corner':
            self.canvas.configure(cursor="sizing")
        elif hit_area == 'edge':
            if edge_type in ['left', 'right']:
                self.canvas.configure(cursor="sb_h_double_arrow")
            else:
                self.canvas.configure(cursor="sb_v_double_arrow")
        else:
            self.canvas.configure(cursor="arrow")

    def on_mouse_move(self, event):
        """Handle mouse movement for hover effects and cursor updates"""
        if self.is_drawing_new:  # If drawing a new box, don't change cursor
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        box, hit_area, edge_type = self.get_box_at_position(canvas_x, canvas_y)

        # Update cursor
        self.update_cursor(hit_area, edge_type)

        # Update hover effect
        if box != self.hover_box:
            self.hover_box = box
            self.draw_all_boxes()

    def on_key_press(self, event):
        """Handle key press events"""
        if event.char == 'h' and not self.hide_key_pressed:
            self.hide_key_pressed = True
            self.boxes_hidden = True
            self.draw_all_boxes()

    def on_key_release(self, event):
        """Handle key release events"""
        if event.char == 'h':
            self.hide_key_pressed = False
            self.boxes_hidden = False
            self.draw_all_boxes()

    def on_button_press(self, event):
        """Handle mouse button press for all interactions"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Check if clicking on existing box
        box, hit_area, edge_type = self.get_box_at_position(canvas_x, canvas_y)

        if box:
            # Interacting with existing box
            self.active_box = box
            self.box_drag_mode = hit_area
            self.drag_start = (canvas_x, canvas_y)
            if hit_area == 'corner':
                self.resize_handle = edge_type
            elif hit_area == 'edge':
                self.resize_handle = edge_type
        else:
            # Starting to draw new box
            self.is_drawing_new = True
            self.drag_start = (canvas_x, canvas_y)
            self.current_box = None

    def on_drag(self, event):
        """Handle all dragging operations"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if self.is_drawing_new:
            # Drawing new box
            if self.current_box:
                self.canvas.delete(self.current_box)
            self.current_box = self.canvas.create_rectangle(
                self.drag_start[0], self.drag_start[1],
                canvas_x, canvas_y,
                outline="red", width=2
            )
        elif self.active_box and self.drag_start:
            # Modifying existing box
            dx = (canvas_x - self.drag_start[0]) / self.scale
            dy = (canvas_y - self.drag_start[1]) / self.scale

            if self.box_drag_mode == 'inside':
                # Move entire box
                self.active_box[0] += dx
                self.active_box[1] += dy
                self.active_box[2] += dx
                self.active_box[3] += dy

            elif self.box_drag_mode == 'corner':
                # Resize from corner
                if self.resize_handle == 0:  # Top-left
                    self.active_box[0] += dx
                    self.active_box[1] += dy
                elif self.resize_handle == 1:  # Top-right
                    self.active_box[2] += dx
                    self.active_box[1] += dy
                elif self.resize_handle == 2:  # Bottom-right
                    self.active_box[2] += dx
                    self.active_box[3] += dy
                elif self.resize_handle == 3:  # Bottom-left
                    self.active_box[0] += dx
                    self.active_box[3] += dy

            elif self.box_drag_mode == 'edge':
                # Resize from edge
                if self.resize_handle == 'left':
                    self.active_box[0] += dx
                elif self.resize_handle == 'right':
                    self.active_box[2] += dx
                elif self.resize_handle == 'top':
                    self.active_box[1] += dy
                elif self.resize_handle == 'bottom':
                    self.active_box[3] += dy

            self.drag_start = (canvas_x, canvas_y)
            self.draw_all_boxes()

    def on_button_release(self, event):
        """Handle mouse button release for all interactions"""
        if self.is_drawing_new and self.drag_start:
            # Finish drawing new box
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Convert to original image coordinates
            x1 = min(self.drag_start[0], canvas_x) / self.scale
            y1 = min(self.drag_start[1], canvas_y) / self.scale
            x2 = max(self.drag_start[0], canvas_x) / self.scale
            y2 = max(self.drag_start[1], canvas_y) / self.scale

            # Add the new box to all_boxes with a default confidence of 1.0 for user-drawn boxes
            self.all_boxes.append([int(x1), int(y1), int(x2), int(y2), 1.0])

            if self.current_box:
                self.canvas.delete(self.current_box)

            # Update statistics in GUI
            if hasattr(self.GUI, 'update_box_statistics'):
                self.GUI.update_box_statistics()

        # Reset all interaction states
        self.active_box = None
        self.box_drag_mode = None
        self.drag_start = None
        self.resize_handle = None
        self.current_box = None
        self.is_drawing_new = False

        # Redraw all boxes
        self.draw_all_boxes()

    def delete_box(self, event):
        """Delete box on right click"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        box, hit_area, _ = self.get_box_at_position(canvas_x, canvas_y)
        if box:
            self.all_boxes.remove(box)
            self.draw_all_boxes()
            # Update statistics in GUI
            if hasattr(self.GUI, 'update_box_statistics'):
                self.GUI.update_box_statistics()

    def draw_all_boxes(self):
        """Draw all bounding boxes with hover effects and corner handles"""
        self.canvas.delete("box")

        # Don't draw boxes if they're hidden
        if self.boxes_hidden:
            return

        visible_boxes = self.get_visible_boxes()

        for box in visible_boxes:
            scaled_box = [coord * self.scale for coord in box[:4]]

            # Draw the box with thicker outline if hovered
            line_width = 4 if box == self.hover_box else 2
            self.canvas.create_rectangle(
                scaled_box[0], scaled_box[1],
                scaled_box[2], scaled_box[3],
                outline="red", width=line_width, tags="box"
            )

            # Draw red corner dots only if box is hovered
            if box == self.hover_box:
                dot_radius = 3
                corners = [
                    (scaled_box[0], scaled_box[1]),
                    (scaled_box[2], scaled_box[1]),
                    (scaled_box[2], scaled_box[3]),
                    (scaled_box[0], scaled_box[3])
                ]

                for cx, cy in corners:
                    self.canvas.create_oval(
                        cx - dot_radius, cy - dot_radius,
                        cx + dot_radius, cy + dot_radius,
                        fill="red", outline=None, tags="box"
                    )

    def load_image(self, image_path, boxes=None):
        """Load image and store all detection boxes"""
        self.loading_label.place(relx=0.5, rely=0.5, anchor='center')
        self.loading_label.lift()
        self.update()

        try:
            # Clear existing cache and canvas
            self.tile_cache.clear()
            self.canvas.delete("all")

            # Store image path and load image
            self.image_path = image_path
            self.original_image = load_image_safely(image_path)

            # Store all boxes with confidence scores
            self.all_boxes = boxes if boxes else []

            # Calculate initial scale
            self.scale = self.calculate_initial_scale()
            self.initial_scale = self.scale

            # Update canvas size
            scaled_width = int(self.original_image.width * self.scale)
            scaled_height = int(self.original_image.height * self.scale)
            self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))

            # Pre-load tiles and draw
            self.preload_tiles()

            # Draw boxes that meet the current threshold
            self.draw_all_boxes()

        except Exception as e:
            self.loading_label.place_forget()
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def calculate_initial_scale(self):
        """Calculate initial scale to fit image in viewport"""
        if not self.original_image:
            return 1.0

        # Get viewport size
        viewport_width = self.canvas.winfo_width()
        viewport_height = self.canvas.winfo_height()

        if viewport_width == 1:  # Widget not yet realized
            viewport_width = 800  # Default width
            viewport_height = 600  # Default height

        # Calculate scale to fit
        width_scale = viewport_width / self.original_image.width
        height_scale = viewport_height / self.original_image.height

        # Use the smaller scale to ensure entire image is visible
        return min(width_scale, height_scale) * 0.95  # 95% to leave some margin


    def preload_tiles(self):
        """Preload all tiles in background"""
        if not self.original_image:
            return

        def load_tile_job(row, col):
            tile = self.load_tile(row, col)
            if tile:
                return (row, col, tile)
            return None

        # Calculate number of tiles needed
        num_rows = math.ceil(self.original_image.height / self.tile_size)
        num_cols = math.ceil(self.original_image.width / self.tile_size)

        # Submit all tile loading jobs
        futures = []
        for row in range(num_rows):
            for col in range(num_cols):
                future = self.executor.submit(load_tile_job, row, col)
                futures.append(future)

        # Process completed tiles
        for future in futures:
            result = future.result()
            if result:
                row, col, tile = result
                tile_key = self.get_tile_key(row, col, self.scale)
                self.tile_cache[tile_key] = ImageTk.PhotoImage(tile)
        # Hide loading label and draw tiles
        self.loading_label.place_forget()
        self.draw_visible_tiles()
        self.draw_all_boxes()
        # Center the image
        self.center_image()

    def center_image(self):
        """Center the image in the viewport"""
        self.canvas.update_idletasks()

        # Get viewport and image dimensions
        viewport_width = self.canvas.winfo_width()
        viewport_height = self.canvas.winfo_height()
        scaled_width = int(self.original_image.width * self.scale)
        scaled_height = int(self.original_image.height * self.scale)

        # Calculate center position
        x_center = max(0, (scaled_width - viewport_width) / 2)
        y_center = max(0, (scaled_height - viewport_height) / 2)

        # Set scroll position
        self.canvas.xview_moveto(x_center / scaled_width if scaled_width > viewport_width else 0)
        self.canvas.yview_moveto(y_center / scaled_height if scaled_height > viewport_height else 0)


    def load_tile(self, row, col):
        """Load a specific tile of the image with a small overlap"""
        if not self.original_image:
            return None

        # Add 2-pixel overlap to prevent seams while keeping performance
        overlap = 2

        # Calculate tile coordinates in original image with overlap
        x1 = max(0, col * self.tile_size - overlap)
        y1 = max(0, row * self.tile_size - overlap)
        x2 = min(x1 + self.tile_size + 2 * overlap, self.original_image.width)
        y2 = min(y1 + self.tile_size + 2 * overlap, self.original_image.height)

        # Ensure dimensions are even to prevent pixel rounding issues
        width = x2 - x1
        height = y2 - y1
        if width % 2 == 1:
            x2 = min(x2 + 1, self.original_image.width)
        if height % 2 == 1:
            y2 = min(y2 + 1, self.original_image.height)

        # Crop tile from original image
        tile = self.original_image.crop((x1, y1, x2, y2))

        # Scale tile
        if self.scale != 1.0:
            new_size = (
                int((x2 - x1) * self.scale + 0.5),  # Add 0.5 for proper rounding
                int((y2 - y1) * self.scale + 0.5)
            )
            tile = tile.resize(new_size, Image.Resampling.NEAREST)  # Using NEAREST for speed

        return tile

    def get_tile_key(self, row, col, scale):
        """Generate a unique key for a tile"""
        return f"{row}_{col}_{scale}"

    def draw_visible_tiles(self):
        """Draw only the tiles that are currently visible"""
        if not self.original_image:
            return

        # Calculate tile ranges
        num_rows = math.ceil(self.original_image.height / self.tile_size)
        num_cols = math.ceil(self.original_image.width / self.tile_size)

        # Clear existing tiles
        self.canvas.delete("tile")

        # Draw all tiles
        for row in range(num_rows):
            for col in range(num_cols):
                tile_key = self.get_tile_key(row, col, self.scale)

                if tile_key in self.tile_cache:
                    x = col * self.tile_size * self.scale
                    y = row * self.tile_size * self.scale
                    self.canvas.create_image(
                        x, y,
                        image=self.tile_cache[tile_key],
                        anchor="nw",
                        tags="tile"
                    )


    def zoom(self, event):
        """Handle zoom with mouse wheel"""
        if not self.original_image or hasattr(self, '_zooming'):
            return

        # Store ROI state before zooming
        had_roi = self.GUI.current_image in self.roi_polygons

        # Get mouse position relative to canvas and image
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Calculate relative position within the image
        rel_x = canvas_x / (self.original_image.width * self.scale)
        rel_y = canvas_y / (self.original_image.height * self.scale)

        # Calculate new scale
        old_scale = self.scale
        target_scale = self.scale * (1.2 if event.delta > 0 else 1 / 1.2)

        # Limit minimum and maximum zoom
        min_scale = self.initial_scale * 0.5
        max_scale = self.initial_scale * 7.0
        target_scale = max(min_scale, min(max_scale, target_scale))

        # Only proceed if scale will actually change
        if target_scale != old_scale:
            self._zooming = True

            # Store current view state
            current_view = {
                'scale': self.scale,
                'tiles': self.tile_cache.copy(),
                'x': rel_x,
                'y': rel_y
            }

            def prepare_new_view():
                # Prepare new tiles without displaying them
                new_cache = {}
                self.scale = target_scale

                # Calculate dimensions for new scale
                scaled_width = int(self.original_image.width * self.scale)
                scaled_height = int(self.original_image.height * self.scale)

                # Load all new tiles
                num_rows = math.ceil(self.original_image.height / self.tile_size)
                num_cols = math.ceil(self.original_image.width / self.tile_size)

                for row in range(num_rows):
                    for col in range(num_cols):
                        tile = self.load_tile(row, col)
                        if tile:
                            tile_key = self.get_tile_key(row, col, self.scale)
                            new_cache[tile_key] = ImageTk.PhotoImage(tile)

                def swap_views():
                    # Configure new scroll region
                    self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))

                    # Update tiles
                    self.tile_cache = new_cache
                    current_view['tiles'].clear()

                    # Calculate and set new view position
                    view_width = self.canvas.winfo_width()
                    view_height = self.canvas.winfo_height()
                    new_x = rel_x * scaled_width
                    new_y = rel_y * scaled_height
                    frac_x = max(0, min(1, (new_x - view_width / 2) / scaled_width))
                    frac_y = max(0, min(1, (new_y - view_height / 2) / scaled_height))

                    # Update view
                    self.canvas.delete("tile")
                    self.draw_visible_tiles()
                    self.draw_all_boxes()

                    # Redraw ROI if it existed before zooming
                    if had_roi:
                        self.draw_roi()

                    self.canvas.xview_moveto(frac_x)
                    self.canvas.yview_moveto(frac_y)

                    # Clean up
                    del self._zooming

                # Swap views once everything is ready
                self.after(1, swap_views)

            # Start preparing the new view
            self.after(1, prepare_new_view)

    def after_zoom(self, delta_x, delta_y):
        """Called after zoom to update display"""
        self.canvas.xview_scroll(int(delta_x), "units")
        self.canvas.yview_scroll(int(delta_y), "units")
        self.draw_visible_tiles()
        self.draw_all_boxes()
        self.zoom_task = None

    def start_pan(self, event):
        """Start panning with middle mouse button"""
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        """Pan image with middle mouse button"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.draw_visible_tiles()
        self.draw_all_boxes()

        # Redraw ROI if it exists for current image
        if self.GUI.current_image in self.roi_polygons:
            self.draw_roi()

    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.draw_visible_tiles()

    def start_box(self, event):
        """Start drawing a bounding box"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.drag_start = (canvas_x, canvas_y)
        self.current_box = None

    def draw_box(self, event):
        """Draw bounding box while dragging"""
        if self.drag_start:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            if self.current_box:
                self.canvas.delete(self.current_box)

            self.current_box = self.canvas.create_rectangle(
                self.drag_start[0], self.drag_start[1],
                canvas_x, canvas_y,
                outline="red", width=2
            )

    def end_box(self, event):
        """Finish drawing bounding box"""
        if self.drag_start and self.current_box:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            # Convert to original image coordinates
            box = [
                int(self.drag_start[0] / self.scale),
                int(self.drag_start[1] / self.scale),
                int(canvas_x / self.scale),
                int(canvas_y / self.scale)
            ]
            self.boxes.append(box)
            self.drag_start = None
            self.current_box = None
            self.draw_all_boxes()


class ModernVarroaDetectorGUI:
    def __init__(self):
        # This is for closing the splash screen (https://stackoverflow.com/questions/48315785/)
        try:
            import pyi_splash
            pyi_splash.update_text('UI Loaded ...')
            pyi_splash.close()
        except:
            pass

        self.root = ctk.CTk()
        self.root.title("VarroDetector")
        # Set window icon
        icon_path = self.get_resource_path("icon.ico")
        try:
            self.root.iconbitmap(icon_path)
        except Exception as e:
            print(f"Could not load icon: {str(e)}")

        # Add dictionary to store image-specific confidence thresholds
        self.image_confidence_thresholds = {}
        self.current_image = None
        self.current_boxes = {}  # Store boxes for each image

        # Set default font
        self.default_font = ("Inter", 13)
        self.header_font = ("Inter", 16, "bold")

        # Initialize model
        self.model_path = self.get_resource_path("model/weights/best.pt")
        self.model = YOLO(self.model_path, verbose=False)
        self.current_folder = None
        self.output_path = None

        # Setup UI elements
        self.setup_ui()

        # Bind cleanup to window closing
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)

        # Maximize window
        self.root.update()
        self.root.state('zoomed')

    def setup_ui(self):
        # Create main container with padding and rounded corners
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=COLORS['surface'],
            corner_radius=15
        )
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Create sidebar with enhanced styling
        self.sidebar = ctk.CTkFrame(
            self.main_container,
            width=320,
            fg_color=COLORS['surface'],
            corner_radius=10
        )
        self.sidebar.pack(side="left", fill="y", padx=(0, 20), pady=10)
        # Prevent the sidebar from expanding
        self.sidebar.pack_propagate(False)

        # Create header frame to hold icon and text
        self.header_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        self.header_frame.pack(pady=(5, 10), padx=10)

        # Load and display the icon
        icon_path = self.get_resource_path("icon_for_sidebar.png")
        try:
            # Create CTkImage
            icon = ctk.CTkImage(
                light_image=Image.open(icon_path),
                dark_image=Image.open(icon_path),
                size=(64, 64)  # Adjust size as needed
            )

            # Create label for icon
            self.header_icon = ctk.CTkLabel(
                self.header_frame,
                image=icon,
                text=""  # No text, just the icon
            )
            self.header_icon.pack(side="left", padx=(0, 5))

        except Exception as e:
            print(f"Could not load icon: {e}")

        # Add text label
        self.header_text = ctk.CTkLabel(
            self.header_frame,
            text="VarroDetector",
            font=("Inter", 24, "bold"),
            text_color=COLORS['primary']
        )
        self.header_text.pack(side="left")

        # Select folder button with hover effect
        self.select_button = ctk.CTkButton(
            self.sidebar,
            text="Select Input Folder",
            command=self.run_task_in_thread,
            height=45,
            #font=self.default_font,
            font=("Inter", 13, "bold"),
            fg_color=COLORS['primary'],
            hover_color=COLORS['accent'],
            corner_radius=8
        )
        self.select_button.pack(pady=(0, 10), padx=20, fill="x")

        # Save button
        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="Save Results",
            command=self.save_results,
            height=45,
            font=("Inter", 13, "bold"),
            fg_color=COLORS['success'],  # Using success color for save button
            hover_color=COLORS['primary'],
            corner_radius=8
        )
        self.save_button.pack(pady=(0, 10), padx=20, fill="x")
        # Disable save button initially
        self.save_button.configure(state="disabled")

        # Help button
        self.help_button = ctk.CTkButton(
            self.sidebar,
            text="Help",
            command=self.show_help,
            height=45,
            font=("Inter", 13, "bold"),
            fg_color=COLORS['secondary'],  # Using secondary color for help button
            hover_color=COLORS['accent'],
            corner_radius=8
        )
        self.help_button.pack(pady=(0, 20), padx=20, fill="x")

        # Progress section (new addition)
        self.progress_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            mode="determinate",
            height=6,
            corner_radius=3
        )
        self.progress_bar.pack(fill="x", pady=(0, 5))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready",
            font=("Inter", 12),
            text_color=COLORS['secondary']
        )
        self.progress_label.pack(anchor="w")

        # Add confidence threshold slider
        self.confidence_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        self.confidence_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.confidence_label = ctk.CTkLabel(
            self.confidence_frame,
            text="Confidence Threshold: 0.10",
            font=self.default_font,
            text_color=COLORS['text']
        )
        self.confidence_label.pack(anchor="w", pady=(0, 5))

        self.confidence_slider = ctk.CTkSlider(
            self.confidence_frame,
            from_=0.1,
            to=1.0,
            number_of_steps=90,
            command=self.update_confidence_threshold,
            height=16
        )
        self.confidence_slider.set(0.1)
        self.confidence_slider.pack(fill="x")

        # Add Apply to All button
        self.apply_all_button = ctk.CTkButton(
            self.confidence_frame,
            text="Apply this threshold to all images",
            command=self.apply_threshold_to_all,
            height=30,
            font=("Inter", 12, "bold"),
            fg_color=COLORS['secondary'],
            hover_color=COLORS['accent'],
            corner_radius=8,
            state="disabled"  # Start disabled
        )
        self.apply_all_button.pack(pady=(10, 0), fill="x")

        self.roi_button = ctk.CTkButton(
            self.sidebar,
            text="Edit ROI",  # Initial text
            command=self.toggle_roi_mode,
            height=30,
            font=("Inter", 12, "bold"),  # Added bold
            fg_color=COLORS['secondary'],
            hover_color=COLORS['accent'],
            corner_radius=8
        )
        self.roi_button.pack(pady=(0, 10), padx=20, fill="x")
        # Disable ROI button initially
        self.roi_button.configure(state="disabled")


        # Add statistics frame
        self.setup_statistics_frame()

        # Image list section with improved styling
        self.list_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        self.list_frame.pack(fill="both", expand=True, padx=20)

        self.list_label = ctk.CTkLabel(
            self.list_frame,
            text="Processed Images",
            font=self.header_font,
            text_color=COLORS['text']
        )
        self.list_label.pack(pady=(0, 10), anchor="w")

        # Modern listbox with custom styling
        self.image_listbox = tk.Listbox(
            self.list_frame,
            bg=COLORS['background'],
            fg=COLORS['text'],
            selectmode="single",
            borderwidth=1,
            highlightthickness=1,
            font=("Inter", 10),
            selectbackground=COLORS['primary'],
            selectforeground=COLORS['background'],
            cursor="hand2",  # Add hand cursor
            selectborderwidth=0,  # Remove border when selected
            activestyle='none',  # Remove underline when active
            height=15  # Set a fixed height
        )
        self.image_listbox.pack(fill="both", expand=True)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_select_image)
        # Disable listbox initially
        self.image_listbox.configure(state="disabled")

        # Console output section with enhanced styling
        self.console_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        #self.console_frame.pack(fill="x", padx=20, pady=20)

        self.console_label = ctk.CTkLabel(
            self.console_frame,
            text="Console Output",
            font=self.header_font,
            text_color=COLORS['text']
        )
        #self.console_label.pack(anchor="w", pady=(0, 10))

        # Modern console output
        self.text_output = ctk.CTkTextbox(
            self.console_frame,
            height=200,
            font=("JetBrains Mono", 12),  # Monospace font for console
            fg_color=COLORS['background'],
            text_color=COLORS['text'],
            corner_radius=8
        )
        self.text_output.pack(fill="x")

        # Redirect stdout
        self.redirector = RedirectText(self.text_output)
        sys.stdout = self.redirector

        # Image viewer with improved styling
        self.image_viewer = ModernTiledImageViewer(
            self.main_container, self
        )
        self.image_viewer.pack(side="left", fill="both", expand=True)

        # Status bar (new addition)
        self.status_bar = ctk.CTkLabel(
            self.root,
            text="Ready to process images",
            font=("Inter", 12),
            text_color=COLORS['secondary']
        )
        self.status_bar.pack(side="bottom", fill="x", padx=20, pady=5)

    def get_all_images(self, folder):
        """Recursively get all JPG and DNG files from folder and subfolders"""
        image_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg', '.dng')):
                    # Get the full path and the relative path
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, folder)
                    image_files.append((full_path, rel_path))
        return image_files


    def apply_threshold_to_all(self):
        """Apply the current confidence threshold to all images"""
        try:
            # Get current threshold from slider
            current_threshold = self.confidence_slider.get()

            # Temporarily store current image and image viewer state
            temp_current = self.current_image
            temp_scale = self.image_viewer.scale if self.current_image else None

            # Update progress bar
            self.update_progress(0, "Applying threshold to all images...")
            total_images = len(self.current_boxes)

            # Update thresholds for all images
            for idx, image_name in enumerate(self.current_boxes.keys()):
                # Update progress
                progress = (idx + 1) / total_images
                self.update_progress(progress, f"Updating threshold for {image_name}")

                # Set the new threshold for this image
                self.image_confidence_thresholds[image_name] = current_threshold

                # If this is the current image, update the viewer
                if image_name == self.current_image and hasattr(self, 'image_viewer'):
                    self.image_viewer.set_confidence_threshold(current_threshold)

            # Restore original current image and scale
            self.current_image = temp_current
            if temp_scale:
                self.image_viewer.scale = temp_scale

            # Update statistics to reflect new thresholds
            self.update_box_statistics()

            # Reset progress bar and show completion message
            self.update_progress(1.0, "Threshold applied to all images")
            self.root.after(2000, lambda: self.update_progress(0, "Ready"))

        except Exception as e:
            print(f"Error applying threshold to all images: {str(e)}")
            messagebox.showerror("Error", f"Error applying threshold: {str(e)}")
            self.update_progress(0, "Ready")


    def toggle_roi_mode(self):
        if not hasattr(self.image_viewer, 'drawing_roi') or not self.image_viewer.drawing_roi:
            # Enter ROI editing mode
            self.roi_button.configure(
                fg_color=COLORS['primary'],  # Highlight button
                text="Editing ROI (click here to exit)"  # Update button text
            )
            self.image_viewer.start_roi_drawing()
        else:
            # Exit ROI editing mode
            self.roi_button.configure(
                fg_color=COLORS['secondary'],  # Reset button color
                text="Edit ROI"  # Restore original text
            )
            self.image_viewer.stop_roi_drawing()

    # Add this new method to the ModernVarroaDetectorGUI class:
    def cleanup(self, exit_program=True):
        """Clean up temporary files and folders before exit"""
        try:
            # Remove processed_images folder if it exists
            if hasattr(self, 'output_path') and self.output_path and os.path.exists(self.output_path):
                shutil.rmtree(self.output_path)
                print("Cleaned up temporary files")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            if exit_program:
                self.root.destroy()

    def setup_statistics_frame(self):
        self.stats_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        self.stats_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Detection Statistics",
            font=self.header_font,
            text_color=COLORS['text']
        )
        self.stats_label.pack(anchor="w", pady=(0, 5))

        self.current_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text="Current Image: 0 varroa mites",
            font=self.default_font,
            text_color=COLORS['text']
        )
        self.current_boxes_label.pack(anchor="w")

        self.subfolder_boxes_label = ctk.CTkLabel(  # New label for subfolder count
            self.stats_frame,
            text="Current Subfolder: 0 varroa mites",
            font=self.default_font,
            text_color=COLORS['text']
        )
        self.subfolder_boxes_label.pack(anchor="w")

        self.total_boxes_label = ctk.CTkLabel(
            self.stats_frame,
            text="Total (all images): 0 varroa mites",
            font=self.default_font,
            text_color=COLORS['text']
        )
        self.total_boxes_label.pack(anchor="w")

    # def update_box_statistics(self):
    #     """Update the box count statistics considering ROI if present"""
    #     # Count boxes in current image
    #     current_count = 0
    #     subfolder_count = 0
    #
    #     if self.current_image:
    #         current_threshold = self.image_confidence_thresholds.get(self.current_image, 0.1)
    #         boxes = self.image_viewer.get_boxes_in_roi(threshold=current_threshold)
    #         current_count = len(boxes)
    #
    #         # Get current subfolder
    #         current_folder = os.path.dirname(self.current_image)
    #
    #         # Find all images in the same subfolder
    #         subfolder_images = []
    #         for root, _, files in os.walk(self.output_path):
    #             for f in files:
    #                 if f.lower().endswith('.jpg'):
    #                     # Get relative path
    #                     rel_path = os.path.relpath(os.path.join(root, f), self.output_path)
    #                     if os.path.dirname(rel_path) == current_folder:
    #                         subfolder_images.append(rel_path)
    #
    #         # Count boxes in all images from the same subfolder
    #         for image_name in subfolder_images:
    #             # Load boxes if not already loaded
    #             if image_name not in self.current_boxes:
    #                 self.load_boxes_for_image(image_name)
    #
    #             if image_name in self.current_boxes:
    #                 threshold = self.image_confidence_thresholds.get(image_name, 0.1)
    #                 boxes = self.current_boxes[image_name]
    #
    #                 if image_name in self.image_viewer.roi_polygons:
    #                     # Temporarily set up image viewer state to check ROI
    #                     temp_current = self.current_image
    #                     temp_boxes = self.image_viewer.all_boxes
    #
    #                     self.current_image = image_name
    #                     self.image_viewer.all_boxes = boxes
    #
    #                     roi_boxes = self.image_viewer.get_boxes_in_roi(threshold=threshold)
    #                     subfolder_count += len(roi_boxes)
    #
    #                     # Restore original state
    #                     self.current_image = temp_current
    #                     self.image_viewer.all_boxes = temp_boxes
    #                 else:
    #                     subfolder_count += sum(1 for box in boxes if box[4] >= threshold)
    #
    #     # Count total boxes across all images
    #     total_count = 0
    #     for image_name, boxes in self.current_boxes.items():
    #         threshold = self.image_confidence_thresholds.get(image_name, 0.1)
    #
    #         if image_name in self.image_viewer.roi_polygons:
    #             temp_current = self.current_image
    #             temp_boxes = self.image_viewer.all_boxes
    #
    #             self.current_image = image_name
    #             self.image_viewer.all_boxes = boxes
    #
    #             roi_boxes = self.image_viewer.get_boxes_in_roi(threshold=threshold)
    #             total_count += len(roi_boxes)
    #
    #             self.current_image = temp_current
    #             self.image_viewer.all_boxes = temp_boxes
    #         else:
    #             total_count += sum(1 for box in boxes if box[4] >= threshold)
    #
    #     # Update labels
    #     roi_text = " (in ROI)" if self.current_image in self.image_viewer.roi_polygons else ""
    #     current_folder_text = f" ({os.path.dirname(self.current_image)})" if self.current_image and os.path.dirname(
    #         self.current_image) else ""
    #
    #     self.current_boxes_label.configure(text=f"Current Image{roi_text}: {current_count} varroa mites")
    #     self.subfolder_boxes_label.configure(
    #         text=f"Current Subfolder{current_folder_text}: {subfolder_count} varroa mites")
    #     self.total_boxes_label.configure(text=f"Total (in ROIs or full images): {total_count} varroa mites")
    #
    #     # Update display
    #     if self.current_image:
    #         self.image_viewer.draw_all_boxes()
    #         if self.current_image in self.image_viewer.roi_polygons:
    #             self.image_viewer.draw_roi()

    def update_box_statistics(self):
        """Update the box count statistics considering ROI if present"""
        # Initialize counts
        current_count = 0
        subfolder_count = 0
        total_count = 0

        # Check if output path exists
        if not hasattr(self, 'output_path') or not self.output_path or not os.path.exists(self.output_path):
            # Update labels with zero counts
            self.current_boxes_label.configure(text="Current Image: 0 varroa mites")
            self.subfolder_boxes_label.configure(text="Current Subfolder: 0 varroa mites")
            self.total_boxes_label.configure(text="Total (in ROIs or full images): 0 varroa mites")
            return

        # Process current image and subfolder if one is selected
        if self.current_image:
            # Current image count
            current_threshold = self.image_confidence_thresholds.get(self.current_image, 0.1)
            boxes = self.image_viewer.get_boxes_in_roi(threshold=current_threshold)
            current_count = len(boxes)

            # Get current subfolder
            current_folder = os.path.dirname(self.current_image)

            # Find all images in the same subfolder
            subfolder_images = []
            for root, _, files in os.walk(self.output_path):
                if "predict 0.1" in root:  # Skip predict directory
                    continue
                for f in files:
                    if f.lower().endswith('.jpg'):
                        try:
                            rel_path = os.path.relpath(os.path.join(root, f), self.output_path)
                            if os.path.dirname(rel_path) == current_folder:
                                subfolder_images.append(rel_path)
                        except Exception as e:
                            print(f"Error processing path: {str(e)}")

            # Count boxes in all images from the same subfolder
            for image_name in subfolder_images:
                try:
                    # Load boxes if not already loaded
                    if image_name not in self.current_boxes:
                        self.load_boxes_for_image(image_name)

                    if image_name in self.current_boxes:
                        threshold = self.image_confidence_thresholds.get(image_name, 0.1)
                        boxes = self.current_boxes[image_name]

                        if image_name in self.image_viewer.roi_polygons:
                            temp_current = self.current_image
                            temp_boxes = self.image_viewer.all_boxes

                            self.current_image = image_name
                            self.image_viewer.all_boxes = boxes

                            roi_boxes = self.image_viewer.get_boxes_in_roi(threshold=threshold)
                            subfolder_count += len(roi_boxes)

                            self.current_image = temp_current
                            self.image_viewer.all_boxes = temp_boxes
                        else:
                            subfolder_count += sum(1 for box in boxes if box[4] >= threshold)
                except Exception as e:
                    print(f"Error processing subfolder image {image_name}: {str(e)}")

        # Count total boxes across ALL images in all directories
        try:
            # Find all images in the output directory and its subdirectories
            all_images = []
            for root, _, files in os.walk(self.output_path):
                if "predict 0.1" in root:  # Skip predict directory
                    continue
                for f in files:
                    if f.lower().endswith('.jpg'):
                        try:
                            rel_path = os.path.relpath(os.path.join(root, f), self.output_path)
                            all_images.append(rel_path)
                        except Exception as e:
                            print(f"Error processing path: {str(e)}")

            # Process each image
            for image_name in all_images:
                try:
                    # Load boxes if not already loaded
                    if image_name not in self.current_boxes:
                        self.load_boxes_for_image(image_name)

                    if image_name in self.current_boxes:
                        threshold = self.image_confidence_thresholds.get(image_name, 0.1)
                        boxes = self.current_boxes[image_name]

                        if image_name in self.image_viewer.roi_polygons:
                            temp_current = self.current_image
                            temp_boxes = self.image_viewer.all_boxes

                            self.current_image = image_name
                            self.image_viewer.all_boxes = boxes

                            roi_boxes = self.image_viewer.get_boxes_in_roi(threshold=threshold)
                            total_count += len(roi_boxes)

                            self.current_image = temp_current
                            self.image_viewer.all_boxes = temp_boxes
                        else:
                            total_count += sum(1 for box in boxes if box[4] >= threshold)
                except Exception as e:
                    print(f"Error processing image {image_name}: {str(e)}")
        except Exception as e:
            print(f"Error calculating total count: {str(e)}")

        # Update labels
        roi_text = " (in ROI)" if self.current_image and self.current_image in self.image_viewer.roi_polygons else ""
        current_folder_text = f" ({os.path.dirname(self.current_image)})" if self.current_image and os.path.dirname(
            self.current_image) else ""

        self.current_boxes_label.configure(text=f"Current Image{roi_text}: {current_count} varroa mites")
        self.subfolder_boxes_label.configure(
            text=f"Current Subfolder{current_folder_text}: {subfolder_count} varroa mites")
        self.total_boxes_label.configure(text=f"Total (in ROIs or full images): {total_count} varroa mites")

        # Update display
        if self.current_image:
            self.image_viewer.draw_all_boxes()
            if self.current_image in self.image_viewer.roi_polygons:
                self.image_viewer.draw_roi()

    def update_confidence_threshold(self, value):
        """Update confidence threshold for the current image"""
        self.confidence_label.configure(text=f"Confidence Threshold: {value:.2f}")

        # Store the threshold for the current image
        if self.current_image:
            self.image_confidence_thresholds[self.current_image] = float(value)

            # Update the image viewer with new threshold without reloading the image
            if hasattr(self, 'image_viewer'):
                self.image_viewer.set_confidence_threshold(value)

            # Update statistics
            self.update_box_statistics()

    def update_progress(self, value, text="Processing..."):
        self.progress_bar.set(value)
        self.progress_label.configure(text=text)
        self.status_bar.configure(text=text)
        self.root.update_idletasks()

    def save_image_with_boxes(self, image_path, boxes, output_path):
        """Save image with visible bounding boxes drawn on it and detection count"""
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Draw each visible box
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color, 2px width

            # Add text with detection count
            num_detections = len(boxes)
            text = f"{num_detections} varroa mite{'s' if num_detections != 1 else ''} detected"
            if self.current_image in self.image_viewer.roi_polygons:
                text += " in ROI"

            # Settings for text background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 2
            padding = 10

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Calculate background rectangle position and size
            rect_x1 = 10
            rect_y1 = 10
            rect_x2 = rect_x1 + text_width + 2 * padding
            rect_y2 = rect_y1 + text_height + 2 * padding + baseline

            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image, dtype=cv2.CV_8U)

            # Draw text
            text_x = rect_x1 + padding
            text_y = rect_y1 + text_height + padding
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

            # If there's a ROI, draw it on the saved image
            if self.current_image in self.image_viewer.roi_polygons:
                points = self.image_viewer.roi_polygons[self.current_image]
                points = [(int(x), int(y)) for x, y in points]  # Convert to integer coordinates
                # Draw filled polygon with some transparency
                overlay = image.copy()
                cv2.fillPoly(overlay, [np.array(points)], (0, 255, 255))  # Yellow color (BGR)
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)  # 20% opacity
                # Draw ROI outline in yellow
                cv2.polylines(image, [np.array(points)], True, (0, 255, 255), 2)  # Yellow outline

            # Save the image
            cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"Error saving image with boxes: {str(e)}")
            raise

    def save_yolo_labels(self, image_path, boxes, output_path):
        """Save bounding boxes in YOLO format"""
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read image dimensions
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            img_height, img_width = image.shape[:2]

            # Convert boxes to YOLO format and save
            with open(output_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2 = map(float, box[:4])

                    # Convert to YOLO format (normalized coordinates)
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    # Write in YOLO format: class x_center y_center width height
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

        except Exception as e:
            print(f"Error saving YOLO labels: {str(e)}")
            raise

    def save_results(self):
        """Save all images and labels with current thresholds and ROIs"""
        try:
            # Disable save button while processing
            self.save_button.configure(state="disabled")

            # Create results directory structure
            results_dir = os.path.join(self.current_folder, "results")
            images_dir = os.path.join(results_dir, "images")
            labels_dir = os.path.join(results_dir, "labels")

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            # Get all images, including those in subdirectories
            all_images = []
            for root, _, files in os.walk(self.output_path):
                if "predict 0.1" in root:  # Skip predict directory
                    continue
                for f in files:
                    if f.lower().endswith('.jpg'):
                        rel_path = os.path.relpath(os.path.join(root, f), self.output_path)
                        all_images.append(rel_path)

            total_files = len(all_images)
            processed = 0

            # Store current image and scale to restore later
            temp_current = self.current_image
            temp_scale = self.image_viewer.scale

            # Update initial progress
            self.update_progress(0, "Starting to save results...")

            for image_name in all_images:
                # Calculate and update progress
                progress = processed / total_files
                self.update_progress(progress, f"Saving results: {image_name} ({processed}/{total_files})")

                # Get image-specific threshold
                threshold = self.image_confidence_thresholds.get(image_name, 0.1)

                # Load boxes if not already loaded
                if image_name not in self.current_boxes:
                    self.load_boxes_for_image(image_name)

                # Filter boxes based on threshold
                threshold_boxes = [box for box in self.current_boxes.get(image_name, []) if box[4] >= threshold]

                # Set current image for ROI checking
                self.current_image = image_name
                self.image_viewer.scale = 1.0

                # If this image has an ROI, filter boxes by ROI
                if image_name in self.image_viewer.roi_polygons:
                    self.image_viewer.all_boxes = threshold_boxes
                    final_boxes = self.image_viewer.get_boxes_in_roi()
                else:
                    final_boxes = threshold_boxes

                # Create subdirectory structure in output if needed
                subdir = os.path.dirname(image_name)
                if subdir:
                    os.makedirs(os.path.join(images_dir, subdir), exist_ok=True)
                    os.makedirs(os.path.join(labels_dir, subdir), exist_ok=True)

                # Save image with visible boxes
                image_path = os.path.join(self.output_path, image_name)
                output_image_path = os.path.join(images_dir, image_name)
                self.save_image_with_boxes(image_path, final_boxes, output_image_path)

                # Save labels in YOLO format
                output_label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
                self.save_yolo_labels(image_path, final_boxes, output_label_path)

                processed += 1
                self.root.update()

            # Restore original current image and scale
            self.current_image = temp_current
            self.image_viewer.scale = temp_scale

            # Restore current image's boxes
            if self.current_image:
                self.image_viewer.all_boxes = self.current_boxes.get(self.current_image, [])
                self.image_viewer.draw_all_boxes()
                if self.current_image in self.image_viewer.roi_polygons:
                    self.image_viewer.draw_roi()

            # Update final progress
            self.update_progress(1.0, "Results saved successfully!")
            messagebox.showinfo("Success", f"Results saved to {results_dir}")

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            messagebox.showerror("Error", f"Error saving results: {str(e)}")

        finally:
            # Re-enable save button
            self.save_button.configure(state="normal")
            # Reset progress bar after a short delay
            self.root.after(2000, lambda: self.update_progress(0, "Ready"))

    def load_boxes_for_image(self, image_name):
        """Load all detection boxes for an image"""
        if image_name not in self.current_boxes:
            try:
                # Load and check image dimensions
                image_path = os.path.join(self.output_path, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not read image: {image_path}")
                img_h, img_w = img.shape[:2]

                # Load all detection results
                # Split the image_name into directory and filename parts
                image_dir = os.path.dirname(image_name)
                image_basename = os.path.basename(image_name)

                # Construct the path to the labels file
                labels_path = os.path.join(
                    self.output_path,
                    "predict 0.1",
                    image_dir,  # Include subdirectory path
                    "labels",
                    os.path.splitext(image_basename)[0] + ".txt"
                )

                boxes = []
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            class_id, x, y, w, h, confidence = map(float, parts)

                            # Convert to pixel coordinates
                            x1 = int((x - w / 2) * img_w)
                            y1 = int((y - h / 2) * img_h)
                            x2 = int((x + w / 2) * img_w)
                            y2 = int((y + h / 2) * img_h)
                            boxes.append([x1, y1, x2, y2, confidence])

                self.current_boxes[image_name] = boxes

            except Exception as e:
                print(f"Error loading boxes for {image_name}: {str(e)}")
                self.current_boxes[image_name] = []

        return self.current_boxes[image_name]


    @staticmethod
    def get_resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def run_task_in_thread(self):
        self.select_button.configure(state="disabled")
        thread = threading.Thread(target=self.select_folder)
        thread.start()

    def select_folder(self):
        self.save_button.configure(state="disabled")
        self.apply_all_button.configure(state="disabled")
        try:
            # Clear the image listbox and disable it
            self.image_listbox.configure(state="normal")  # Enable temporarily to clear
            self.image_listbox.delete(0, tk.END)
            self.image_listbox.configure(state="disabled")  # Disable again

            # Reuse the cleanup method to remove processed_images folder
            self.cleanup(exit_program=False)

            # Reset current image and boxes
            self.current_image = None
            self.current_boxes = {}
            self.image_confidence_thresholds = {}

            # Update statistics to show zero counts
            self.update_box_statistics()

            # Clear the image viewer
            if hasattr(self, 'image_viewer'):
                self.image_viewer.canvas.delete("all")

            # Get the new folder
            self.current_folder = filedialog.askdirectory()
            if not self.current_folder:
                self.select_button.configure(state="normal")
                return

            self.current_folder = os.path.join(self.current_folder, "")
            self.output_path = os.path.join(self.current_folder, "processed_images")

            # Process images
            self.process_images()

            # Update image list
            self.update_image_list()

            # Run detection
            self.run_detection()

        except Exception as e:
            print(f"Error in processing: {str(e)}")
            messagebox.showerror("Error", f"Error in processing: {str(e)}")
        finally:
            self.select_button.configure(state="normal")
            self.select_button.configure(state="normal")

    def process_images(self):
        # Get all images recursively
        file_images = self.get_all_images(self.current_folder)

        if not file_images:
            messagebox.showwarning("Warning", "No JPG or DNG images found in selected folder or subfolders")
            self.select_button.configure(state="normal")
            return

        os.makedirs(self.output_path, exist_ok=True)
        print("**********************************")
        print("STEP 1: Detection of green strings")
        print("**********************************")

        total_files = len(file_images)
        for idx, (input_path, rel_path) in enumerate(file_images, 1):
            # Create output directory structure matching input
            output_dir = os.path.join(self.output_path, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)

            # Always save output as JPG
            output_path = os.path.join(self.output_path,
                                       os.path.splitext(rel_path)[0] + '.jpg')

            try:
                progress = idx / total_files
                self.update_progress(progress, f"Processing image {idx}/{total_files}: {rel_path}")

                # Handle DNG files
                if input_path.lower().endswith('.dng'):
                    img = process_dng(input_path)
                    if img is None:
                        print(f"Failed to process DNG file: {rel_path}")
                        continue

                    crop_img = crop_green_lines_from_array(img)
                    if crop_img is None:
                        cv2.imwrite(output_path, img)
                    else:
                        cv2.imwrite(output_path, crop_img)
                else:
                    # Handle JPG files
                    crop_img = crop_green_lines(input_path)
                    if crop_img is None:
                        shutil.copyfile(input_path, output_path)
                    else:
                        cv2.imwrite(output_path, crop_img)

            except Exception as e:
                print(f"Error processing image {rel_path}: {str(e)}")
                if not input_path.lower().endswith('.dng'):
                    shutil.copyfile(input_path, output_path)

    def run_detection(self):
        self.image_listbox.configure(state="disabled")
        if not self.output_path or not os.path.exists(self.output_path):
            return

        try:
            # Recursively find all JPG files in output_path and its subdirectories
            file_images = []
            for root, _, files in os.walk(self.output_path):
                for f in files:
                    if f.lower().endswith('.jpg'):
                        # Get full path but store relative path for later use
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, self.output_path)
                        file_images.append((full_path, rel_path))

            print("\n**********************************")
            print("STEP 2: Performing inference")
            print("**********************************")

            total_files = len(file_images)
            suma = 0
            conf = 0.1

            for idx, (img_path, rel_path) in enumerate(file_images, 1):
                try:
                    progress = idx / total_files
                    self.update_progress(progress, f"Analyzing image {idx}/{total_files}")

                    # Create the output directory structure matching input
                    output_dir = os.path.join(self.output_path, "predict 0.1")
                    labels_dir = os.path.join(output_dir, "labels")
                    os.makedirs(labels_dir, exist_ok=True)

                    # Create output directories preserving structure
                    rel_dir = os.path.dirname(rel_path)
                    predict_dir = os.path.join(self.output_path, "predict 0.1")
                    output_dir = os.path.join(predict_dir, rel_dir) if rel_dir else predict_dir
                    os.makedirs(output_dir, exist_ok=True)

                    results = self.model(
                        img_path, imgsz=(6016), max_det=2000, conf=0.1,
                        save=True, show_labels=False, line_width=2, save_txt=True, save_conf=True,
                        project=os.path.dirname(output_dir),
                        name=os.path.basename(output_dir) if rel_dir else "predict 0.1",
                        verbose=False, batch=1, exist_ok=True
                    )

                    # Save results with confidence scores
                    for result in results:
                        print(f"Total varroas in file {rel_path}: {len(result.boxes)}")
                        suma += len(result.boxes)
                        if len(result.boxes) > 0:
                            # Ensure subdirectory structure exists in labels directory
                            rel_dir = os.path.dirname(rel_path)
                            if rel_dir:
                                os.makedirs(os.path.join(labels_dir, rel_dir), exist_ok=True)

                                # Create output file path preserving directory structure
                            output_file = os.path.join(
                                os.path.dirname(output_dir),
                                os.path.basename(output_dir),
                                "labels",
                                os.path.splitext(os.path.basename(rel_path))[0] + '.txt'
                            )

                            # Ensure the directory exists
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)

                            with open(output_file, 'w') as f:
                                for box in result.boxes:
                                    conf = float(box.conf)
                                    x, y, w, h = box.xywhn[0].tolist()
                                    f.write(f"0 {x} {y} {w} {h} {conf}\n")
                except Exception as e:
                    print(f"Error analyzing image {rel_path}: {str(e)}")

            print("\nTotal varroas detected:", suma)

            # Initialize current_boxes dictionary with detections from txt files
            labels_dir = os.path.join(self.output_path, "predict 0.1", "labels")
            if os.path.exists(labels_dir):
                # Walk through all subdirectories in labels_dir
                for root, _, files in os.walk(labels_dir):
                    for txt_file in files:
                        if txt_file.endswith('.txt'):
                            # Get relative path from labels_dir to txt file
                            rel_txt_path = os.path.relpath(os.path.join(root, txt_file), labels_dir)
                            # Convert to corresponding image path
                            image_rel_path = os.path.splitext(rel_txt_path)[0] + '.jpg'

                            # Full paths for both txt and image files
                            txt_path = os.path.join(root, txt_file)
                            img_path = os.path.join(self.output_path, image_rel_path)

                            boxes = []

                            # Get image dimensions
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_h, img_w = img.shape[:2]

                                # Read boxes from txt file
                                with open(txt_path, 'r') as f:
                                    for line in f:
                                        parts = line.strip().split()
                                        class_id, x, y, w, h, confidence = map(float, parts)

                                        # Convert normalized coordinates to pixel coordinates
                                        x1 = int((x - w / 2) * img_w)
                                        y1 = int((y - h / 2) * img_h)
                                        x2 = int((x + w / 2) * img_w)
                                        y2 = int((y + h / 2) * img_h)
                                        boxes.append([x1, y1, x2, y2, confidence])

                                self.current_boxes[image_rel_path] = boxes
                                # Set initial confidence threshold for this image
                                self.image_confidence_thresholds[image_rel_path] = 0.1

            self.update_progress(1.0, "Analysis complete")
            # Enable the listbox after processing is complete
            self.image_listbox.configure(state="normal")
            self.update_image_list()
            # Update statistics with the newly loaded boxes
            self.update_box_statistics()

            # Enable both save and apply-to-all buttons after successful analysis
            self.save_button.configure(state="normal")
            self.apply_all_button.configure(state="normal")

        except Exception as e:
            print(f"Error in detection: {str(e)}")
            messagebox.showerror("Error", f"Error in detection: {str(e)}")

    def update_image_list(self):
        self.image_listbox.delete(0, tk.END)
        if self.output_path and os.path.exists(self.output_path):
            # Get all jpg files recursively, excluding predict 0.1 directory
            files = []
            for root, _, filenames in os.walk(self.output_path):
                # Skip the predict 0.1 directory and its subdirectories
                if "predict 0.1" in root:
                    continue

                for f in filenames:
                    if f.lower().endswith('.jpg'):
                        rel_path = os.path.relpath(os.path.join(root, f), self.output_path)
                        files.append(rel_path)

            max_width = 50  # Increased to accommodate paths
            for file in sorted(files):
                # Truncate long filenames
                if len(file) > max_width:
                    display_name = "..." + file[-(max_width - 3):]
                else:
                    display_name = file

                # Add padding using spaces
                padded_file = f"  {display_name}  "
                self.image_listbox.insert(tk.END, padded_file)

                # Store the full filename for reference
                self.image_listbox.fullnames = getattr(self.image_listbox, 'fullnames', {})
                self.image_listbox.fullnames[display_name] = file

            # Add extra visual spacing
            self.image_listbox.insert(tk.END, "")

    def highlight_same_folder_images(self):
        if not self.current_image:
            return

        current_folder = os.path.dirname(self.current_image)

        # Reset all items to default background
        for i in range(self.image_listbox.size()):
            self.image_listbox.itemconfig(i, {'bg': COLORS['background']})

        # Highlight items from the same folder
        for i in range(self.image_listbox.size()):
            item = self.image_listbox.get(i).strip()
            if item:  # Skip empty items
                full_name = self.image_listbox.fullnames.get(item.strip(), item.strip())
                if os.path.dirname(full_name) == current_folder:
                    # Use a lighter shade of blue for highlighting
                    self.image_listbox.itemconfig(i, {'bg': '#E3F2FD'})  # Light blue

    def on_select_image(self, event):
        if not self.image_listbox.curselection():
            return

        selected_display = self.image_listbox.get(self.image_listbox.curselection()).strip()
        if not selected_display:
            return

        # Get the original filename from our stored dictionary
        selected = self.image_listbox.fullnames.get(selected_display.strip(), selected_display.strip())

        try:
            # Update current image
            self.current_image = selected

            # Highlight same-folder images
            self.highlight_same_folder_images()

            # Rest of the existing code...
            image_path = os.path.join(self.output_path, selected)
            if not os.path.exists(image_path):
                messagebox.showerror("Error", f"Image file not found: {image_path}")
                return

            boxes = self.load_boxes_for_image(selected)
            threshold = self.image_confidence_thresholds.get(selected, 0.1)
            self.confidence_slider.set(threshold)
            self.confidence_label.configure(text=f"Confidence Threshold: {threshold:.2f}")

            self.image_viewer.load_image(image_path, boxes)
            self.image_viewer.set_confidence_threshold(threshold)
            self.roi_button.configure(state="normal")

            if hasattr(self.image_viewer, 'drawing_roi') and self.image_viewer.drawing_roi:
                self.roi_button.configure(fg_color=COLORS['secondary'])
                self.image_viewer.stop_roi_drawing()

            self.image_viewer.draw_roi()
            self.update_box_statistics()

        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def show_help(self):
        """Show help dialog with information about the program"""
        help_window = ctk.CTkToplevel(self.root)
        help_window.title("VarroDetector - Help")
        help_window.geometry("750x600")

        # Make the window modal
        help_window.transient(self.root)
        help_window.grab_set()

        # Center the window
        help_window.update_idletasks()
        width = help_window.winfo_width()
        height = help_window.winfo_height()
        x = (help_window.winfo_screenwidth() // 2) - (width // 2)
        y = (help_window.winfo_screenheight() // 2) - (height // 2)
        help_window.geometry(f'{width}x{height}+{x}+{y}')

        # Create a scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(
            help_window,
            fg_color=COLORS['surface']
        )
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Help content
        help_text = """
        VarroDetector - Quick Guide
        
        This software provides a simple way to count varroa mites from an image captured by a smartphone. 
        It runs under low hardware-constraints (no GPU is needed).
        
        How to use:
        1. Click "Select Input Folder" and choose a folder with your images
        2. Wait for the processing to complete
        3. Click on any image in the list to view it
        4. Use the confidence slider to adjust detection sensitivity
        5. Click "Save Results" when you're done

        Controls:
        - Zoom: Mouse wheel
        - Pan: Middle mouse button
        - Add varroa mite: Left click and drag
        - Delete varroa mite: Right click on box
        - Press (and keep pressing) the key h to temporarily hide the detections. 
        - Once the key h is released, the detections will be shown again
        
        Region of Interest (ROI):
        - Click the "Edit ROI" button to define a specific area for counting varroa mites
        - Left click to add points and create your ROI polygon
        - Double click to complete the ROI
        - Right click to delete the current ROI
        - The statistics will update to show mite counts only within the ROI
        - ROIs are saved per image and will be included in the final results
                
        
        The analysis will be also performed to any image contained in subfolders of the input folder. The confidence
        score can be set up individually for each image. Lower confidence score will show more detections, but 
        possibly with more false positives. The "Apply Threshold to All Images" button allows the user to quickly 
        set the same confidence threshold across all the images. The Save Button will save the images (with the 
        printed detections) and the coordinates of the detections (the labels) in a folder named  "results" within 
        the input folder.
        
        This software is completely free. If you wish to collaborate (for instance, providing new images with 
        corrected detections to improve the underlying AI model), please contact:
        - Jose Divasón (jose.divason@unirioja.es)
        - Jesús Yániz (jyaniz@unizar.es)

        """
        # Add the help text
        help_label = ctk.CTkLabel(
            scroll_frame,
            text=help_text,
            font=self.default_font,
            text_color=COLORS['text'],
            justify="left",
            wraplength=700
        )
        help_label.pack(padx=20, pady=20)

        # Close button at the bottom
        close_button = ctk.CTkButton(
            help_window,
            text="Close",
            command=help_window.destroy,
            height=40,
            font=self.default_font,
            fg_color=COLORS['primary'],
            hover_color=COLORS['accent']
        )
        close_button.pack(pady=(0, 20))


def main():
    app = ModernVarroaDetectorGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()

# pyinstaller --onefile --noconsole --add-data "model/weights/best.pt;model/weights" --add-data "icon.ico;." --add-data "icon_for_sidebar.png;." --icon "icon.ico" --splash splash.PNG varroa_mite_gui.py --name=VarroDetector