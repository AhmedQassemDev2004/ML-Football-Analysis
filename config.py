# ===========================
# ⚙️ Configuration File
# ===========================

# --- Model Configuration --- #
MODEL_PATH = "./models/best.pt"
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 20

# --- Video I/O --- #
INPUT_VIDEO_PATH = "./input_videos/input_4.mp4"
OUTPUT_VIDEO_PATH = "./output_videos/output_video_4.avi"

# --- Stubs (for caching detections & movement) --- #
STUB_PATH = "stubs/track_stubs_new_4.pkl"
CAMERA_MOVEMENT_STUB = "stubs/camera_movement_stub_4.pkl"

# --- Processing --- #
TEST_FRAMES_LIMIT = 30
FPS = 24

# --- Team Assignment --- #
KMEANS_CLUSTERS = 2
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 10

# --- Drawing (OpenCV Config) --- #
ELLIPSE_THICKNESS = 3
RECTANGLE_WIDTH = 40
RECTANGLE_HEIGHT = 20
FONT_SCALE = 0.5
FONT_THICKNESS = 2

# --- Colors (BGR format for OpenCV) --- #
REFEREE_COLOR = (0, 255, 255)   # Yellow
BALL_COLOR = (0, 255, 0)        # Green
TEXT_COLOR = (0, 0, 0)          # Black
BACKGROUND_COLOR = (255, 255, 255)  # White
