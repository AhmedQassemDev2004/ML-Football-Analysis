# Configuration file for ML Football Analysis

# Model Configuration
MODEL_PATH = "./models/best.pt"
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 20

# Video Configuration
INPUT_VIDEO_PATH = "./input_videos/08fd33_4.mp4"
OUTPUT_VIDEO_PATH = "./output_videos/output_video.avi"
STUB_PATH = "stubs/track_stubs_new.pkl"

# Processing Configuration
TEST_FRAMES_LIMIT = 30
FPS = 24

# Team Assignment Configuration
KMEANS_CLUSTERS = 2
KMEANS_INIT = "k-means++"
KMEANS_N_INIT = 10

# Drawing Configuration
ELLIPSE_THICKNESS = 3
RECTANGLE_WIDTH = 40
RECTANGLE_HEIGHT = 20
FONT_SCALE = 0.5
FONT_THICKNESS = 2

# Colors (BGR format for OpenCV)
REFEREE_COLOR = (0, 255, 255)  # Yellow
BALL_COLOR = (0, 255, 0)       # Green
TEXT_COLOR = (0, 0, 0)         # Black
BACKGROUND_COLOR = (255, 255, 255)  # White
