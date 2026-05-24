import os
import cv2
from typing import Generator, Tuple, Optional

class VideoStreamHandler:
    """
    Utility wrapper for robust OpenCV video reading and writing.
    Handles file checks, property retrieval, and seamless frame output.
    """
    
    def __init__(self, input_path: str):
        """
        Initializes the video capture stream.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found at: {input_path}")
            
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {input_path}")
            
        # Retrieve stream specs
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.writer: Optional[cv2.VideoWriter] = None

    def get_properties(self) -> Tuple[int, int, float, int]:
        """
        Returns (width, height, fps, total_frames).
        """
        return self.width, self.height, self.fps, self.total_frames

    def frame_generator(self) -> Generator[cv2.Mat, None, None]:
        """
        Generates frames sequentially until the end of the video.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
        self.release_input()

    def init_writer(self, output_path: str, codec: str = "mp4v") -> None:
        """
        Initializes the OpenCV VideoWriter to write outputs.
        """
        # Ensure the directory path exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )

    def write_frame(self, frame: cv2.Mat) -> None:
        """
        Writes a single frame to the output stream.
        """
        if self.writer is not None:
            self.writer.write(frame)

    def release_input(self) -> None:
        """
        Releases the input capture resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def release_output(self) -> None:
        """
        Releases the video writer resources.
        """
        if self.writer is not None:
            self.writer.release()
            
    def release_all(self) -> None:
        """
        Releases both inputs and outputs.
        """
        self.release_input()
        self.release_output()
