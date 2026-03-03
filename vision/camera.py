import cv2


class CameraStream:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam. Check camera index/permissions.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from webcam.")
        return frame

    def release(self) -> None:
        self.cap.release()
