
# Credits to: 
# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import numpy as np
import cv2
import datetime

# DONE: put into own project
# DONE: detect state changes explicitly
# DONE: break loop down into methods
# DONE: play buffer back and forth
# DONE: enable buffer for two faces
# TODO: hide debug output
# TODO: implement frames_to_skip
# TODO: work out what to do if face re-appears for less than one buffer length
# TODO: explicit action for enabling buffer (via web server?)
# TODO: gstreamer pipeline for external camera
# DONE: mirror image

class State:    
    show_live = 0
    show_buffered = 1
    show_preset = 2
    display_to_string = ["Live", "Buffered", "Preset"]

    reason_init = 100
    reason_user = 101
    reason_no_face = 102
    reason_face = 103
    reason_multiple_faces = 104    
    
    def __init__(self, buffer_start=0, buffer_end=100, frame_tolerance=10, non_live_default=show_buffered, frames_to_skip=21,mark_faces=True, detect_interval=10):
        self.display = State.show_live
        self.reason = State.reason_init
        self.buffer_start = buffer_start
        self.buffer_end = buffer_end
        self.buffer_index = 0 
        self.read_step = 1 # -1 for decrement
        self.frame_tolerance = frame_tolerance
        self.frame_count = 0
        self.last_fps_frame = 0
        self.last_timestamp = datetime.datetime.now()
        self.last_frame_with_faces = [ 0, 0, 0]
        self.buffer_write_index = -1
        self.buffer_full = False
        self.non_live_default = non_live_default
        self.frames_to_skip = frames_to_skip
        self.mark_faces = mark_faces
        self.detect_interval = detect_interval

    def get_buffer_size(self):
        return self.buffer_end + self.frames_to_skip
    
    def next_buffer_read_index(self):
        self.buffer_index = self.buffer_index + self.read_step
        if self.buffer_index >= self.buffer_end:
            self.buffer_index = self.buffer_end -1
            self.read_step = -1
        if self.buffer_index < self.buffer_start:
            self.buffer_index = self.buffer_start
            self.read_step = 1
        return self.buffer_index

    def next_buffer_write_index(self):
        self.buffer_write_index += 1
        if self.buffer_write_index >= self.buffer_end:
            self.buffer_write_index = self.buffer_start
            self.buffer_full = True
        return self.buffer_write_index

    def observed_face_count(self, faces):
        self.last_frame_with_faces[min(faces,2)] = self.frame_count

        if self.reason == State.reason_user:
            return

        if self.reason == State.reason_init:
            if self.buffer_full:
                self.reason = State.reason_face
                self.display = State.show_live
                print("Detected buffer full @"+str(self.frame_count))
            else:
                return

        if self.display == State.show_live:
            if self.frame_count - self.last_frame_with_faces[1] > self.frame_tolerance:
                self.display = self.non_live_default
                self.reason = State.reason_no_face if self.last_frame_with_faces[0] > self.last_frame_with_faces[2] else State.reason_multiple_faces
                print("Detected 0 or 2 faces @"+str(self.frame_count))
        else:
            if self.frame_count - self.last_frame_with_faces[1] < self.frame_tolerance:
                self.display = self.show_live
                self.reason = self.reason_face
                print("Detected 1 face @"+str(self.frame_count))

    def register_frame(self):
        self.frame_count += 1
        
    def should_run_face_reco(self):
        return self.frame_count % self.detect_interval == 0 and self.reason != State.reason_user

    def get_fps(self):
        new_timestamp = datetime.datetime.now()
        num_frames = self.frame_count - self.last_fps_frame
        fps = num_frames / (new_timestamp - self.last_timestamp).total_seconds()
        if num_frames > self.frame_tolerance:
            self.last_timestamp = new_timestamp
            self.last_fps_frame = self.frame_count
        return fps

    def get_overlay_text(self):
        return State.display_to_string[self.display] + " - " + str(self.frame_count) + " FPS: " + str(round(self.get_fps()))
        
    def user_request_live(self):
        self.display = State.show_live
        self.reason = State.reason_user

    def user_request_buffered(self):
        self.display = State.show_buffered
        self.reason = State.reason_user
    
    def user_request_preset(self):
        self.display = State.show_preset
        self.reason = State.reason_user
    
    def user_request_auto(self):
        self.user_request_live(self)
        self.reason = State.reason_face


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=820,
    display_height=616,
    framerate=21,
    flip_method=0, # 2 for 180 degrees
):
    return (
        "nvarguscamerasrc ! "
        # "v4l2src device=/dev/video1 |"
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def output(text):
    cv2.setWindowTitle("Face Detect", text)


def loop(
    pipeline,
    face_cascade,
    state,
    detect_interval = 10,
):
    if pipeline.isOpened():
        cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)
        # frames_since_last_face = 0
        frame_num = 0
        buffer = None
        last_faces = None

        # fps = 0.0
        # last_timestamp = datetime.datetime.now()
        # frame_tolerance = 24

        
        while cv2.getWindowProperty("Face Detect", 0) >= 0:
            ret, img = pipeline.read()   

            if not ret:
                output ("Waiting for camera")
                wait(10)            
                continue

            state.register_frame()
            cv2.putText(img, str(state.frame_count),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            # if (frame_num % frame_tolerance == 0 and frame_num > 0):            
            #     new_timestamp = datetime.datetime.now()
            #     fps = frame_tolerance / (new_timestamp - last_timestamp).total_seconds()
            #     last_timestamp = new_timestamp
            # skip missed frames
            # while ret:
            #     ret, img2 =  cap.read()                
            #     if ret: 
            #         print ("skipping frame")
            #         img = img2

            
            # if frame_num % detect_interval == 0:                
            if state.should_run_face_reco():
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # faces = None
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                last_faces = faces
            else:
                faces = last_faces

            # cv2.putText(img, str(frame_num) + " FPS: " + str(round(fps)),(50,150),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            overlay_text = state.get_overlay_text()
            

            if not buffer:
                buffer = [img] * state.get_buffer_size()
            
            state.observed_face_count(0 if faces is None else len(faces))

            if faces is None or len(faces) == 0:
                # frames_since_last_face = frames_since_last_face + 1
                # print("frames since last face: " + str(frames_since_last_face))
                output(overlay_text)                
            else:
                if state.mark_faces:
                    for (x, y, w, h) in faces:
                        output("face detected: " + overlay_text)
                        # frames_since_last_face = 0
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # roi_gray = gray[y : y + h, x : x + w]
                        # roi_color = img[y : y + h, x : x + w]

            if state.display == State.show_buffered:
                img = buffer[state.next_buffer_read_index()]
            # if frames_since_last_face > frame_tolerance and frame_num > buffer_size:
            #     img = buffer[frames_since_last_face % (buffer_size - frames_to_skip)]
            #     cv2.putText(img, "buffered",(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            else:
                # buffer[frame_num % buffer_size] = img
                buffer[state.next_buffer_write_index()] = img

            # TODO: reset buffer and frame_num on re-entry

            frame_num = frame_num + 1
            # cv2.putText(img, overlay_text,(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            cv2.imshow("Face Detect", img)
            keyCode = cv2.waitKey(3) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break

        pipeline.release()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    flip_method = 6 # 6 for vertical flip (if upside down) 4 for horizontal flip
    state = State(buffer_end=100, frames_to_skip=21)
    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )
    pipeline = cv2.VideoCapture(gstreamer_pipeline(
        flip_method=flip_method,capture_width=820,capture_height=616),  cv2.CAP_GSTREAMER)
    loop(pipeline, face_cascade, state)
    cv2.destroyAllWindows()
