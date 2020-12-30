
# Credits to: 
# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import numpy as np
import cv2
import datetime

# TODO: gstreamer pipeline for external camera
# TODO: put into own project
# TODO: detect state changes explicitly
# TODO: break loop down into methods
# TODO: play buffer back and forth
# TODO: enable buffer for two faces
# TODO: hide debug output
# TODO: explicit action for enabling buffer (via web server?)


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


def face_detect(
    frame_tolerance=20,
    buffer_size = 100,
    flip_method = 2,
    frames_to_skip= 24,
    detect_interval = 10
):
    face_cascade = cv2.CascadeClassifier(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(gstreamer_pipeline(
        flip_method=flip_method,capture_width=820,capture_height=616),  cv2.CAP_GSTREAMER)
    if cap.isOpened():
        cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)
        frames_since_last_face = 0
        frame_num = 0
        buffer = None
        last_faces = None

        fps = 0.0
        last_timestamp = datetime.datetime.now()

        
        while cv2.getWindowProperty("Face Detect", 0) >= 0:
            ret, img = cap.read()   
            img2 = img

            if not ret:
                print ("Waiting for camera")
                wait(10)            
                continue


            if (frame_num % frame_tolerance == 0 and frame_num > 0):
                new_timestamp = datetime.datetime.now()
                fps = frame_tolerance / (new_timestamp - last_timestamp).total_seconds()
                last_timestamp = new_timestamp
            # skip missed frames
            # while ret:
            #     ret, img2 =  cap.read()                
            #     if ret: 
            #         print ("skipping frame")
            #         img = img2

            
            if frame_num % detect_interval == 0:                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # faces = None
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                last_faces = faces
            else:
                faces = last_faces

            cv2.putText(img, str(frame_num) + " FPS: " + str(round(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            if not buffer:
                buffer = [img] * buffer_size
            
            if faces is None or len(faces) == 0:
                frames_since_last_face = frames_since_last_face + 1
                print("frames since last face: " + str(frames_since_last_face))
            else:
                for (x, y, w, h) in faces:
                    print("face detected: " + str(frame_num))
                    frames_since_last_face = 0
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi_gray = gray[y : y + h, x : x + w]
                    roi_color = img[y : y + h, x : x + w]

            if frames_since_last_face > frame_tolerance and frame_num > buffer_size:
                img = buffer[frames_since_last_face % (buffer_size - frames_to_skip)]
                cv2.putText(img, "buffered",(50,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            else:
                buffer[frame_num % buffer_size] = img

            # TODO: reset buffer and frame_num on re-entry

            frame_num = frame_num + 1

            cv2.imshow("Face Detect", img)
            keyCode = cv2.waitKey(3) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    face_detect()
