#!/usr/lib/env python

import os
import cv2
import random
import time
import numpy as np
import json
from uploader import Uploader
from datetime import datetime

PURPLE = (68, 54, 66)
BRAND = "Watch Monster v0.0.1"
isFun = True
REDUCTION_FACTOR = 1.  # Reduction factor for timing.
FONT = cv2.FONT_HERSHEY_SIMPLEX
OPACITY = 0.4
FRAMES_PER_PHOTO = 30
# Set Haar cascade path.
CASCADE_PATH = '/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
if not os.path.exists(CASCADE_PATH):
    # Try alternative file path
    CASCADE_PATH = 'face.xml'
    if not os.path.exists(CASCADE_PATH):
        raise NameError('File not found:', CASCADE_PATH)

FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def get_image_path():
    """
    Get path for saving a new image.
    """
    img_prefix = 'img_'
    extension = '.png'
    nr = 0
    for file in os.listdir(os.getcwd() + '/img'):
        if file.endswith(extension):
            file = file.replace(img_prefix, '')
            file = file.replace(extension, '')
            # print file
            file_nr = int(file)
            nr = max(nr, file_nr)
    img_nr = nr + 1
    imagePath = 'img/' + str(img_prefix) + \
        str(img_nr) + str(extension)
    return imagePath


class WatchMonster(object):

    def __init__(self, piCam=False, windowSize=(1280 // 2, 1024 // 2), resolution=(1280 // 2, 1024 // 2), gray=False, debug=True):
        self.piCam = piCam
        self.windowSize = windowSize
        self.gray = gray
        self.calibrated = False
        # To prepare window transition.
        self.photo = cv2.imread('img/img_1.png')
        self.screenheight, self.screenwidth = self.windowSize

        # Setup for Raspberry Pi.
        if 'raspberrypi' in os.uname():
            self.initialize_raspberry(resolution)
        else:
            self.initialize_webcam()

        # Reinitialize screenwidth and height in case changed by system.
        self.screenwidth, self.screenheight = self.frame.shape[:2]
        print("first screenheight:", self.screenheight,
              self.screenwidth, self.frame.shape)
        print("Window size:", self.frame.shape)

        # Complete setup.
        self.setup_system()

    def initialize_webcam(self):
        """ Initialize camera and screenwidth and screenheight.
        """
        self.raspberry = False
        self.cam = cv2.VideoCapture(0)
        _, self.frame = self.cam.read()
        # Update class variables.
        # self.screenheight, self.screenwidth = self.frame.shape[:2]
        print(self.frame.shape)
        self.cam.set(3, self.screenwidth)
        self.cam.set(4, self.screenheight)
        _, self.frame = self.cam.read()
        cv2.imshow('WatchMonster', self.frame)

    def initialize_raspberry(self, resolution):
        """ Set up piCamera module or webcam.

        """
        print(BRAND, "loading")
        self.raspberry = True
        self.resolution = resolution
        # Set up picamera module.
        if self.piCam:
            self.setup_picamera()
        else:  # Use webcam (Note: not completely tested).
            self.cam = cv2.VideoCapture(0)
            _, self.frame = self.cam.read()
            # self.cam.set(3, self.screenwidth)
            # self.cam.set(4, self.screenheight)

    def setup_picamera(self):
        """ Set up piCamera for rasbperry pi camera module.

        """
        from picamera import PiCamera
        from picamera.array import PiRGBArray
        piCamera = PiCamera()
        # self.piCamera.resolution = (640, 480)
        piCamera.resolution = self.resolution[0], self.resolution[1]
        self.screenwidth, self.screenheight = piCamera.resolution
        # self.piCamera.framerate = 10
        piCamera.hflip = True
        piCamera.brightness = 55
        self.rawCapture = PiRGBArray(
            piCamera, size=(self.screenwidth, self.screenheight))
        self.frame = np.empty(
            (self.screenheight, self.screenwidth, 3), dtype=np.uint8)
        self.piCamera = piCamera
        time.sleep(1)

    def setup_system(self):
        """ Initialize variables, set up icons and face cascade.

        """
        self.looping = True
        self.calibrated = False
        self.looping = True
        self.frameCount = FRAMES_PER_PHOTO
        self.curr_level = 0
        self.result = []
        self.uploader = Uploader()

        print("Camera initialized")
        if not self.raspberry:
            print("MAC or PC initialize")
            self.cam.set(3, self.screenwidth)
            self.cam.set(4, self.screenheight)
        self.currCount = None
        self.photoMode = False
        cv2.namedWindow("WatchMonster", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            "WatchMonster", self.windowSize[0], self.windowSize[1])
        # Returns - TypeError: Required argument 'prop_value' (pos 3) not found
        # cv2.setWindowProperty(
        #     "PartyPi", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("PartyPi", cv2.WND_PROP_AUTOSIZE,
        #                       cv2.WINDOW_AUTOSIZE)
        if self.piCam == True:
            # Capture frames from the camera.
            for _frame in self.piCamera.capture_continuous(self.rawCapture, format='bgr', use_video_port=True):
                # self.frame = cv2.flip(_frame.array, 1)
                self.frame = _frame.array
                self.frame.flags.writeable = True
                self.screenheight, self.screenwidth = self.frame.shape[:2]
                # TODO: Consider passing frame as local variable rather than
                # global.
                self.video_loop()
        else:
            while self.looping:
                self.video_loop()

    def video_loop(self):
        global FRAMES_PER_PHOTO
        """ Start the video loop. Listen for escape key.

        """
        # TODO: Check if following line is redundant.
        self.screenheight, self.screenwidth = self.frame.shape[:2]
        if self.curr_level == 0:
            if not self.calibrated & self.frameCount < FRAMES_PER_PHOTO:  # Allow loading
                start = datetime.now()
                self.level0()
                end = datetime.now()
                delta = end - start
                print("Time per frame: {}", str(delta.total_seconds()))
                FRAMES_PER_PHOTO = int(
                    delta.total_seconds() * 10)  # adjust for rpi
                self.calibrated = True
            else:
                self.level0()

        if self.frameCount == 0:
            self.frameCount = FRAMES_PER_PHOTO
        keypress = cv2.waitKey(1) & 0xFF
        if self.curr_level == 0:
            if keypress == 32:  # Spacebar
                self.take_photo()

        # Catch escape key 'q'.
        keypress = cv2.waitKey(1) & 0xFF

        # Clear the stream in preparation for the next frame.
        if self.piCam == True:
            self.rawCapture.truncate(0)

        self.listen_for_end(keypress)

    def level0(self):
        """ Get image and save when faces are present.

        """
        self.capture_frame()

        # Detect and recognize faces.
        faces = self.find_faces()
        self.recognize_faces(faces)
        self.frame = self.draw_face_box(faces, self.frame)

        # Write timestamp
        self.add_text(self.frame, str(datetime.now().strftime("%A, %d. %B %Y %I:%M%p:%S")), (
            10, self.screenheight - 50), size=0.7, thickness=0.1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL)

        # Display frame with watermark.
        if isFun:
            BRAND = 'Big Brother v0.1'
        self.add_text(self.frame, BRAND, ((self.screenwidth // 5) * 3,
                                          self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        if isFun:
            if len(faces):
                rows, cols, ch = self.frame.shape
                x, y, w, h = faces[0]
                faceImg = self.frame[y:y + h, x:x + w]
                height, width = faceImg.shape[:2]
                res = cv2.resize(faceImg, (cols, rows),
                                 interpolation=cv2.INTER_CUBIC)
                cv2.imshow('WatchMonster', res)
        else:
            cv2.imshow('WatchMonster', self.frame)

    def capture_frame(self):
        """ Capture frame-by-frame.

        """

        if not self.piCam:
            ret, frame = self.cam.read()
            self.frame = cv2.flip(frame, 1)

        self.overlay = self.frame.copy()

    def find_faces(self):
        global FACE_CASCADE
        """
        Find faces using Haar cascade.
        """
        # TODO: Add masking to limit search space to user's probable position to avoid
        # mixing people.
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50),
            #         flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            flags=0)
        # If image has been taken, countdown until next image
        if len(faces) == 1 & self.frameCount > 0:
            self.frameCount -= 1
        elif len(faces) == 1 and self.frameCount == 0 and not self.debug:  # Time to take photo
            self.take_photo()
        elif len(faces) == 0:  # No faces found, reset counter
            self.frameCount == FRAMES_PER_PHOTO

        return faces

    def draw_face_box(self, faces, frame):
        for (x, y, w, h) in faces:
            if isFun:
                pass
            else:
                cv2.rectangle(frame, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                self.add_text(frame, "Name", (x, y + h + 20))
            print("draw face")
        print("draw any face")
        return frame

    def recognize_faces(self, faces):
        """Look up face and print name beneath."""
        # FIXME: Add function details.
        return None

    def add_text(self, frame, text, origin, size=1.0, color=(255, 255, 255), thickness=1, font=FONT):
        """
        Put text on current frame.
        """
        originx, originy = origin
        origin = (int(originx), int(originy))
        cv2.putText(frame, text, origin,
                    font, size, color, 2)

    def take_photo(self):
        """ Take photo and prepare to write, then send to PyImgur.

        """

        photo = self.frame  # Take photo
        # Write timestamp
        self.add_text(photo, str(datetime.now().strftime("%A, %d. %B %Y %I:%M%p:%S")), (10,
                                                                                        self.screenheight - 50), size=0.7, thickness=0.1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL)

        # Display frame with watermark.
        self.add_text(photo, BRAND, ((self.screenwidth // 5) * 3,
                                     self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        imagePath = get_image_path()
        print("Attempting to upload {}".format(imagePath))
        # If internet connection is poor, use black and white image.
        if self.gray:
            bwphoto = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(imagePath, bwphoto)
        else:
            cv2.imwrite(imagePath, photo)

        # FIXME: Replace with threading only
        # Upload image via Uploader module
        self.uploader.add_image_to_queue(imagePath)
        self.frameCount = FRAMES_PER_PHOTO

    def listen_for_end(self, keypress):
        """ Listen for 'q' to end program.

        """
        if keypress != 255:
            print(keypress)
            if keypress == ord('q'):  # 'q' pressed to quit
                print("Escape key entered")
                self.looping = False
                self.end_system()

    def end_system(self):
        """ When everything is done, release the capture.

        """
        if not self.piCam:
            self.cam.release()
            self.add_text(self.frame, "Press any key to quit_",
                          (self.screenwidth // 4, self.screenheight // 3))
            # self.presentation(self.frame)
            self.add_text(self.frame, BRAND, ((self.screenwidth // 5) * 4,
                                              self.screenheight // 7), color=(255, 255, 255), size=0.5, thickness=0.5)
        else:
            self.piCamera.close()

        cv2.imshow('WatchMonster', self.frame)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """ Run application.
    """
    app = WatchMonster()


if __name__ == '__main__':
    main()
