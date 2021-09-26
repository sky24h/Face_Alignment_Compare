import dlib
from imutils import face_utils
import cv2


def get_landmarks(frame_list):
    predictor_path = './dlib_python/shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)

    dlibRect = dlib.rectangle(0, 0, 256, 256)
    landmarks = []
    for frame in frame_list:
        landmark = face_predictor(frame, dlibRect)
        landmark = face_utils.shape_to_np(landmark)
        landmarks.append(landmark)

    return landmarks


