import dlib
from imutils import face_utils
import cv2


def get_landmarks(frame_list):
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_python/shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    shape = frame_list[0].shape[1]
    dlibRect = dlib.rectangle(0, 0, shape, shape)
    landmarks = []
    for frame in frame_list:
        faces = face_detector(frame, 1)
        landmark = face_predictor(frame, faces[0])
        landmark = face_utils.shape_to_np(landmark)
        landmarks.append(landmark)

    return landmarks


