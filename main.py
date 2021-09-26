import numpy as np
import cv2, os, sys, glob
from tqdm import tqdm
import subprocess
from natsort import natsorted
import sys
sys.path.append('./face_3ddfa_v2')
device = "cuda"

import face_alignment
from dlib_python.get_landmarks import get_landmarks as get_landmarks_dlib
from face_alignment_python.get_landmarks import get_landmarks as get_landmarks_face_alignment
from face_3ddfa_v2.get_landmarks import get_landmarks as get_landmarks_3ddfa_v2


landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }


def draw_mask(img_ori, image_landmarks, color=(0, 0, 0), thickness=1):
    # image_landmarks = pts[:2].T
    img = np.ones(img_ori.shape, np.uint8)*255
    int_lmrks = np.array(image_landmarks, dtype=np.int)
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
    left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
    nose = int_lmrks[slice(*landmarks_68_pt["nose"])]
    jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
    right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
    left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
    # open shapes
    cv2.polylines(img, tuple(np.array([v]) for v in ( right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])) )),
                False, color, thickness=thickness, lineType=cv2.LINE_AA)
    # closed shapes
    cv2.polylines(img, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                True, color, thickness=thickness, lineType=cv2.LINE_AA)

    return img


if __name__ == '__main__':

    data_root = '../../datasets/joint_datasets_0811'
    dirs = glob.glob(os.path.join(data_root,'*'))
    dirs.sort()
    dataset_name = os.path.basename(data_root)
    # xseg, xseg_res = initialize_model_xseg()
    save_videos = '../results'
    os.makedirs(save_videos, exist_ok=True)
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                flip_input=False, device=device)
    for dir_ in tqdm(dirs):
        try:
            frame_list = glob.glob(os.path.join(dir_, '*.jpg')) + glob.glob(os.path.join(dir_, '*.png'))
            frame_list = [cv2.resize(cv2.imread(f), (256,256)) for f in natsorted(frame_list)]

            save_dir = dir_.replace(dataset_name, dataset_name+'_landmarks')
            os.makedirs(save_dir, exist_ok=True)
            # shutil.copyfile(dir_+'/mel.npy', save_dir+'/mel.npy')
            # print(frame_list[0].shape)

            landmarks_list_1 = get_landmarks_dlib(frame_list)#, xseg, xseg_res)
            landmarks_list_2 = get_landmarks_face_alignment(frame_list, detector=detector)#, xseg, xseg_res)
            landmarks_list_3 = get_landmarks_3ddfa_v2(frame_list)#, xseg, xseg_res)

            save_video = os.path.join(save_videos, os.path.basename(dir_)+'.mp4')
            results = []
            # print(landmarks_list_1[0].shape)
            for i in range(len(landmarks_list_1)):
                img_ori = frame_list[i]
                img_1 = draw_mask(img_ori, landmarks_list_1[i])
                img_2 = draw_mask(img_ori, landmarks_list_2[i])
                img_3 = draw_mask(img_ori, landmarks_list_3[i])
                result = cv2.hconcat([img_ori, img_1, img_1, img_1])
                results.append(result)
                save_path = os.path.join(save_dir, str(i+1)+'.png')
                cv2.imwrite(save_path, result)
            cmd = 'ffmpeg -y -hide_banner -loglevel error -i ' + save_dir + '/%d.png  -pix_fmt yuv420p ' + save_video
            # print(cmd)
            subprocess.call(cmd, shell=True)
            # fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            # out = cv2.VideoWriter(save_video, fourcc, 25.0, (256*3, 256))

            # for i in range(len(landmarks_list_1)):
            #     out.write(results[i])
            # out.release()
        except Exception as e:
            print(e)
        # exit()


