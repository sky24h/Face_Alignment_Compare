# coding: utf-8

__author__ = 'cleardusk'

import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
import glob
import shutil

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark_mask, get_suffix
import cv2
import sys

# sys.path.append("./Xseg/")

# from Xseg.XSegUtil import initialize_model_xseg, apply_xseg
import os

def get_landmarks(frame_list):#, xseg, xseg_res):
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    # if onnx:
    #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    #     os.environ['OMP_NUM_THREADS'] = '4'

    #     from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    #     from TDDFA_ONNX import TDDFA_ONNX

    #     face_boxes = FaceBoxes_ONNX()
    #     tddfa = TDDFA_ONNX(**cfg)
    # else:
    mode = 'gpu'
    gpu_mode = mode == 'gpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    # Given a video path
    # fn = video_fp.split('/')[-1]
    # reader = imageio.get_reader(video_fp)

    # fps = reader.get_meta_data()['fps']
    # suffix = get_suffix(video_fp)
    # video_wfp = f'examples/results/videos/{fn.replace(suffix, "")}_{opt}_smooth.mp4'
    # writer = imageio.get_writer(video_wfp, fps=fps)
    img_list = []
    landmarks_list = []
    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = 0, 0
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    opt = '2d_sparse'
    dense_flag = opt in ('2d_dense', '3d',)
    pre_ver = None
    start, end = -1, -1
    for i, frame in enumerate(frame_list):
        if start > 0 and i < start:
            continue
        if end > 0 and i > end:
            break

        # frame_bgr = cv2.imread(frame)
        # = frame[..., ::-1]  # RGB->BGR
        # frame_bgr = cv2.resize(frame[:,140:-140,:], (256,256))
        frame_bgr = frame

        if i == 0:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                # boxes = face_boxes(frame_bgr)
                # boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)
            # print(ver_ave.shape)
            if opt == '2d_sparse':
                img_draw, landmarks = cv_draw_landmark_mask(queue_frame[n_pre], ver_ave)  # since we use padding
            elif opt == '2d_dense':
                img_draw, landmarks = cv_draw_landmark_mask(queue_frame[n_pre], ver_ave, thickness=1)
            elif opt == '3d':
                img = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=1.0)
            else:
                raise ValueError(f'Unknown opt {opt}')

            # masked, _ = apply_xseg(img, xseg, xseg_res)
            # writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB
            # print(img.shape, masked.shape, img_draw.shape)
            # save_img = cv2.hconcat([img, masked, img_draw])
            # save_img = cv2.resize(save_img, (128*3, 128))
            # img_list.append(img_draw)
            landmarks_list.append(landmarks)
            queue_ver.popleft()
            queue_frame.popleft()
    return landmarks_list
    # we will lost the last n_next frames, still padding
    # for _ in range(n_next):
    #     queue_ver.append(ver.copy())
    #     queue_frame.append(frame_bgr.copy())  # the last frame

    #     ver_ave = np.mean(queue_ver, axis=0)
    #     # print(ver_ave.shape)
    #     if opt == '2d_sparse':
    #         img_draw = cv_draw_landmark_mask(queue_frame[n_pre], ver_ave)  # since we use padding
    #     elif opt == '2d_dense':
    #         img_draw = cv_draw_landmark_mask(queue_frame[n_pre], ver_ave, size=1)
    #     elif opt == '3d':
    #         img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
    #     else:
    #         raise ValueError(f'Unknown opt {opt}')

    #         # masked, _ = apply_xseg(img, xseg, xseg_res)
    #         # writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB
    #         # print(img.shape, masked.shape, img_draw.shape)
    #         # save_img = cv2.hconcat([img, masked, img_draw])
    #         img_list.append(img_draw)
    #     queue_ver.popleft()
    #     queue_frame.popleft()

    # # writer.close()
    # for n, img in enumerate(img_list):
    #     cv2.imwrite(os.path.join(save_dir, str(n+1)+'.jpg'), img)
    # np.save(os.path.join(save_dir, 'landmarks'), np.array(landmarks_list))
    # print(f'Dump to {video_wfp}')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='The smooth demo of video of 3DDFA_V2')
#     parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
#     parser.add_argument('-f', '--video_fp', type=str)
#     parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
#     parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
#     parser.add_argument('-n_next', default=0, type=int, help='the next frames of smoothing')
#     parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
#     parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
#     parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
#     parser.add_argument('--onnx', action='store_true', default=False)

#     args = parser.parse_args()
#     dirs = glob.glob(video_fp+'/*')
#     dirs.sort()
#     dataset_name = os.path.basename(video_fp)
#     # xseg, xseg_res = initialize_model_xseg()
#     for dir_ in tqdm(dirs):
#         frame_list = glob.glob(os.path.join(dir_, '*.jpg')) + glob.glob(os.path.join(dir_, '*.png'))
#         # if len(frame_list) > 150:
#         #     pass
#         # else:
#         #     continue
#         frame_list = [cv2.imread(f) for f in frame_list]

#         save_dir = dir_.replace(dataset_name, dataset_name+'_landmarks')
#         os.makedirs(save_dir, exist_ok=True)
#         # shutil.copyfile(dir_+'/mel.npy', save_dir+'/mel.npy')
#         try:
#             main(args, frame_list, save_dir)#, xseg, xseg_res)
#         except:
#             print(dir_)
