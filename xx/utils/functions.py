# coding: utf-8

__author__ = 'cleardusk'

import numpy as np
import cv2
from math import sqrt
import matplotlib.pyplot as plt

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }

def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """Draw landmarks using matplotlib"""
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    dense_flag = kwargs.get('dense_flag')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if dense_flag:
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)
    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plt.show()





def cv_draw_landmark_mask(img_ori, pts, box=None, color=BLACK, thickness=1):
    image_landmarks = pts[:2].T
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

    #img = img_ori.copy()
    '''
    n = pts.shape[1]
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)
    '''
    return img, image_landmarks

def cv_draw_landmark(img_ori, pts, box=None, color=WHITE, size=4):
    img = img_ori.copy()
    n = pts.shape[1]
    # print(pts.shape)
    # mouth = np.array(np.round((pts[:2]).T[48:68]), dtype=np.int).T
    h, w, _ = img.shape

    # print(mouth.shape)
    # print(mouth)
    # cv2.polylines(img, tuple(np.array([v]) for v in (mouth,)),
    #             True, color, thickness=size, lineType=cv2.LINE_AA)
    # if n <= 106:
    #     for i in range(n):
    #         cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    # else:
    #     sep = 1
    #     for i in range(0, n, sep):
    #         cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)

    # box = box[0][:4]
    if box is not None:
        # left, top, right, bottom = np.round(box).astype(np.int32)
        # left_top = (left, top)
        # right_top = (right, top)
        # right_bottom = (right, bottom)
        # left_bottom = (left, bottom)


        rect = np.round(box[0][:4]).astype(np.int32)
        y1 = max(0, rect[1])
        y2 = min(img.shape[0], rect[3])
        x1 = max(0, rect[0])
        x2 = min(img.shape[1], rect[2])
        y_gap, x_gap = (y2 - y1)//6, (x2 - x1)//6
        img = img[y1-y_gap: y2+y_gap, x1-x_gap:x2+x_gap]
        mouth = ((pts[:2]).T[48:68])
        mouth = mouth - [x1-x_gap, y1-y_gap]
        h_, w_, _ = img.shape
        resize_h_ = 512/h_
        resize_w_ = 512/w_
        # # print(img.shape)
        
        img = cv2.resize(img, (512,512))
        img_draw = np.zeros((512,512,3), np.uint8)
        # img = np.zeros((w_, h_, 3), np.uint8)
        mouth = mouth.T
        mouth = np.array([mouth[0]*resize_w_, mouth[1]*resize_h_])
        mouth = np.array(np.round(mouth.T), dtype=np.int)
        # print(mouth.shape,)
        # print(mouth)
        # cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        # cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)
        # return img[box[0]:box[2], box[1]:box[3]]
        
        cv2.polylines(img_draw, tuple(np.array([v]) for v in (mouth,)),
                    True, color, thickness=size, lineType=cv2.LINE_AA)
        return cv2.resize(img, (512,512)), cv2.resize(img_draw, (512, 512))
    else:
        return img
