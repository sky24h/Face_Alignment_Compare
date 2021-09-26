import numpy as np
import cv2, os, sys, glob
import face_alignment
from tqdm import tqdm
from natsort import natsorted
device = "cuda"
import torch


def get_landmarks(images, detector=None):
    images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])
    images = images.transpose(0, 3, 1, 2)
    face_det_batch_size = 16
    if detector is None:
        detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                flip_input=False, device=device)

    batch_size = face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
                predictions.extend(detector.get_landmarks_from_batch(torch.Tensor(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    #boxes = np.array(results)
    # if not nosmooth:
    #     boxes = get_smoothened_boxes(boxes, T=5)
    # results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)]
    #            for image, (x1, y1, x2, y2) in zip(images, boxes)]
    if detector is not None:
        del detector
    return np.array(predictions)
    #return boxes

# if __name__ == '__main__':
#     # これを実行する。
#     # dirs = glob.glob("/mnt/NAS4/98_personal/kou/Wav2Lip/dataset_0226/*")
#     dirs = glob.glob("../../../datasets/joint_datasets_0811/*")
#     # dirs = glob.glob("/mnt/NAS4/98_personal/kou/Automatic_Training/kiriyama_0913_for_vid2vid/raw_frames_audio_10/*")
#     dirs.sort()
#     n = 0
#     for dir in tqdm(dirs):
#         if False:#int(os.path.basename(dir)) < 168 or int(os.path.basename(dir)) > 1500:
#             continue
#         else:
#             pass
#             # save_dir = "./dataset_0226_processed/" + os.path.basename(dir) + "/"
#             #if os.path.exists(save_dir):
#             #    continue
#             # os.makedirs(save_dir, exist_ok=True)
#             all_img_paths = glob.glob(dir+"/*.jpg") + glob.glob(dir+"/*.png")
#             imgs = [cv2.imread(path) for path in natsorted(all_img_paths)]

#             # results = face_detect_rect(imgs)
#             results = face_detect_blinks(imgs)
#             try:
#                 if results.shape[0] == len(all_img_paths):
#                     np.save(dir+'/landmarks', results)
#                 else:
#                     print('!!!!!!!')#break_except
#             except:
#                 print('failed at', dir)
#             '''
#             for i, res in enumerate(results):
#                 #print(i, res.shape)
#                 try:
#                     if res.shape[0] != 0 and res.shape[1] != 0:
#                         np.save(dir+'/landmarks', res)
#                 except:
#                     print(res, dir)
#             '''
