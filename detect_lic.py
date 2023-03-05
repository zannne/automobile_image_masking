import argparse
import os
import shutil
import time
from pathlib import Path
import cv2
import sys

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-1])
sys.path.append(os.path.join(root_path, "licence"))

import torch.backends.cudnn as cudnn
import torch

from licence.utils.torch_utils import select_device, time_synchronized
from licence.models.experimental import attempt_load
from licence.utils.utils import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from licence.utils.datasets import LoadStreams, LoadImages

from add_utils import plot_one_box



def detect_lic(save_img=True):
    out, source, det_weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.det_weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder

    # Load yolov5 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
        # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for de in det:
                    # xyxy,conf,cls,lic_plat=de[:4],de[4],de[5],de[6:]
                    *xyxy, conf, cls=de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        im0 = plot_one_box(xyxy, im0, color=(0, 0, 255), line_thickness=3)

            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # rec_result video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if sys.platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
    parser.add_argument('--det-weights', nargs='+', type=str, default='./licence/weights/yolov5_best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./licence/demo/images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='demo/rec_result', help='rec_result folder')  # rec_result folder
    parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented demo')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect_lic()
