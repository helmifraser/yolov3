import sys

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except Exception as e:
    print("No need to remove ROS stuff from path")

import torch
import cv2
import numpy as np
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import random
import os.path as osp
import os
from datetime import datetime
import time


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection')
    parser.add_argument('-i', '--input', required=True, help='input image or directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('-o', '--outdir', default='detection', help='output directory, DEFAULT: detection/')
    parser.add_argument('-v', '--video', action='store_true', default=False, help='flag for detecting a video input')
    parser.add_argument('-w', '--webcam', action='store_true',  default=False, help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single webcam connected')
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False, help='do not show the detected video in real time')

    args = parser.parse_args()

    return args

def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    return batches

def draw_bbox(imgs, bbox, colors, classes):
    img = imgs[int(bbox[0])]
    label = classes[int(bbox[-1])]
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())
    color = random.choice(colors)
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)

def parse_detection(detection):
    # Returns detection data in the form: [[class id, x1, y1, x2, y2, objectness score, prediction score], ... ]
    parsed = detection.detach().numpy()
    np.rint(parsed[:, :5], out=parsed[:, :5])
    parsed = np.roll(parsed[:, 1:], 1, axis=1)[:, :7]
    return parsed

def write_detection(file, parsed_detections, frame_number, filter=[None, None]):
    shape = parsed_detections.shape
    detection_string = ""
    bad_first = False
    for idx, detection in enumerate(parsed_detections):
        # Filter out car dash, don't care about it
        if filter[0] is not None:
            if ((detection[3] - detection[1]) >= int(filter[0]*0.8)):
                print("Filtered out {} at frame {}".format(detection[0], frame_number))
                if idx == 0:
                    bad_first = True
                continue

        data = "{}, {}, {}, {}, {}, {}, {}, ".format(*detection)

        if idx == 0 or (bad_first and len(detection_string) == 0):
            data = "{}, ".format(frame_number) + data

        if idx == shape[0] - 1:
            data = data[:-2] + "\n"

        detection_string += data

    file.write(detection_string)


def save_detections(file=None, detections=None, frame_number=0, filter=[None, None]):
    if file is None:
        print("No file specified!")
        return False

    try:
        parsed = parse_detection(detections)
        write_detection(file, parsed,frame_number, filter)
    except Exception as e:
        print("Error: {}".format(e))
        return False

    return True

def detect_video(model, args):

    input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")
    colors = [colors[1]]
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        window = cv2.namedWindow(args.input, cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    read_frames = 0

    start_time = datetime.now()

    outpath = os.path.dirname(args.input) + '/' + osp.basename(args.input).rsplit('.')[0] + '.csv'
    bounding_boxes_file = open(outpath,'w+')

    print('Detecting...')
    while cap.isOpened():
        total_tic = time.time()
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame = cv2.resize(frame, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh)
            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)

                saved = save_detections(bounding_boxes_file, detections, read_frames, [width*0.2, height*0.2])

                if not saved:
                    print("Something went wrong saving boundary boxes to file :(")
                    sys.exit(1)

                for detection in detections:
                    draw_bbox([frame], detection, colors, classes)

            if not args.no_show:
                cv2.imshow(args.input, frame)

            if read_frames % 120 == 0:
                total_toc = time.time()
                total_time = total_toc - total_tic
                frame_rate = 1 / total_time
                print('Frame rate:', frame_rate)
                print('Number of frames processed:', read_frames)
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('p'):
                print("Saving frame and bbox")
                cv2.imwrite("frame.png", frame)
                bounding_boxes = open('./bbox_' + str(read_frames) + '.csv','a+')
                for i, detection in enumerate(detections):
                    print("Writing detection {} to file".format(i))
                    bounding_boxes.write(str(detection) + '\n')
                bounding_boxes.close()
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    bounding_boxes_file.close()

    if not args.no_show:
        cv2.destroyAllWindows()

    return


def detect_image(model, args):

    print('Loading input image(s)...')
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    batch_size = int(model.net_info['batch'])

    imlist, imgs = load_images(args.input)
    print('Input image(s) loaded')

    img_batches = create_batches(imgs, batch_size)

    # load colors and classes
    colors = pkl.load(open("pallete", "rb"))
    classes = load_classes("data/coco.names")

    if not osp.exists(args.outdir):
        os.makedirs(args.outdir)

    start_time = datetime.now()
    print('Detecting...')
    for batchi, img_batch in enumerate(img_batches):
        img_tensors = [cv_image2tensor(img, input_size) for img in img_batch]
        img_tensors = torch.stack(img_tensors)
        img_tensors = Variable(img_tensors)
        if args.cuda:
            img_tensors = img_tensors.cuda()
        detections = model(img_tensors, args.cuda).cpu()
        detections = process_result(detections, args.obj_thresh, args.nms_thresh)
        if len(detections) == 0:
            continue

        detections = transform_result(detections, img_batch, input_size)
        for detection in detections:
            draw_bbox(img_batch, detection, colors, classes)

        for i, img in enumerate(img_batch):
            save_path = osp.join(args.outdir, 'det_' + osp.basename(imlist[batchi*batch_size + i]))
            cv2.imwrite(save_path, img)

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))

    return

def main():

    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if not os.path.exists(args.input):
        print("Video file doesn't exist at: {}".format(args.input))
        sys.exit(1)


    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    if args.video:
        detect_video(model, args)

    else:
        detect_image(model, args)



if __name__ == '__main__':
    main()
