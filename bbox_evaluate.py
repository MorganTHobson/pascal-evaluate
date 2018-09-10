#!/usr/bin/env python
import sys
import os
import os.path
import json
from detail import Detail, bboxEval

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print(str(submit_dir) + " doesn't exist")

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output file
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')

    # trainval file
    truth_file = os.path.join(truth_dir, "truth.json")

    details = Detail(truth_file, '', 'trainval')

    bboxeval = bboxEval.bboxEval(details)

    # submission file, change filename for different results
    submission_answer_file = os.path.join(submit_dir, "full.json")

    li = []  # list of image ids
    lc = []  # list of category ids

    with open(truth_file) as f:
        data = json.load(f)

    # extract image and cat ids from trainval
    for object in data['annos_segmentation']:
        img = int(object['image_id'])
        cat = int(object['category_id'])
        if img not in li:
            li.append(img)
        if cat not in lc:
            lc.append(cat)

    bboxeval.params.imgIds = li
    bboxeval.params.catIds = lc

    bboxeval.loadBoxRes(resFile=submission_answer_file)
    print('*assembled*')
    bboxeval.evaluate()
    print('*evaluated*')
    bboxeval.accumulate()
    print('*accumulated*')

    ap95 = bboxeval.ap[9,:,-1]
    ap75 = bboxeval.ap[5,:,-1]
    ap50 = bboxeval.ap[0,:,-1]

    # account for -1 offset
    valid_ind = [0, 3, 5, 10, 11, 12, 18, 20, 21, 25, 27, 29, 32, 35, 40, 42, 45, 50, 56, 57]
    for i,v in enumerate(ap95):
        if i not in valid_ind:
            ap95[i] = ap95[i] + 1
            ap75[i] = ap75[i] + 1
            ap50[i] = ap50[i] + 1

    avg_95 = str(sum(ap95)/len(ap95))
    avg_75 = str(sum(ap75)/len(ap75))
    avg_50 = str(sum(ap50)/len(ap50))

    print(avg_95)
    print(avg_75)
    print(avg_50)

    # output scores
    output_file.write(("ap95:" + avg_95 + "\n").encode())
    output_file.write(("ap75:" + avg_75 + "\n").encode())
    output_file.write(("ap50:" + avg_50).encode())

    output_file.close()
