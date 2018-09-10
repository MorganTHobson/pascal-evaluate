#!/usr/bin/env python
import sys
import os
import os.path
import json
from detail import Detail, instsegEval

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

    instsegeval = instsegEval.instsegEval(details)

    # submission file, change filename for different results
    submission_answer_file = os.path.join(submit_dir, "instseg_answer.json")

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

    instsegeval.params.imgIds = li
    instsegeval.params.catIds = lc

    instsegeval.loadSegRes(resFile=submission_answer_file)
    print('*assembled*')
    instsegeval.evaluate()
    print('*evaluated*')
    instsegeval.accumulate()
    print('*accumulated*')

    ap95 = instsegeval.ap[9,:,-1]
    ap75 = instsegeval.ap[5,:,-1]
    ap50 = instsegeval.ap[0,:,-1]

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
