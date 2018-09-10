#!/usr/bin/env python
import sys
import os
import os.path
import json
import numpy as np
from detail import Detail, detaileval_kpt

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

    kpteval = detaileval_kpt.DetailEvalKpt(details)

    # submission file, change filename for different results
    submission_answer_file = os.path.join(submit_dir, "kpts_answer.json")

    with open(submission_answer_file) as f:
        data = json.load(f)['pred']

    kpteval.loadRes(data)
    print('*assembled*')
    kpteval.evaluate()
    print('*evaluated*')
    kpteval.accumulate()
    print('*accumulated*')

    ap = np.mean(kpteval.eval['precision'],axis=1)

    ap95 = ap[9,:,-1][0][0]
    ap75 = ap[5,:,-1][0][0]
    ap50 = ap[0,:,-1][0][0]

    print(ap95)
    print(ap75)
    print(ap50)

    # output scores
    output_file.write(("ap95:" + str(ap95) + "\n").encode())
    output_file.write(("ap75:" + str(ap75) + "\n").encode())
    output_file.write(("ap50:" + str(ap50)).encode())

    output_file.close()
