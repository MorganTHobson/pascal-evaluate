#!/usr/bin/env python
import sys
import os
import os.path
import json
from detail import Detail, detaileval_cls
from importlib import reload

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print(str(submit_dir) + " doesn't exist")

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')

    truth_file = os.path.join(truth_dir, "truth.json")

    details = Detail(truth_file, '', 'trainval')

    reload(detaileval_cls)
    detaileval = detaileval_cls.DetailEvalCls(details)

    submission_answer_file = os.path.join(submit_dir, "answer_cls.json")

    detailRes = json.load(open(submission_answer_file, 'r'))['pred']

    detaileval.loadRes(detailRes)
    ap = detaileval.evaluate()

    output_file.write(("ap:" + str(ap)).encode())

    output_file.close()
