#!/usr/bin/env python
import sys
import os
import os.path
import json
from detail import Detail
from detail.detaileval_segm import catSegEvalClass
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

    detaileval = catSegEvalClass(details)

    submission_answer_file = os.path.join(submit_dir, "answer_cat.json")

    detaileval.loadJSON(submission_answer_file)
    detaileval.evaluate()
    ap = detaileval.results

    output_file.write(("ap:" + str(ap)).encode())

    output_file.close()
