import json
import os

with open('truth.json') as f:
    data = json.load(f)

output = open('answer.json', 'wb')
output.write(('[').encode())


for object in data['annos_segmentation']:
    output.write(('{').encode())
    output.write(('\"image_id\":' + str(object['image_id']) + ',').encode())
    output.write(('\"category_id\":' \
                  + str(object['category_id']) + ',').encode())
    output.write(('\"bbox\":' + str(object['bbox']) + ',').encode())
    output.write(('\"score\":1').encode())
    output.write(('},\n').encode())

output.seek(-2,os.SEEK_END)
output.truncate

output.write((']').encode())
output.close()
