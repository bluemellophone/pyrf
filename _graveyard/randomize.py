import os
import random
from collections import defaultdict
import xml.etree.ElementTree as xml


# want to map a the filename -> all the animals present in the image
def parse_annotations(direct):
    all_files = [f for f in os.listdir(direct) if os.path.isfile(os.path.join(direct, f)) and f.lower().endswith('.xml')]
    filenames = defaultdict(list)
    for f in all_files:
        target_file = os.path.join(direct, f)
        # check that the annotation's xml file exists
        if os.path.isfile(target_file):
            # print 'parsing %s' % target_file
            with open(target_file, 'r') as xml_file:
                # get the raw xml file from the annotation file
                raw_xml = xml_file.read().replace('\n', '')
                # read it into an Element
                data_xml = xml.XML(raw_xml)
                # get all instances of filename, there should only be one!
                filename_xml = [f for f in data_xml.findall('filename')]
                if len(filename_xml) > 1:
                    print 'problem with %s, more than one filename!' % target_file
                fname = filename_xml[0]
                filenames[fname.text[0:-4]] = []
                # get all bounding boxes in this annotation
                for obj in data_xml.findall('object'):
                    # get the animals present in this image, don't want the file extension
                    for classname in obj.findall('name'):
                        filenames[fname.text[0:-4]].append(classname.text)

        else:
            print 'could not find %s, ignoring' % target_file

    return filenames

if __name__ == '__main__':
    # the ratio of data to be set aside for training
    test_ratio = 0.2
    val_ratio = 0.0

    classnames = ['elephant', 'giraffe', 'rhino', 'wilddog', 'zebra_grevys', 'zebra_plains']

    for classname in classnames:
        print "Parsing:", classname
        # class that will be marked as positive training examples
        positives = [classname]
        # directory that contains the xml annotations
        xml_dir = 'IBEIS2014/Annotations'
        out_dir = 'IBEIS2014/ImageSets/Main'
        annotations = parse_annotations(xml_dir)

        keys = sorted(annotations.keys())
        
        # open the files to write the assignments to
        with open(os.path.join(out_dir, classname + '_train.txt'), 'w') as train, \
             open(os.path.join(out_dir, classname + '_trainval.txt'), 'w') as trainval, \
             open(os.path.join(out_dir, classname + '_val.txt'), 'w') as val, \
             open(os.path.join(out_dir, classname + '_test.txt'), 'w') as test:
            for i, filename in enumerate(keys):
                pos = False
                for _pos in positives:
                    if _pos in annotations[filename]:
                        pos = True
                        break

                line = filename + ' ' + (' 1' if pos else '-1') + '\n'

                bucket = random.uniform(0.0, 1.0)
                if bucket < test_ratio:
                    test.write(line)
                elif test_ratio <= bucket < test_ratio + val_ratio:
                    trainval.write(line)
                    val.write(line)
                else:
                    trainval.write(line)
                    train.write(line)
                