from os.path import join, exists
import cv2
import random
from pyrf import Random_Forest_Detector
from detecttools.directory import Directory


def _draw_box(img, annotation, xmin, ymin, xmax, ymax, color, stroke=2, top=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    width, height = cv2.getTextSize(annotation, font, scale, -1)[0]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, stroke)
    if top:
        cv2.rectangle(img, (xmin, ymin - height), (xmin + width, ymin), color, -1)
        cv2.putText(img, annotation, (xmin + 5, ymin), font, 0.4, (255, 255, 255))
    else:
        cv2.rectangle(img, (xmin, ymax - height), (xmin + width, ymax), color, -1)
        cv2.putText(img, annotation, (xmin + 5, ymax), font, 0.4, (255, 255, 255))


def openImage(filename, color=False, alpha=False):
    if not exists(filename):
        return None

    if not color:
        mode = 0   # Greyscale by default
    elif not alpha:
        mode = 1   # Color without alpha channel
    else:
        mode = -1  # Color with alpha channel

    return cv2.imread(filename, mode)


def randInt(lower, upper):
    return random.randint(lower, upper)


def randColor():
    return [randInt(50, 205), randInt(50, 205), randInt(50, 205)]


train_pos_direct = Directory(join('train', 'pos'))
train_neg_direct = Directory(join('train', 'neg'))
test_direct = Directory('test')

train_pos_gpath_list = train_pos_direct.files()
train_neg_gpath_list = train_pos_direct.files()
test_gpath_list = test_direct.files()
tree_path = 'trees'
# tree_path = '/Users/bluemellophone/Library/Application Support/ibeis/detectmodels/rf/zebra_grevys/'
zebras_path = 'trees-zebras'
# zebras_path = '/Users/bluemellophone/Library/Application Support/ibeis/detectmodels/rf/zebra_grevys/'

detector = Random_Forest_Detector()
# detector.train(train_pos_gpath_list, train_neg_gpath_list, tree_path)

# trees = Directory(tree_path, include_file_extensions=['txt'])
# forest = detector.forest(trees.files())
# results = detector.detect(forest, test_gpath_list)
# for result in results:
#     print 'RESULT: %r' % (result, )

test_gpath_list = test_gpath_list[:10]
output_list = [ 'output/%d.JPEG' % (i) for i in range(len(test_gpath_list))]
trees = Directory(zebras_path, include_file_extensions=['txt'])
forest = detector.forest(trees.files())
# results_iter = detector.detect(forest, test_gpath_list, output_gpath_list=output_list, mode=1, sensitivity=0.75)
results_iter = detector.detect(forest, test_gpath_list, output_gpath_list=output_list)
for input_gpath, result_list in results_iter:
    print result_list
    # original = openImage(input_gpath, color=True)
    # for result in result_list:
    #     color = randColor()
    #     print result
    #     _draw_box(original, '', result['xtl'], result['ytl'], result['xtl'] + result['width'], result['ytl'] + result['height'], color)
    #     cv2.circle(original, (result['centerx'], result['centery']), 3, color, -1)

    # cv2.imshow('IMG', original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
