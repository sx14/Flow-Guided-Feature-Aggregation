import random
import matplotlib.pyplot as plt


def get_colors(n_color=100):
    colors = []
    for i in range(n_color):
        r = random.randint(0, 255) / 255.0
        g = random.randint(0, 255) / 255.0
        b = random.randint(0, 255) / 255.0
        colors.append([r,g,b])
    return colors


def show_boxes(im_path, dets, cls, colors):
    """Draw detected bounding boxes."""
    im = plt.imread(im_path)
    plt.imshow(im, aspect='equal')

    for i in range(0, len(dets)):

        bbox = dets[i]

        if bbox is None:
            continue

        color = colors[i]
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2],
                             bbox[3], fill=False,
                             edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        plt.text(bbox[0], bbox[1] - 2,
                '%s' % (cls[i]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_box_cls(xml_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    objs = tree.findall('object')

    boxes = []
    tids = []
    clses = []
    for obj in objs:

        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append([xmin, ymin, xmax-xmin+1, ymax-ymin+1])

        tids.append(int(obj.find('trackid').text))
        clses.append(obj.find('name').text)
    return boxes, clses, tids


if __name__ == '__main__':
    im_path = '/media/sunx/Data/linux-workspace/python-workspace/Flow-Guided-Feature-Aggregation/data/VidOR/Data/VID/val/0001/2793806282/000000.JPEG'
    xml_path = '/media/sunx/Data/linux-workspace/python-workspace/Flow-Guided-Feature-Aggregation/data/VidOR/Annotations/VID/val/0001/2793806282/000000.xml'
    dets, clss, tids = get_box_cls(xml_path)
    all_colors = get_colors()
    colors = [all_colors[tid] for tid in tids]
    show_boxes(im_path, dets, clss, colors)

