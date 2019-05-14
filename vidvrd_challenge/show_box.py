

def show_boxes(im_path, dets, cls):
    from matplotlib import pyplot as plt
    """Draw detected bounding boxes."""
    im = plt.imread(im_path)
    for i in range(0, len(dets)):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2],
                          bbox[3], fill=False,
                          edgecolor='red', linewidth=1.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{}'.format(cls[i]),
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
    clses = []
    for obj in objs:
        clses.append(obj.find('name').text)
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append([xmin, ymin, xmax-xmin+1, ymax-ymin+1])
    return boxes, clses


if __name__ == '__main__':
    im_path = '/home/magus/dataset3/VidOR/vidor-ilsvrc/Data/VID/train/0000/2401075277/000000.JPEG'
    xml_path = '/home/magus/dataset3/VidOR/vidor-ilsvrc/Annotations/VID/train/0000/2401075277/000000.xml'
    dets, clss = get_box_cls(xml_path)
    show_boxes(im_path, dets, clss)
