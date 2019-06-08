import os
import shutil
import xml.dom.minidom


def output_ilsvrc_vid_format(mid_data, output_path):
    # mid_data:
    # folder
    # filename
    # database
    # width
    # height
    # objects
    #   -- trackid
    #   -- xmin
    #   -- ymin
    #   -- xmax
    #   -- ymax
    #   -- name
    #   -- generated
    #   -- tracker


    des_xml_dom = xml.dom.minidom.Document()
    # annotation
    des_root_node = des_xml_dom.createElement('annotation')
    # folder
    des_folder_node = des_xml_dom.createElement('folder')
    des_folder = des_xml_dom.createTextNode(mid_data['folder'])
    des_folder_node.appendChild(des_folder)
    des_root_node.appendChild(des_folder_node)
    # filename
    des_filename_node = des_xml_dom.createElement('filename')
    des_filename = des_xml_dom.createTextNode(mid_data['filename'])
    des_filename_node.appendChild(des_filename)
    des_root_node.appendChild(des_filename_node)

    # source
    des_dataset_name = des_xml_dom.createTextNode(mid_data['database'])
    des_dataset_node = des_xml_dom.createElement('database')
    des_dataset_node.appendChild(des_dataset_name)
    des_source_node = des_xml_dom.createElement('source')
    des_source_node.appendChild(des_dataset_node)
    des_root_node.appendChild(des_source_node)

    # size
    des_size_node = des_xml_dom.createElement('size')
    des_width_node = des_xml_dom.createElement('width')
    des_height_node = des_xml_dom.createElement('height')
    des_width = des_xml_dom.createTextNode(str(mid_data['width']))
    des_height = des_xml_dom.createTextNode(str(mid_data['height']))
    des_width_node.appendChild(des_width)
    des_height_node.appendChild(des_height)
    des_size_node.appendChild(des_width_node)
    des_size_node.appendChild(des_height_node)
    des_root_node.appendChild(des_size_node)

    # object
    org_objects = mid_data['objects']
    for j in range(0, len(org_objects)):
        org_object = org_objects[j]
        des_object_node = des_xml_dom.createElement('object')
        x_min = int(org_object['xmin'])
        y_min = int(org_object['ymin'])
        x_max = int(org_object['xmax'])
        y_max = int(org_object['ymax'])
        if x_min < 0:
            org_object['xmin'] = '0'
        if y_min < 0:
            org_object['ymin'] = '0'
        if x_max >= int(mid_data['width']):
            org_object['xmax'] = str(int(mid_data['width']) - 1)
        if y_max >= int(mid_data['height']):
            org_object['ymax'] = str(int(mid_data['height']) - 1)

        # track id
        des_track_id = des_xml_dom.createTextNode(str(org_object['trackid']))
        des_track_id_node = des_xml_dom.createElement('trackid')
        des_track_id_node.appendChild(des_track_id)
        des_object_node.appendChild(des_track_id_node)

        # name
        des_object_name = des_xml_dom.createTextNode(org_object['name'])
        des_object_name_node = des_xml_dom.createElement('name')
        des_object_name_node.appendChild(des_object_name)
        des_object_node.appendChild(des_object_name_node)

        # bndbox
        des_xmax_node = des_xml_dom.createElement('xmax')
        des_xmax = des_xml_dom.createTextNode(str(org_object['xmax']))
        des_xmax_node.appendChild(des_xmax)
        des_xmin_node = des_xml_dom.createElement('xmin')
        des_xmin = des_xml_dom.createTextNode(str(org_object['xmin']))
        des_xmin_node.appendChild(des_xmin)
        des_ymax_node = des_xml_dom.createElement('ymax')
        des_ymax = des_xml_dom.createTextNode(str(org_object['ymax']))
        des_ymax_node.appendChild(des_ymax)
        des_ymin_node = des_xml_dom.createElement('ymin')
        des_ymin = des_xml_dom.createTextNode(str(org_object['ymin']))
        des_ymin_node.appendChild(des_ymin)
        des_object_box_node = des_xml_dom.createElement('bndbox')
        des_object_box_node.appendChild(des_xmax_node)
        des_object_box_node.appendChild(des_xmin_node)
        des_object_box_node.appendChild(des_ymax_node)
        des_object_box_node.appendChild(des_ymin_node)
        des_object_node.appendChild(des_object_box_node)


        # occluded
        des_occluded = des_xml_dom.createTextNode('0')
        des_occluded_node = des_xml_dom.createElement('occluded')
        des_occluded_node.appendChild(des_occluded)
        des_object_node.appendChild(des_occluded_node)

        # generated
        des_generated = des_xml_dom.createTextNode(str(org_object['generated']))
        des_generated_node = des_xml_dom.createElement('generated')
        des_generated_node.appendChild(des_generated)
        des_object_node.appendChild(des_generated_node)

        # tracker
        des_tracker = des_xml_dom.createTextNode(org_object['tracker'])
        des_tracker_node = des_xml_dom.createElement('tracker')
        des_tracker_node.appendChild(des_tracker)
        des_object_node.appendChild(des_tracker_node)

        des_root_node.appendChild(des_object_node)

    with open(output_path, 'w') as des_file:
        des_root_node.writexml(des_file, addindent='\t', newl='\n')