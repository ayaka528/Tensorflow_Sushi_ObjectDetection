import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        base = os.path.basename(xml_file)
        realname = os.path.splitext(base)[0]
        print(realname)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bb = member.findall('bndbox')[0]
            value = (realname,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(bb.findall('xmin')[0].text),
                     int(bb.findall('ymin')[0].text),
                     int(bb.findall('xmax')[0].text),
                     int(bb.findall('ymax')[0].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'Annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('sushi_labels.csv', index=None)
    print('Successfully converted xml to csv.')


main()
