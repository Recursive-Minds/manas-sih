import xml.etree.ElementTree as ET

tree = ET.parse('ui_dump.xml')
for node in tree.iter('node'):
    text = node.attrib.get('text', '')
    res_id = node.attrib.get('resource-id', '')
    bounds = node.attrib.get('bounds', '')
    if any(k in text.lower() or k in res_id.lower() for k in ['dismiss', 'benchmark', 'start', 'stop', 'reset', 'spinner', 'speed', 'btn']):
        print(f'{res_id} | "{text}" | {bounds}')
