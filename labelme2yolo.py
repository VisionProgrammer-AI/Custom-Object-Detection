import json
import os
import cv2


def convert_labelme_to_yolo(json_file, output_dir, label_mapping):
    # Load the LabelMe JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)


    image_path = data.get('imagePath')

    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(json_file), image_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    height, width, _ = image.shape

    yolo_lines = []
    for shape in data.get('shapes', []):
        label = shape['label']

        class_index = label_mapping.get(label, -1)
        if class_index == -1:

            continue


        points = shape['points']
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)


        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin


        center_x /= width
        center_y /= height
        bbox_width /= width
        bbox_height /= height

        yolo_lines.append(f"{class_index} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    base_name = os.path.splitext(os.path.basename(json_file))[0]
    txt_file = os.path.join(output_dir, base_name + ".txt")


    with open(txt_file, 'w') as f:
        for line in yolo_lines:
            f.write(line + "\n")
    print(f"Converted {json_file} to {txt_file}")


def process_folder(input_dir, output_dir, label_mapping):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            json_file = os.path.join(input_dir, filename)
            try:
                convert_labelme_to_yolo(json_file, output_dir, label_mapping)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")


if __name__ == '__main__':

    label_mapping = {
        "Yarn Cone": 0,

    }


    input_dir = 'labelme_json_dir'
    output_dir = 'YOLOtxt'

    process_folder(input_dir, output_dir, label_mapping)
