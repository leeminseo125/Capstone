import yaml

def load_regions(region_yaml_path):
    with open(region_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['regions']

def box_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def is_inside_region(point, region):
    x, y = point
    x_min, y_min, x_max, y_max = region
    return x_min <= x <= x_max and y_min <= y <= y_max

def count_heads_in_regions(bboxes, image_shape, region_yaml_path):
    """
    Args:
        bboxes: list of (x1, y1, x2, y2) head detection results
        image_shape: (H, W, C) of the input image
        region_yaml_path: path to yaml file defining ROI regions

    Returns:
        List of counts per region
    """
    regions = load_regions(region_yaml_path)  # list of [x1, y1, x2, y2]
    counts = [0] * len(regions)

    for box in bboxes:
        center = box_center(*box)
        for i, region in enumerate(regions):
            if is_inside_region(center, region):
                counts[i] += 1
                break  # 하나의 head는 하나의 region에만 속한다고 가정

    return counts