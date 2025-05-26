areas = []
with open('./configs/areas.txt', 'r', encoding='utf-8') as f:
    for line in f:
        coords = eval(line.strip().rstrip(','))
        areas.append(coords)

first_point = areas[0]
print(first_point)  # 예: (100, 200)