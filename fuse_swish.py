import re

param_path = 'app/src/main/assets/best.param'

with open(param_path, 'r') as f:
    lines = f.readlines()

magic = lines[0].strip()
total_layers, total_blobs = map(int, lines[1].split())

layers = []
for line in lines[2:]:
    line = line.rstrip()
    if not line:
        continue
    parts = line.split()
    layer_type = parts[0]
    layer_name = parts[1]
    input_count = int(parts[2])
    output_count = int(parts[3])
    input_blobs = parts[4:4+input_count]
    output_blobs = parts[4+input_count:4+input_count+output_count]
    params = parts[4+input_count+output_count:]
    layers.append({
        'type': layer_type,
        'name': layer_name,
        'inputs': input_blobs,
        'outputs': output_blobs,
        'params': params,
    })

conv_output_map = {}  # conv_output_blob -> conv_index
for i, layer in enumerate(layers):
    if layer['type'] == 'Convolution':
        for ob in layer['outputs']:
            conv_output_map[ob] = i

swish_indices = set()
conv_indices = set()
fusions = []

for i, layer in enumerate(layers):
    if layer['type'] != 'Swish':
        continue
    if len(layer['inputs']) != 1:
        continue
    swish_in = layer['inputs'][0]
    if swish_in not in conv_output_map:
        continue
    conv_idx = conv_output_map[swish_in]
    # Check no other layer uses conv's output blob
    shared = False
    for j, other in enumerate(layers):
        if j == conv_idx or j == i:
            continue
        if swish_in in other['inputs']:
            shared = True
            break
    if not shared:
        fusions.append((conv_idx, i))
        swish_indices.add(i)
        conv_indices.add(conv_idx)

print(f"Found {len(fusions)} Conv+Swish fusions")

for conv_idx, swish_idx in fusions:
    conv = layers[conv_idx]
    swish = layers[swish_idx]
    conv['outputs'] = swish['outputs']
    params_str = ' '.join(conv['params'])
    if '9=' not in params_str:
        params_str = (params_str + ' 9=6').strip()
    conv['params'] = params_str.split()
    # The Swish layer is now skipped

new_layers = [l for i, l in enumerate(layers) if i not in swish_indices]
new_total = len(new_layers)

lines_out = [magic + '\n', f'{new_total} {total_blobs - len(swish_indices)}\n']
for layer in new_layers:
    parts = [layer['type'], layer['name'],
             str(len(layer['inputs'])), str(len(layer['outputs']))]
    parts.extend(layer['inputs'])
    parts.extend(layer['outputs'])
    parts.extend(layer['params'])
    lines_out.append(' '.join(parts) + '\n')

with open(param_path, 'w') as f:
    f.writelines(lines_out)

print(f"Written {new_total} layers")
