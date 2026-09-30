# Reverse the Conv+Swish fusion to get back the original model
# The fused model has Convolution with 9=6 (SiLU activation) instead of Conv + Swish layers
# We need to split these back to Conv + Swish

import re

param_path = 'app/src/main/assets/best.param'

with open(param_path, 'r') as f:
    lines = f.readlines()

magic = lines[0].strip()
header = lines[1].strip().split()
orig_layers_count = int(header[0])
orig_blobs_count = int(header[1])

# Parse all layer lines
layers = []
for line in lines[2:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    layer_type = parts[0]
    layer_name = parts[1]
    input_count = int(parts[2])
    output_count = int(parts[3])
    
    idx = 4
    inputs = parts[idx:idx + input_count]
    idx += input_count
    outputs = parts[idx:idx + output_count]
    idx += output_count
    params = parts[idx:]
    
    layers.append({
        'type': layer_type,
        'name': layer_name,
        'input_count': input_count,
        'output_count': output_count,
        'inputs': inputs,
        'outputs': outputs,
        'params': params,
        'line': line
    })

# Find Conv layers with 9=6 (fused SiLU activation) and split them
new_layers = []
next_blob_id = 10000  # Use high numbers to avoid collision
silu_counter = 0

for layer in layers:
    new_layers.append(layer)
    
    if layer['type'] == 'Convolution':
        # Check if 9=6 is in params (fused SiLU)
        has_silu = False
        new_params = []
        for p in layer['params']:
            if p == '9=6':
                has_silu = True
            else:
                new_params.append(p)
        
        if has_silu:
            # Remove the 9=6 parameter
            layer['params'] = new_params
            
            # Create an intermediate blob
            inter_blob = str(next_blob_id)
            next_blob_id += 1
            
            # The Conv's output becomes the intermediate blob
            orig_output = layer['outputs'][0]
            layer['outputs'] = [inter_blob]
            layer['output_count'] = 1
            
            # Create a Swish layer that takes inter_blob and produces orig_output
            silu_layer = {
                'type': 'Swish',
                'name': f'silu_{silu_counter}',
                'input_count': 1,
                'output_count': 1,
                'inputs': [inter_blob],
                'outputs': [orig_output],
                'params': []
            }
            new_layers.append(silu_layer)
            silu_counter += 1

# Recalculate blob count
blob_set = set()
for layer in new_layers:
    for b in layer['inputs']:
        blob_set.add(b)
    for b in layer['outputs']:
        blob_set.add(b)
new_blobs_count = len(blob_set)
new_layers_count = len(new_layers)

print(f'Original: {orig_layers_count} layers, {orig_blobs_count} blobs')
print(f'Reversed: {new_layers_count} layers, {new_blobs_count} blobs')
print(f'Restored {silu_counter} Swish layers')

# Write the reversed param file
with open(param_path, 'w') as f:
    f.write(f'{magic}\n')
    f.write(f'{new_layers_count} {new_blobs_count}\n')
    for layer in new_layers:
        parts = [layer['type'], layer['name'],
                 str(layer['input_count']), str(layer['output_count'])]
        parts.extend(layer['inputs'])
        parts.extend(layer['outputs'])
        parts.extend(layer['params'])
        f.write(' '.join(parts) + '\n')

print(f'Written to {param_path}')
