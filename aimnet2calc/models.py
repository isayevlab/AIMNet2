import os
import requests

# model registry aliases
model_registry_aliases = {}
model_registry_aliases['aimnet2'] = 'aimnet2/aimnet2_wb97m_0'
model_registry_aliases['aimnet2_wb97m'] = model_registry_aliases['aimnet2']
model_registry_aliases['aimnet2_b973c'] = 'aimnet2/aimnet2_b973c_0'
model_registry_aliases['aimnet2-qr'] = 'aimnet2-qr/aimnet2-qr_b97md4_qzvp_2'


def get_model_path(s: str):
    # direct file path
    if os.path.isfile(s):
        print('Found model file:', s)
        return s
    # check aliases
    if s in model_registry_aliases:
        s = model_registry_aliases[s]
    # add jpt extension
    if not s.endswith('.jpt'):
        s = s + '.jpt'
    sdir = os.path.dirname(s)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'assets', sdir), exist_ok=True)
    s_local = os.path.join(os.path.dirname(__file__), 'assets', s)
    if os.path.isfile(s_local):
        print('Found model file:', s_local)
    else:
        url = f'https://github.com/zubatyuk/aimnet-model-zoo/raw/main/{s}'
        print('Downloading model file from', url)
        r = requests.get(url)
        r.raise_for_status()
        with open(s_local, 'wb') as f:
            f.write(r.content)
        print('Saved to ', s_local)
    return s_local
