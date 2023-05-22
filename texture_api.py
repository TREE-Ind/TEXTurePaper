from flask import Flask, request, jsonify
import datetime
import pathlib
import shlex
import subprocess
import sys
from typing import Optional

import trimesh

#sys.path.append('TEXTurePaper')

from src.configs.train_config import GuideConfig, LogConfig, TrainConfig
from src.training.trainer import TEXTure

app = Flask(__name__)
max_num_faces = 100000

def load_config(shape_path: str, text: str, seed: int, guidance_scale: float) -> TrainConfig:
    text += ', {} view'
    log = LogConfig(exp_name=gen_exp_name())
    guide = GuideConfig(text=text)
    guide.background_img = 'TEXTurePaper/textures/brick_wall.png'
    guide.shape_path = 'TEXTurePaper/shapes/spot_triangulated.obj'
    config = TrainConfig(log=log, guide=guide)
    config.guide.shape_path = shape_path
    config.optim.seed = seed
    config.guide.guidance_scale = guidance_scale
    return config

def gen_exp_name() -> str:
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')

def check_num_faces(path: str) -> bool:
    with open(path) as f:
        lines = [line for line in f.readlines() if line.startswith('f')]
    return len(lines) <= max_num_faces

def zip_results(exp_dir: pathlib.Path) -> str:
    mesh_dir = exp_dir / 'mesh'
    out_path = f'{exp_dir.name}.zip'
    subprocess.run(shlex.split(f'zip -r {out_path} {mesh_dir}'))
    return out_path

@app.route('/generate', methods=['POST'])
def run():
    data = request.json
    shape_path = data['shape_path']
    text = data['text']
    seed = int(data['seed'])
    guidance_scale = float(data['guidance_scale'])

    if not shape_path.endswith('.obj'):
        return jsonify(error='The input file is not .obj file.'), 400
    if not check_num_faces(shape_path):
        return jsonify(error='The number of faces is over 100,000.'), 400

    config = load_config(shape_path, text, seed, guidance_scale)
    trainer = TEXTure(config)

    # ... (continue the rest of the run method here)

    # Finally, instead of yielding the results, return them
    return jsonify(sample_image_paths=sample_image_paths, mesh_path=mesh_path.as_posix(), zip_path=zip_path, status='Done!')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
