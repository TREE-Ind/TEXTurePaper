from flask import Flask, request, jsonify, send_file
import datetime
import pathlib
import shlex
import subprocess
import sys
from typing import Optional
from werkzeug.utils import secure_filename
import tempfile
import os

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
    guide.background_img = 'textures/brick_wall.png'
    guide.shape_path = 'shapes/spot_triangulated.obj'
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
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    # Other data from the request
    text = request.form.get('text')
    seed = int(request.form.get('seed'))
    guidance_scale = float(request.form.get('guidance_scale'))

    if file and file.filename.endswith('.obj'):
        shape_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
        file.save(shape_path)
    else:
        return jsonify(error='The input file is not .obj file.'), 400

    if not check_num_faces(shape_path):
        os.remove(shape_path)
        return jsonify(error='The number of faces is over 100,000.'), 400


    trainer.mesh_model.train()

    total_steps = len(trainer.dataloaders['train'])
    sample_image_paths = []
    for step, data in enumerate(trainer.dataloaders['train'], start=1):
        trainer.paint_step += 1
        trainer.paint_viewpoint(data)
        trainer.evaluate(trainer.dataloaders['val'],
                            trainer.eval_renders_path)
        trainer.mesh_model.train()

        sample_image_dir = config.log.exp_dir / 'vis' / 'eval'
        sample_images = sorted(sample_image_dir.glob(f'step_{trainer.paint_step:05d}_*.jpg'))
        sample_image_paths.extend(path.as_posix() for path in sample_images)

    trainer.mesh_model.module.change_default_to_median()

    save_dir = trainer.exp_path / 'mesh'
    save_dir.mkdir(exist_ok=True, parents=True)
    trainer.mesh_model.module.export_mesh(save_dir)
    model_path = save_dir / 'mesh.obj'
    mesh = trimesh.load(model_path)
    mesh_path = save_dir / 'mesh.glb'
    mesh.export(mesh_path, file_type='glb')

    #zip_path = zip_results(config.log.exp_dir)
    
    # Return the generated point cloud as a response
    return send_file(mesh_path)

@app.route('/download', methods=['GET'])
def download():
    mesh_path = run()
    return send_file(mesh_path, as_attachment=True, attachment_filename='result.glb')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
