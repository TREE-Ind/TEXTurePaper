from flask import Flask, request, jsonify, send_file
import datetime
import pathlib
import shlex
import subprocess
import sys
from typing import Optional
import sys

import trimesh
import boto3
from botocore.exceptions import NoCredentialsError

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
    data = request.json
    shape_path = data['shape_path']  # Now this will be S3 URL
    text = data['text']
    seed = int(data['seed'])
    guidance_scale = float(data['guidance_scale'])

    if not shape_path.endswith('.obj'):
        return jsonify(error='The input file is not .obj file.'), 400

    # Parse S3 URL
    path_parts = shape_path.replace("https://", "").split("/")
    bucket_name = path_parts[1]
    file_path_in_bucket = "/".join(path_parts[2:])

    # Download the file
    s3 = boto3.client('s3')
    local_file_path = "/" + path_parts[-1]
    try:
        s3.download_file(bucket_name, file_path_in_bucket, local_file_path)
    except NoCredentialsError:
        return jsonify(error='No AWS credentials found.'), 400
    except Exception as e:
        return jsonify(error=str(e)), 400

    if not check_num_faces(local_file_path):
        os.remove(local_file_path)
        return jsonify(error='The number of faces is over 100,000.'), 400

    config = load_config(local_file_path, text, seed, guidance_scale)
    
    trainer = TEXTure(config)


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
