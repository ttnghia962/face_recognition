from setuptools import setup

setup(
    name='FaceRTS',
    version='1.0.0',
    description='None',
    author='d4rk3r',
    author_email='huytvo.2003@gmail.com',
    url='https://github.com/ttnghia962/face_recognize',
    install_requires=[
        'opencv-python',
        'opencv-contrib-python==4.3.0.36',
        'tqdm',
        'torch==1.6.0',
        'torchvision==0.7.0',
        'facenet-pytorch==2.3.0',
        'Flask==1.1.2',
        'numpy==1.19.1',
        'Pillow==7.2.0',
        'requests==2.24.0',
        'filterpy==1.4.5',
    ],
    license='MIT',
    keywords='face tracking, face recognition, facenet, vggface2, pytorch, opencv',
)
