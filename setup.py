# setup.py
from setuptools import setup, find_packages

setup(
    name='lip_sync_demo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'torch',
        'transformers',
        'sounddevice',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'lip-sync-demo = main:main',
        ],
    },
    author='Your Name',
    description='Real-time lip-sync and speech-to-text demo (no MediaPipe)',
    python_requires='>=3.8',
)
