from setuptools import find_packages, setup

setup(name="euterpe",
    version="0.2",
    description="goddess of music",
    author="Andros Tjandra (modified by: Sashi Novitasari)",
    author_email='andros.tjandra@gmail.com',
    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
    license="BSD",
    url="",
    packages=find_packages(),
    install_requires=['numpy','scipy', 'torch', 'pytest', 'torchev', 
        'utilbox', 'tabulate', 'tqdm', 'pathos', 'librosa', 'tensorboardX', 
        'pandas', 'tables', 'python-speech-features', 'soundfile','psutil','tensorboard','kenlm','pyyaml']);
