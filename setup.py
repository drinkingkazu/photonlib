from skbuild import setup
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="photonlib",
    version="0.1",
    author=['Ka Vang (Patrick) Tsang, Carolyn Smith, Sam Young, Kazuhiro Terao'],
    description='Photon transportation physics models',
    license='MIT',
    keywords='Interface software for photon libraries in LArTPC experiments',
    scripts=['bin/download_icarus_plib.sh','bin/download_2x2_plib.sh'],
    packages=['photonlib'],
    install_requires=[
        'numpy',
        'scikit-build',
        'torch',
        'h5py',
        'gdown',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
