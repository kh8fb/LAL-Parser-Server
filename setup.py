from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='lal-parser-server',
    entry_points={
        'console_scripts': [
            'lal-parser-server=src_joint.server.serve:serve'
        ],
    },
)
