from setuptools import setup

packages = ['gym_monitor']
install_requires = [
    'gymnasium',
    'pygame'
]

entry_points = {
    'gymnasium.envs': ['gym_monitor=gym_monitor.gym:register_envs']
}

setup(
    name='gym_monitor',
    version='0.0.1',
    license='GPL',
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
)
