from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='image_helper',
        version='0.0.1',
        description="A simple image helper for combining images and adding texts",
        python_requires=">=3.8",
        install_requires=[
            'Pillow',
            'opencv-python',
            'numpy',
        ],
        packages=find_packages(
            include=['image_helper*'],
        ),
        include_package_data=True,
    )
