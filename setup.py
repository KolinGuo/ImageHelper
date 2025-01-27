from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="image_helper",
        description="A simple image helper for combining images and adding texts",
        packages=find_packages(
            include=["image_helper*"],
        ),
        include_package_data=True,
    )
