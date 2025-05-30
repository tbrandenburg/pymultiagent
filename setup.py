from setuptools import setup, find_packages

setup(
    name="pymultiagent",
    version="0.1.0",
    description="Python multi-agent framework",
    author="Tom Brandenburg",
    author_email="kabelkaspertom@googlemail.com",
    license="Apache",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "gradio",
        "tqdm",
        "openai",
        "openai-agents",
        "ipykernel",
        "openpyxl",
        "python-dotenv",
        "requests",
        "pillow",
        "cairosvg",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "pymultiagent=pymultiagent.main:cli_main",
        ],
    },
    package_data={
        "pymultiagent": ["prompts/*.txt"],
    },
)
