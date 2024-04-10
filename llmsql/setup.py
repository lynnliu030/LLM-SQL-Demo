from setuptools import setup, find_packages

vllm = ["vllm", "triton==2.2.0"]

setup(
    name="llmsql",
    version="0.1",
    packages=find_packages(),
    install_requires = ["openai"],
    extras_require = {
        "vllm": vllm,
    }
)
