import os
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="extension",
    ext_modules=[
        cpp_extension.CppExtension(
            name="extension",
            sources=[
                "src/extension.cpp",
                "src/feat_to_v_attr.cpp",
                "src/feat_to_e.cpp",
                "src/large_scale_feat_to_v.cpp",
                "src/large_scale_feat_to_e.cpp",
                "src/utils.cpp"
            ],
            include_dirs=[os.path.abspath("include")],
            extra_compile_args=[
                "-Wall",
                "-std=c++14"
            ],
            define_macros=[
                # ("_DEBUG", None)
            ]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension}
)
