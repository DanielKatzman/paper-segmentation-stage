# Third-party components

The standalone application includes Python, NumPy, SciPy, Pillow, PyMuPDF,
ONNX Runtime, and their transitive runtime components. Their license files are
included by the packaging system through installed distribution metadata.

The SegFormer implementation used during training comes from Hugging Face
Transformers and the MIT-licensed NVIDIA SegFormer model family. PyTorch and
Transformers are training/export dependencies and are not bundled in the
standalone CPU annotation application.

