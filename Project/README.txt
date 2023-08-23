git clone https://gitlab.com/libeigen/eigen.git
nvcc -I ./eigen VisionTransformer.cu --expt-relaxed-constexpr -o VisionTransformer