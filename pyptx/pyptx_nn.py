from pyptx_ml import PyPTXModel



# Fully Connected Layer (Matrix Multiply + Activation)


pyptx_fc_layer = """
wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32.f32.f32 %rd3, %rd1, %rd2;
max.f32 %rd3, %rd3, 0.0;
"""

# Convolution Layer (2D Tensor Conv)
pyptx_conv_layer = """
mma.sync.aligned.m16n16k16.row.col.f32.f32.f32.f32 %rd3, %rd1, %rd2;
"""

# Softmax Output Layer (Exponential Normalization)
pyptx_softmax_layer = """
ex2.approx.f32 %rd3, %rd1;
"""

# Add to the PyPTX Model
model = PyPTXModel()
model.add(pyptx_fc_layer)
model.add(pyptx_conv_layer)
model.add(pyptx_softmax_layer)

# Run the Neural Network
model.run()
