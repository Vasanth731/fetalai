# import torch

# BATCH_SIZE = 1

# def model2onnx(model, save_path):

#     model.eval()
#     model = model.cuda()
#     print("Batch size @ ", BATCH_SIZE)
#     dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224, device="cuda")  # Adjust shape based on your model

#     torch.onnx.export(
#         model,
#         dummy_input,
#         save_path,
#         input_names=["input"],
#         output_names=["logits"],
#         # dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
#         opset_version=18,
#         do_constant_folding=True,
#         dynamo=True
#     )
#     print(f"ONNX model saved to: {save_path}")

###############################################################################

import torch
from model import YNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_channels = 1
output_channels = 64
n_class = 1

# load model
model = YNet(input_channels=input_channels, output_channels=output_channels, n_class=n_class)
model.load_state_dict(torch.load("fuvai_weights.pt", map_location=device))
model = model.to(device)
model.eval()

# Dummy input: (T, B, C, H, W) since batch_first=False
dummy_input = torch.randn(1, 1, 1, 224, 224, device=device)  

# Make sure model is in eval mode
model.eval()

onnx_path = "ynet_fuvai.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,  # You can increase to 16 if needed
    do_constant_folding=True,
    input_names=["input"],
    output_names=["classification", "segmentation"],
    dynamic_axes={
        "input": {0: "time_steps", 1: "batch_size"},  # allow variable sequence length & batch
        "classification": {0: "batch_size"},
        "segmentation": {0: "batch_size"}
    }
)

print(f"Model successfully exported to {onnx_path}")
