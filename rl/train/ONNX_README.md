# 1. 将torch模型转换为ONNX模型
- **动态轴支持**：对于具有动态输入形状（例如可变批量大小）的模型，你需要在转换过程中明确指定动态轴。这可确保ONNX模型具有灵活性以处理任何批量大小或其他可变尺寸。
- **dummpy_input**:注意生成初始dunmmy_input的时候不要使用`random`，因为input tensor需要为正。
- **版本兼容性**：请注意ONNX和PyTorch之间的版本兼容性。这里设置了`opset_version=15`
- **device**: 模型无需特殊的转换到GPU上。
- ****输出节点名称**：输出节点名称必须与模型中的输出节点名称相同，否则会出现错误。这里有4个output节点，分配`output{i}`。
  
```python
    def convert_to_onnx(self, onnx_path):
        self.restore()
        self.model.eval()
        dummy_input = torch.ones([gBatch_size, FS]).to(DEVICE)
        traced_model = torch.jit.trace(self.model, dummy_input)
        output_names = ["output" + str(i) for i in range(4)]
        torch.onnx.export(model=traced_model, args=dummy_input, f=onnx_path, opset_version=15, input_names=["input"], output_names=output_names, verbose=True)

        return f"ONNX model saved at {onnx_path}"
```
# 2. 将 ONNX 模型编译到 TVM 运行
- 得到onnx模型后，下一步就是编译它。为了实现这一点，我们将使用. 我们从编译过程中获得的输出是模型的 TAR 包，编译为目标平台的动态库。我们可以使用 TVM 运行时在目标设备上运行该模型。`tvmc compile` **注意输入的input名字和维度要对应上**。
```bash
tvmc compile --target "llvm" --input-shapes "input:[512, 37592]" --output checkpoint.tar ~/checkpoint.onnx
```
- 解压`checkpoint.tar`后将看到列出的三个文件:

1. `mod.so`是模型，表示为 C++ 库，可由 TVM 运行时加载。

2. `mod.json`是 TVM Relay 计算图的文本表示。

3. `mod.params`是包含预训练模型参数的文件。

# 3. 使用 TVMC 从已编译的模块运行模型

现在我们已经将模型编译到该模块中，我们可以使用 TVM 运行时来进行预测。TVMC 内置了 TVM 运行时，允许您运行编译后的 TVM 模型。要使用 TVMC 运行模型并进行预测，我们需要两件事：
- 我们刚刚生成的编译模块。
- 模型的有效输入以进行预测。

每个模型在预期的张量形状、格式和数据类型方面都是特定的。因此，大多数模型需要一些预处理和后处理，以确保输入有效并解释输出。**TVMC 采用 NumPy 的.npz格式作为输入和输出数据**。这是一种得到良好支持的 NumPy 格式，用于将多个数组序列化到一个文件中。

预处理得到**Batch_size大小的input.npz**，有了模型和输入数据，我们现在可以运行 TVMC 来进行预测：
```bash
tvmc run --inputs input.npz --output predictions.npz checkpoint.tar
```
T VMC 包括 TVM 运行时，它可以加载模型并根据输入进行预测。运行上述命令时，TVMC 输出一个新文件 ，**predictions.npz其中包含 NumPy 格式的模型输出张量**。

在此示例中，我们在用于编译的同一台计算机上运行模型。在某些情况下，我们可能希望通过 **RPC Tracker** 远程运行它。`tvmc run --help`

**读取ouput代码** 注意这里的名字不是ONNX中的名字，就是`output_{i}`
```python
#!python ./postprocess.py
import os.path
import numpy as np

output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        print(data["output_0"])
```

# 4. Refrence
- [ONNX 自定义算子](https://cloud.tencent.com/developer/article/2010629)
- [ONNX github文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [TVMC 编译和优化模型](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html#compiling-an-onnx-model-to-the-tvm-runtime)