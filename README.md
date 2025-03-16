![](media/zorch.png?raw=true)


# Zorch: A Tensor Library with a Pytorch-like API in Zig

Zorch is a lightweight, high-performance tensor library written in Zig. It provides a flexible and efficient framework for numerical computations, automatic differentiation, and machine learning. The library is designed to be simple, modular, and easy to extend.


#  Table of Contents

- [Zorch: A Tensor Library with a Pytorch-like API in Zig](#zorch-a-tensor-library-with-a-pytorch-like-api-in-zig)
- [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Building the Project](#building-the-project)
  - [Documentation](#documentation)
  - [Examples](#examples)
    - [Creating a Tensor](#creating-a-tensor)
    - [Performing Tensor Operations](#performing-tensor-operations)
    - [Using Automatic Differentiation](#using-automatic-differentiation)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- **Multi-dimensional Tensors**: Support for tensors of arbitrary shapes and data types.
- **Automatic Differentiation**: Built-in support for backpropagation and gradient computation.
- **Optimization**: Includes common optimization algorithms like Stochastic Gradient Descent (SGD).
- **Activation Functions**: Implements popular activation functions such as ReLU, Tanh, Sigmoid, and Softmax.
- **Broadcasting**: Supports broadcasting for element-wise operations.
- **Custom Errors**: Comprehensive error handling for tensor operations.
- **Logging**: Configurable logging for debugging and monitoring.

## Project Structure

The project is organized as follows:

```
.
├── build.zig            # Build configuration for Zig
├── build.zig.zon        # Dependency management for Zig
├── docs/                # Documentation and generated files
├── src/                 # Source code
│   ├── autograd.zig     # Automatic differentiation
│   ├── data.zig         # Data loading and preprocessing
│   ├── dtypes.zig       # Data type definitions
│   ├── errors.zig       # Custom error handling
│   ├── functional.zig   # Functional programming utilities
│   ├── logger.zig       # Logging utilities
│   ├── main.zig         # Entry point for the application
│   ├── ndarray.zig      # Core tensor operations
│   ├── nn.zig           # Neural network components
│   ├── ops.zig          # Tensor operations
│   ├── optim.zig        # Optimization algorithms
│   ├── root.zig         # Root module for the library
│   ├── tensor.zig       # Tensor abstraction
│   ├── utils.zig        # Utility functions
│   └── zorch.zig        # Main library module
└── zig-out/             # Build output directory
    ├── bin/             # Compiled binaries
    │   └── zorch        # Executable
    └── lib/             # Compiled libraries
        └── libzorch.a   # Static library
```

## Getting Started

### Prerequisites

- [Zig](https://ziglang.org/download/) (version 0.13.0 or later)

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/zorch.git
   cd zorch
   ```

2. Build the project using Zig:
   ```bash
   zig build
   ```

   This will generate the following outputs:
   - Executable: `zig-out/bin/zorch`
   - Static library: `zig-out/lib/libzorch.a`

3. Run the executable:
   ```bash
   ./zig-out/bin/zorch
   ```
   Or use 
    ```bash
   zig build run
   ```

### Using the Library

To use Zorch in your Zig project, add it as a dependency in your `build.zig.zon` file:

```zig
.dependencies = .{
    .zorch = .{
        .url = "https://github.com/your-username/zorch/archive/main.tar.gz",
        .hash = "your-hash-here",
    },
},
```

Then, import the library in your Zig code:

```zig
const zorch = @import("zorch");

pub fn main() !void {
    // Example usage
    const allocator = std.heap.page_allocator;
    const tensor = try zorch.Tensor.from_value(allocator, &[_]usize{2, 2}, .f32, 7.2);
    defer tensor.deinit();

    try tensor.print();
}
```

## Documentation

For detailed documentation, refer to the [docs](./docs/index.html) directory. You can also generate the documentation locally:

```bash
zig build docs
```

This will generate HTML documentation in the `docs/` directory.

## Examples

### Creating a Tensor

```zig
const allocator = std.heap.page_allocator;
const tensor = try zorch.Tensor.from_value(allocator, &[_]usize{2, 2}, .f32, 1.0);
defer tensor.deinit();
```

### Performing Tensor Operations

```zig
const a = try zorch.Tensor.from_value(allocator, &[_]usize{2, 2}, .f32, 1.0);
const b = try zorch.Tensor.from_value(allocator, &[_]usize{2, 2}, .f32, 2.0);
defer a.deinit();
defer b.deinit();

const result = try a.add(b, false);
defer result.deinit();
```

### Using Automatic Differentiation

```zig
const x = try zorch.Tensor.from_value(allocator, &[_]usize{2, 2}, .f32, 1.0);
x.requires_grad = true;
defer x.deinit();

const y = try x.mul_scalar(2.0, false);
defer y.deinit();

try y.backward(null);
```

## Contributing

- Contributions are welcome! Please open an issue or submit a pull request for any bugs, feature requests, or improvements.
- Let me know if you need further help!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


