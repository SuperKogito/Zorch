const std = @import("std");
const utils = @import("math/ndarray/utils.zig");
const dtypes = @import("math/ndarray/dtypes.zig");

const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("errors.zig").TensorError;
const NdArray = @import("math/ndarray/ndarray.zig").NdArray;

// Operation interface
pub const Operation = struct {
    notation: []const u8,
    forward: *const fn (allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor,
    backward: *const fn (tensor: *const Tensor, grad: *NdArray) TensorError!void,
};

// Helper function to initialize an operation with an empty cache
fn initOperation(notation: []const u8, forward: anytype, backward: anytype) Operation {
    return Operation{
        .notation = notation,
        .forward = forward,
        .backward = backward,
    };
}

// Available operations
pub const mse_op = initOperation("Mse", &MseOp.forward, &MseOp.backward);
pub const add_op = initOperation("Add", &AddOp.forward, &AddOp.backward);
pub const mul_op = initOperation("Mul", &MulOp.forward, &MulOp.backward);
pub const sub_op = initOperation("Sub", &SubOp.forward, &SubOp.backward);
pub const div_op = initOperation("Div", &DivOp.forward, &DivOp.backward);
pub const transpose_op = initOperation("Transpose", &TransposeOp.forward, &TransposeOp.backward);
pub const tanh_op = initOperation("Tanh", &TanhOp.forward, &TanhOp.backward);
pub const relu_op = initOperation("Relu", &ReLUOp.forward, &ReLUOp.backward);
pub const sigmoid_op = initOperation("Sigmoid", &SigmoidOp.forward, &SigmoidOp.backward);
pub const pow_op = initOperation("Pow", &PowOp.forward, &PowOp.backward);
pub const gemm_op = initOperation("Gemm", &GemmOp.forward, &GemmOp.backward);
pub const softmax_op = initOperation("Softmax", &SoftmaxOp.forward, &SoftmaxOp.backward);
pub const min_op = initOperation("Min", &MinOp.forward, &MinOp.backward);
pub const max_op = initOperation("Max", &MaxOp.forward, &MaxOp.backward);
pub const mean_op = initOperation("Mean", &MeanOp.forward, &MeanOp.backward);

pub const MseOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const pred = params[0];
        const target = params[1];
        // pred.label = "prediction";
        // target.label = "target";

        const diff = try pred.data.sub(target.data, false); // pred - target
        const squared_diff = try diff.mul(diff, false); // (pred - target)^2
        const mse = try squared_diff.mean(null); // mean((pred - target)^2)

        return try Tensor.init(
            allocator,
            mse.shape,
            mse.dtype,
            mse,
            &[_]*Tensor{ pred, target },
            @as(?*Operation, @constCast(&mse_op)),
            "mse(y-y_hat) = (1/n) . (y-y_hat)^2",
            pred.requires_grad or target.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const pred = tensor.parents[0];
        const target = tensor.parents[1];
        if (pred.grad) |x_grad| {
            const n = utils.compute_size(pred.shape); // Batch size
            const diff = try pred.data.sub(target.data, false); // pred - target
            const pred_grad = try diff.mul_scalar(2.0 / @as(f32, @floatFromInt(n)), false); // 2 * (pred - target) / n
            _ = try x_grad.add(pred_grad, true);
        }
        if (target.grad) |y_grad| {
            // std.debug.print("Warning: Gradient is not implemented for target tensor in MSE operation.", .{});
            _ = y_grad;
            _ = grad;
        }
        // std.debug.print("mse.backwrd() \n", .{});
    }
};

// Add Operation
pub const AddOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.add(y.data, false);

        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&add_op)),
            "add-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        // // std.debug.print("enter add.backward() \n", .{});

        const x = tensor.parents[0];
        const y = tensor.parents[1];
        // try grad.info();
        // try grad.print();

        // // std.debug.print("x: {s}  \n", .{x.label});
        // if (x.grad) |x_grad| {
        //     try x_grad.info();
        //     try x_grad.print();
        // }
        // // std.debug.print("y: {s}  \n", .{y.label});
        // if (y.grad) |y_grad| {
        //     try y_grad.info();
        //     try y_grad.print();
        // }
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad, true);
        }
        // std.debug.print("1 \n", .{});

        if (y.grad) |y_grad| {
            _ = try y_grad.add(grad, true);
        }
        // // std.debug.print("exit add.backward() \n", .{});
    }
};

// Multiply Operation
pub const MulOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.mul(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&mul_op)),
            "mul-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];

        if (x.grad) |x_grad| {
            const grad_x = try y.data.mul(grad, false);
            _ = try x_grad.add(grad_x, true);
        }

        if (y.grad) |y_grad| {
            const grad_y = try x.data.mul(grad, false);
            _ = try y_grad.add(grad_y, true);
        }
        // std.debug.print("mul.backward() \n", .{});
    }
};

// Subtract Operation
pub const SubOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.sub(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&sub_op)),
            "sub-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad, true);
        }
        if (y.grad) |y_grad| {
            const neg_grad = try grad.neg(false);
            _ = try y_grad.add(neg_grad, true);
        }
        // std.debug.print("mul.backward() \n", .{});
    }
};

// Divide Operation
pub const DivOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.div(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&div_op)),
            "div-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];
        if (x.grad) |x_grad| {
            const grad_x = try grad.div(y.data, false);
            _ = try x_grad.add(grad_x, true);
        }
        if (y.grad) |y_grad| {
            const y_squared = try y.data.mul(y.data, false);
            const grad_y = try x.data.mul(grad, false);
            const neg_grad_y = try grad_y.neg(false);
            const final_grad_y = try neg_grad_y.div(y_squared, false);
            _ = try y_grad.add(final_grad_y, true);
        }
        // std.debug.print("mul.backward() \n", .{});
    }
};

// Transpose Operation
pub const TransposeOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.transpose();
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&transpose_op)),
            "transpose-res",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const grad_t = try grad.transpose();

        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_t, true);
        }
        // std.debug.print("trans.backward() \n", .{});
    }
};

// Tanh Operation
pub const TanhOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.tanh(false);
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&tanh_op)),
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        // Compute tanh(x) during the backward pass
        const y_data = try x.data.tanh(false);

        // Compute 1 - tanh(x)^2
        const z_squared = try y_data.mul(y_data, false);
        const neg_z_squared = try z_squared.neg(false);

        const one_minus_z_squared = try neg_z_squared.add_scalar(1.0, true);

        // Compute the gradient: grad * (1 - tanh(x)^2)
        const grad_tanh = try grad.mul(one_minus_z_squared, false);

        // Accumulate the gradient in x.grad
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_tanh, true);
        }
    }
};

// ReLU Operation
pub const ReLUOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.relu(false);
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&relu_op)),
            "relu-res",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        if (x.grad) |x_grad| {
            const mask = try x.data.greater_than_scalar(0.0, false);
            const grad_relu = try grad.mul(mask, false);
            _ = try x_grad.add(grad_relu, true); // Handle the error
        }
        // std.debug.print("relu.backward() \n", .{});
    }
};

// Sigmoid Operation
pub const SigmoidOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.sigmoid(false);
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&sigmoid_op)),
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const z = try x.data.sigmoid(false); // z = sigmoid(x)
        const neg_z = try z.neg(false); // neg_z = -z
        const one_minus_z = try neg_z.add_scalar(1.0, false); // one_minus_z = 1 - z
        const z_times_one_minus_z = try z.mul(one_minus_z, false); // z * (1 - z)
        const grad_sigmoid = try grad.mul(z_times_one_minus_z, false); // grad * z * (1 - z)

        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_sigmoid, true); // Accumulate gradient
        }
    }
};

// Power Operation
pub const PowOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.pow(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&div_op)),
            "Pow-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const n = tensor.parents[1];

        // Compute gradient for x
        if (x.grad) |x_grad| {
            const n_minus_1 = try n.data.sub_scalar(1.0, false);
            defer n_minus_1.deinit(); // Deallocate n_minus_1 after use

            const x_pow_n_minus_1 = try x.data.pow(n_minus_1, false);
            defer x_pow_n_minus_1.deinit(); // Deallocate x_pow_n_minus_1 after use

            const n_times_x_pow_n_minus_1 = try n.data.mul(x_pow_n_minus_1, false);
            defer n_times_x_pow_n_minus_1.deinit(); // Deallocate n_times_x_pow_n_minus_1 after use

            const grad_x = try grad.mul(n_times_x_pow_n_minus_1, false);
            defer grad_x.deinit(); // Deallocate grad_x after use

            _ = try x_grad.add(grad_x, true);
        }

        // Compute gradient for n
        if (n.grad) |n_grad| {
            const x_pow_n = try x.data.pow(n.data, false);
            defer x_pow_n.deinit(); // Deallocate x_pow_n after use

            const log_x = try x.data.log(false);
            defer log_x.deinit(); // Deallocate log_x after use

            const z = try x_pow_n.mul(log_x, false);
            defer z.deinit(); // Deallocate z after use

            const grad_n = try grad.mul(z, false);
            defer grad_n.deinit(); // Deallocate grad_n after use

            _ = try n_grad.add(grad_n, true);
        }
        // std.debug.print("pow.backward() \n", .{});
    }
};

// GEMM Operation (General Matrix Multiplication)
pub const GemmOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const x = params[0];
        const y = params[1];
        const result_data = try x.data.matmul(y.data, alpha, beta, false, false);
        const result = try Tensor.from_ndarray(
            allocator,
            result_data,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&gemm_op)),
            "",
            x.requires_grad or y.requires_grad,
        );
        return result;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];
        if (x.grad) |x_grad| {
            const grad_x = try grad.matmul(y.data, 1.0, 0.0, false, true);
            _ = try x_grad.add(grad_x, true);
        }
        if (y.grad) |y_grad| {
            const grad_y = try x.data.matmul(grad, 1.0, 0.0, true, false);
            _ = try y_grad.add(grad_y, true);
        }
    }
};

// Softmax Operation
pub const SoftmaxOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];

        // Ensure the axis is valid for the input tensor's shape
        const axis: usize = if (x.data.shape.len > 1) 1 else 0;

        // Compute softmax
        const y_data = try x.data.softmax(axis, false);

        // Create a new Tensor from the softmax result
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&softmax_op)),
            "",
            x.requires_grad,
        );

        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        // Ensure the axis is valid for the input tensor's shape
        const axis: usize = if (x.data.shape.len > 1) 1 else 0;
        const softmax_x = try x.data.softmax(axis, false);

        // Compute (1 - softmax_x)
        const neg_softmax_x = try softmax_x.neg(false);
        const one_minus_softmax_x = try neg_softmax_x.add_scalar(1.0, true);

        // Compute softmax_x * (1 - softmax_x)
        const z = try softmax_x.mul(one_minus_softmax_x, false);

        // Multiply by upstream gradient
        const grad_softmax = try grad.mul(z, false);

        // Accumulate gradient
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_softmax, true);
        }
    }
};

// Max Operation
pub const MaxOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.max(false); // Use `max` instead of `min`
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&max_op)), // Use `max_op` instead of `tanh_op`
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        // Compute the gradient of the max operation
        const grad_max = try x.data.maxBackward(grad, false); // Use `maxBackward` to compute the gradient

        // Accumulate the gradient in x.grad
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_max, true);
        }
    }
};

// Mean Operation
pub const MeanOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.mean(null); // Use `mean` instead of `min`
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&mean_op)), // Use `mean_op` instead of `tanh_op`
            "Mean-res",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        // std.debug.print("BACKWARD mean\n", .{});

        const x = tensor.parents[0];
        const z_size = x.data.get_size();
        const grad_val: f32 = 1.0 / @as(f32, @floatFromInt(z_size));
        // Compute the gradient of the mean operation
        const grad_mean = try NdArray.from_value(x.allocator, x.shape, x.data.dtype, grad_val);
        const grad_x = try grad_mean.mul(grad, false);

        // Accumulate the gradient in x.grad
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_x, true);
        }
    }
};

// Min Operation
pub const MinOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.min(false); // Compute the minimum value
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&min_op)), // Use `min_op` for the operation
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        // Compute the gradient of the min operation
        const grad_min = try x.data.min_backward(grad, false); // Use `minBackward` to compute the gradient

        // Accumulate the gradient in x.grad
        if (x.grad) |x_grad| {
            _ = try x_grad.add(grad_min, true);
        }
    }
};

// // Tests
// const testing = std.testing;

// // Test Add Operation
// test "Add operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 2.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try add_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 5.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try add_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Multiply Operation
// test "Multiply operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 2.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try mul_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 6.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try mul_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 3.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 2.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Subtract Operation
// test "Subtract operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 5.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try sub_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 2.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try sub_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, -1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Divide Operation
// test "Divide operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 6.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try div_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 2.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try div_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0 / 3.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, -6.0 / 9.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Power Operation
// test "Power operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     const x_val = 2.0;
//     const n_val = 3.0;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{3}, dtypes.DataType.f32, x_val, true);
//     defer x.deinit(); // Ensure cleanup
//     const n = try Tensor.from_value(allocator, &[_]usize{3}, dtypes.DataType.f32, n_val, true);
//     defer n.deinit(); // Ensure cleanup

//     // Forward pass
//     const y = try pow_op.forward(allocator, &[_]*Tensor{ x, n });
//     defer y.deinit(); // Ensure cleanup

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{0}); // Get the first element of the result
//     try std.testing.expectApproxEqAbs(y_value.f32, 8.0, 1e-6); // Check if y = 2^3 = 8

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{3}, dtypes.DataType.f32);
//     defer grad.deinit(); // Ensure cleanup
//     try pow_op.backward(y, grad); // Compute gradients

//     // Gradient check for x
//     const expected_grad_x = n_val * std.math.pow(f32, x_val, n_val - 1); // dL/dx = 3 * 2^2 = 12
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Get the gradient of x

//     // Check gradient for x
//     try std.testing.expectApproxEqAbs(x_grad_value.f32, expected_grad_x, 1e-6);

//     // Gradient check for n
//     const expected_grad_n = std.math.pow(f32, x_val, n_val) * std.math.log(f32, std.math.e, x_val); // dL/dn = 8 * ln(2)
//     const n_grad_value = try n.grad.?.get(&[_]usize{0}); // Get the gradient of n

//     // Check gradient for n
//     try std.testing.expectApproxEqAbs(n_grad_value.f32, expected_grad_n, 1e-6);
// }

// // Test Transpose Operation
// test "Transpose operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);

//     // Forward pass
//     const y = try transpose_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{ 0, 1 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, 3.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 1.0, 1.0, 1.0 });
//     try transpose_op.backward(y, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 1, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test Tanh Operation
// test "Tanh operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 0.5, true);

//     // Forward pass
//     const y = try tanh_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const expected_tanh = std.math.tanh(@as(f32, 0.5)); // Use runtime tanh for f32
//     const y_value = try y.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, expected_tanh, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try tanh_op.backward(y, grad);

//     // Gradient check
//     const expected_grad = 1.0 - expected_tanh * expected_tanh;
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, expected_grad, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test ReLU Operation
// test "ReLU operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, -1.0, true);

//     // Forward pass
//     const y = try relu_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, 0.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try relu_op.backward(y, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 0.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test GEMM Operation
// test "GEMM operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
//     const y = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);

//     // Forward pass
//     const z = try gemm_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 19.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 1.0, 1.0, 1.0 });
//     try gemm_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 11.0, 1e-6);
//     const y_grad_value = try y.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 4.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Softmax Operation
// test "Softmax operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensor
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };
//     const x = try Tensor.from_data(allocator, shape, dtypes.DataType.f32, &data_a, true);
//     defer x.deinit(); // Clean up x

//     // Forward pass
//     const res = try softmax_op.forward(allocator, &[_]*Tensor{x});
//     defer res.deinit(); // Clean up res

//     // Compute expected softmax values for each row.
//     const exp1_row1 = @exp(@as(f32, 1.0));
//     const exp2_row1 = @exp(@as(f32, 2.0));
//     const exp3_row1 = @exp(@as(f32, 3.0));
//     const sum_row1 = exp1_row1 + exp2_row1 + exp3_row1;
//     const expected_row1 = [_]f32{ exp1_row1 / sum_row1, exp2_row1 / sum_row1, exp3_row1 / sum_row1 };

//     const exp1_row2 = @exp(@as(f32, 4.0));
//     const exp2_row2 = @exp(@as(f32, 5.0));
//     const exp3_row2 = @exp(@as(f32, 6.0));
//     const sum_row2 = exp1_row2 + exp2_row2 + exp3_row2;
//     const expected_row2 = [_]f32{ exp1_row2 / sum_row2, exp2_row2 / sum_row2, exp3_row2 / sum_row2 };

//     // Verify the results for the first row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 0, i });
//         try std.testing.expectApproxEqAbs(expected_row1[i], out.f32, 0.0001);
//     }

//     // Verify the results for the second row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 1, i });
//         try std.testing.expectApproxEqAbs(expected_row2[i], out.f32, 0.0001);
//     }

//     // Backward pass
//     const upstream_grad_data = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }; // Upstream gradient
//     const upstream_grad = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, &upstream_grad_data);
//     defer upstream_grad.deinit(); // Clean up upstream_grad

//     try softmax_op.backward(res, upstream_grad);

//     // Verify the gradients
//     if (x.grad) |x_grad| {
//         // Compute expected gradients manually
//         const softmax_row1 = expected_row1;
//         const softmax_row2 = expected_row2;

//         // Gradient for the first row
//         const grad_row1 = [_]f32{
//             softmax_row1[0] * (1 - softmax_row1[0]) * upstream_grad_data[0],
//             softmax_row1[1] * (1 - softmax_row1[1]) * upstream_grad_data[1],
//             softmax_row1[2] * (1 - softmax_row1[2]) * upstream_grad_data[2],
//         };

//         // Gradient for the second row
//         const grad_row2 = [_]f32{
//             softmax_row2[0] * (1 - softmax_row2[0]) * upstream_grad_data[3],
//             softmax_row2[1] * (1 - softmax_row2[1]) * upstream_grad_data[4],
//             softmax_row2[2] * (1 - softmax_row2[2]) * upstream_grad_data[5],
//         };

//         // Verify the gradients for the first row
//         for (0..3) |i| {
//             const grad = try x_grad.get(&[_]usize{ 0, i });
//             try std.testing.expectApproxEqAbs(grad_row1[i], grad.f32, 0.0001);
//         }

//         // Verify the gradients for the second row
//         for (0..3) |i| {
//             const grad = try x_grad.get(&[_]usize{ 1, i });
//             try std.testing.expectApproxEqAbs(grad_row2[i], grad.f32, 0.0001);
//         }
//     } else {
//         try std.testing.expect(false); // Fail if x.grad is null
//     }
// }

// // Test Sigmoid Operation
// test "Sigmoid operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x_value: f32 = 0.5;
//     const x = try Tensor.from_value(allocator, &[_]usize{ 3, 1 }, dtypes.DataType.f32, x_value, true);

//     // Forward pass
//     const y = try sigmoid_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const expected_sigmoid = 1.0 / (1.0 + @exp(-x_value));
//     const y_value = try y.data.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, expected_sigmoid, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{ 3, 1 }, dtypes.DataType.f32);
//     try sigmoid_op.backward(y, grad);

//     // Gradient check
//     const expected_grad = expected_sigmoid * (1.0 - expected_sigmoid); // ~0.235003
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, expected_grad, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }
