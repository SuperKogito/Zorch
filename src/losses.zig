const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("errors.zig").TensorError;

const utils = @import("math/ndarray/utils.zig");
const dtypes = @import("math/ndarray/dtypes.zig");
const NdArray = @import("math/ndarray/ndarray.zig").NdArray;

const ops = @import("autograd.zig");

pub const Operation = ops.Operation;
pub const mse_op = ops.mse_op;

pub const Loss = struct {
    forward: *const fn (pred: *Tensor, target: *Tensor) TensorError!*Tensor, // Forward pass for loss computation
};

pub const mse_loss = Loss{
    .forward = &MSE.forward,
};

pub const MSE = struct {
    base: Loss,

    pub fn init() MSE {
        return MSE{
            .base = mse_loss,
        };
    }

    /// Forward pass for MSE loss: (pred - target)^2 / n
    pub fn forward(pred: *Tensor, target: *Tensor) TensorError!*Tensor {
        const allocator = std.heap.page_allocator;
        const op = &mse_op;
        return try op.forward(allocator, &[_]*Tensor{ pred, target });
    }
};

const testing = std.testing;

test "MSE Loss - Forward Pass" {
    const allocator = std.heap.page_allocator;

    // Create input tensors
    const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
    const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);

    // Compute MSE loss
    const mse = try mse_loss.forward(pred, target);
    const mse_value = try mse.get(&[_]usize{0});
    try std.testing.expectEqual(0.0, mse_value.f32);

    pred.deinit();
    target.deinit();
    mse.deinit();
}

test "MSE Loss - Forward Pass with Different Values" {
    const allocator = std.heap.page_allocator;

    // Create input tensors
    const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 2.0, 3.0, 4.0, 5.0 }, true);
    const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);

    // Compute MSE loss
    const mse = try mse_loss.forward(pred, target);
    const mse_value = try mse.get(&[_]usize{0});

    // Expected MSE loss
    const expected_mse: f32 = (1.0 * 1.0 + 1.0 * 1.0 + 1.0 * 1.0 + 1.0 * 1.0) / 4.0;
    try testing.expectApproxEqAbs(mse_value.f32, expected_mse, 1e-6);

    // Clean up
    pred.deinit();
    target.deinit();
    mse.deinit();
}

test "MSE Loss - Backward Pass" {
    const allocator = std.heap.page_allocator;

    // Create input tensors
    const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 2.0, 3.0, 4.0, 5.0 }, true);
    const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, false);

    // Compute gradient
    const mse = try mse_loss.forward(pred, target);
    const in_grad = try NdArray.ones(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32);
    try mse.backward(in_grad);

    // Expected gradient
    const expected_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 0.5, 0.5, 0.5, 0.5 });
    
    if (pred.grad) |computed_grad| {    
        const total_size = utils.compute_size(computed_grad.shape);
        try computed_grad.print();

        for (0..total_size) |i| {
            const row = i / computed_grad.shape[1];
            const col = i % computed_grad.shape[1];
            const expected_value = try expected_grad.get(&[_]usize{ row, col });
            const computed_value = try computed_grad.get(&[_]usize{ row, col });
            try std.testing.expectEqual(expected_value.f32, computed_value.f32);
        }
    }
    
    // Clean up
    pred.deinit();
    target.deinit();
    expected_grad.deinit();
}

test "MSE Loss - Backward Pass with Different Batch Size" {
    const allocator = std.heap.page_allocator;

    // Create input tensors
    const pred = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, dtypes.DataType.f32, &[_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }, true);
    const target = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, true);

    // Compute gradient
    const mse = try mse_loss.forward(pred, target);
    const in_grad = try NdArray.ones(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32);
    try mse.backward(in_grad);

    // Expected gradient
    const expected_grad = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, dtypes.DataType.f32, &[_]f32{ 0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333 }, true);

    if (pred.grad) |computed_grad| {    
        const total_size = utils.compute_size(computed_grad.shape);
        try computed_grad.print();

        for (0..total_size) |i| {
            const row = i / computed_grad.shape[1];
            const col = i % computed_grad.shape[1];
            const expected_value = try expected_grad.get(&[_]usize{ row, col });
            const computed_value = try computed_grad.get(&[_]usize{ row, col });

            const is_equal = std.math.approxEqAbs(f32, expected_value.f32, computed_value.f32, 1e-6);
            try std.testing.expect(is_equal);
        }
    }

    // Clean up
    pred.deinit();
    target.deinit();
    expected_grad.deinit();
}