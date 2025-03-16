const std = @import("std");
const zorch = @import("zorch.zig");

const ops = zorch.ops;
const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;
const Tensor = zorch.Tensor;
const NdArray = zorch.NdArray;

const F = zorch.functional;
const autograd = zorch.autograd;
const TensorError = zorch.errors.TensorError;

// =========================================================
// Losses
// =========================================================
/// A structure representing a loss function.
///
/// This struct contains a single field:
/// - `forward`: A function pointer to the forward pass implementation of the loss function.
pub const Loss = struct {
    forward: *const fn (pred: *Tensor, target: *Tensor) TensorError!*Tensor, // Forward pass for loss computation
};

/// A predefined instance of the `Loss` struct for Mean Squared Error (MSE) loss.
///
/// This instance uses the `MSE.forward` function for its forward pass.
pub const mse_loss = Loss{
    .forward = &MSELoss.forward,
};

/// A structure representing the Mean Squared Error (MSE) loss function.
///
/// This struct contains a single field:
/// - `base`: The base `Loss` struct.
pub const MSELoss = struct {
    base: Loss,

    /// Initializes a new `MSE` loss function.
    ///
    /// # Returns
    /// A new `MSE` instance.
    pub fn init() MSELoss {
        return MSELoss{
            .base = mse_loss,
        };
    }

    /// Computes the Mean Squared Error (MSE) loss between predictions and targets.
    ///
    /// # Parameters
    /// - `pred`: The predicted tensor.
    /// - `target`: The target tensor.
    ///
    /// # Returns
    /// A new tensor representing the MSELoss loss.
    ///
    /// # Errors
    /// Returns an error if the computation fails.
    pub fn forward(pred: *Tensor, target: *Tensor) TensorError!*Tensor {
        const allocator = std.heap.page_allocator;
        const op = &autograd.mse_op;
        return try op.forward(allocator, &[_]*Tensor{ pred, target });
    }
};

// =========================================================
// Layer
// =========================================================
/// An enum representing the types of layers.
///
/// This enum defines the following layer types:
/// - `Linear`: A fully connected linear layer.
pub const LayerType = enum { Linear };

/// A union representing different types of layers.
///
/// This union can hold the following layer types:
/// - `Linear`: A fully connected linear layer.
pub const Layer = union(LayerType) {
    Linear: *Linear,

    /// Performs the forward pass for the layer.
    ///
    /// # Parameters
    /// - `input`: The input tensor.
    ///
    /// # Returns
    /// A new tensor representing the output of the layer.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward(self: *Layer, input: *Tensor) !*Tensor {
        return switch (self.*) {
            .Linear => |linear| linear.forward(input),
        };
    }

    /// Retrieves the parameters of the layer.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for the parameters list.
    ///
    /// # Returns
    /// A list of tensors representing the layer's parameters.
    ///
    /// # Errors
    /// Returns an error if the allocation fails.
    pub fn parameters(self: *Layer, allocator: std.mem.Allocator) !std.ArrayList(*Tensor) {
        return switch (self.*) {
            .Linear => |linear| try linear.parameters(allocator),
        };
    }

    /// Resets the gradients of the layer's parameters to zero.
    pub fn zero_grad(self: *Layer) void {
        switch (self.*) {
            .Linear => |linear| linear.zero_grad(),
        }
    }

    /// Deinitializes the layer, freeing all associated resources.
    pub fn deinit(self: *Layer) void {
        switch (self.*) {
            .Linear => |linear| linear.deinit(),
        }
    }
};

/// A structure representing a fully connected linear layer.
///
/// This struct contains the following fields:
/// - `weights`: The weight tensor.
/// - `biases`: The bias tensor.
/// - `input_size`: The size of the input dimension.
/// - `output_size`: The size of the output dimension.
/// - `allocator`: The memory allocator used for the layer.
/// - `activation`: The activation function to apply (optional).
/// - `cache`: A list of cached tensors for intermediate computations.
pub const Linear = struct {
    weights: *Tensor,
    biases: *Tensor,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,
    activation: ?F.Activation,
    cache: std.ArrayList(?*Tensor), // Use `?*Tensor` to allow null

    /// Initializes a new linear layer.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for the layer.
    /// - `input_size`: The size of the input dimension.
    /// - `output_size`: The size of the output dimension.
    /// - `activation`: The activation function to apply (optional).
    ///
    /// # Returns
    /// A new `Linear` instance.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, activation: ?F.Activation) !Linear {
        const weights_shape = &[_]usize{ output_size, input_size };
        const biases_shape = &[_]usize{ 1, output_size };
        const random_seed: u64 = 1234;

        const weights = try Tensor.from_value(allocator, weights_shape, dtypes.DataType.f32, 0.0, true);
        weights.data = try utils.xavier_initialization(allocator, weights_shape, dtypes.DataType.f32, random_seed);

        const biases = try Tensor.from_value(allocator, biases_shape, dtypes.DataType.f32, 0.0, true);
        var cache = std.ArrayList(?*Tensor).init(allocator);

        // 🚀 Precompute transposed weights and store in cache
        const weights_transposed = try weights.transpose();
        weights_transposed.label = try std.fmt.allocPrint(allocator, "Wt", .{});
        try cache.append(weights_transposed);

        // 🚀 Preallocate space for input and matmul_result (set to `null`)
        try cache.append(null);
        try cache.append(null);

        return Linear{
            .weights = weights,
            .biases = biases,
            .input_size = input_size,
            .output_size = output_size,
            .allocator = allocator,
            .activation = activation,
            .cache = cache,
        };
    }

    /// Performs the forward pass for the linear layer.
    ///
    /// # Parameters
    /// - `input`: The input tensor.
    ///
    /// # Returns
    /// A new tensor representing the output of the layer.
    ///
    /// # Errors
    /// Returns an error if the forward pass fails.
    pub fn forward(self: *Linear, input: *Tensor) !*Tensor {
        const weights_transposed = self.cache.items[0].?;

        const matmul_result = try input.matmul(weights_transposed);
        matmul_result.label = try std.fmt.allocPrint(self.allocator, "xWt", .{});

        var linear_output = try matmul_result.add(self.biases);
        linear_output.label = try std.fmt.allocPrint(self.allocator, "x.Wt+b", .{});

        // 🚀 Overwrite existing cache instead of appending
        self.cache.items[1] = input;
        self.cache.items[2] = matmul_result;

        // Apply activation function if provided
        if (self.activation) |activation| {
            const activated_output = try activation(linear_output);
            return activated_output;
        }

        return linear_output;
    }

    /// Retrieves the parameters of the linear layer.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for the parameters list.
    ///
    /// # Returns
    /// A list of tensors representing the layer's parameters.
    ///
    /// # Errors
    /// Returns an error if the allocation fails.
    pub fn parameters(self: *Linear, allocator: std.mem.Allocator) !std.ArrayList(*Tensor) {
        var params = std.ArrayList(*Tensor).init(allocator);
        try params.append(self.weights);
        try params.append(self.biases);
        return params;
    }

    /// Resets the gradients of the layer's parameters to zero.
    pub fn zero_grad(self: *Linear) void {
        if (self.weights.grad) |weights_grad| {
            try weights_grad.fill(0.0);
        }
        if (self.biases.grad) |biases_grad| {
            try biases_grad.fill(0.0);
        }
    }

    /// Deinitializes the linear layer, freeing all associated resources.
    pub fn deinit(self: *Linear) void {
        // std.debug.print("Deallocating tensor: {s}\n", .{self.weights.label});
        self.weights.deinit();
        // std.debug.print("Deallocating tensor: {s}\n", .{self.biases.label});
        self.biases.deinit();

        // 🚀 Free cached tensors
        for (self.cache.items) |tensor| {
            if (tensor) |t| {
                //std.debug.print("Deallocating cached tensor: {s}\n", .{t.label});
                t.deinit();
            }
        }
        self.cache.deinit();
    }
};

// // ======================
// // Tests for loss structs
// // ======================
// const testing = std.testing;
// const expect = std.testing.expect;

// test "MSE Loss - Forward Pass" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensors
//     const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 4.0, 1.0, 2.0, -1.0 }, true);
//     defer pred.deinit();

//     const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
//     defer target.deinit();

//     // Compute MSE loss
//     const mse = try mse_loss.forward(pred, target);
//     defer mse.deinit();

//     const mse_value = try mse.get(&[_]usize{0});
//     try std.testing.expectEqual(9.0, mse_value.f32);
// }

// test "MSE Loss - Forward Pass with Different Values" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensors
//     const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 4.0, 1.0, 2.0, -1.0 }, true);
//     defer pred.deinit();

//     const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
//     defer target.deinit();

//     // Compute MSE loss
//     const mse = try mse_loss.forward(pred, target);
//     defer mse.deinit();

//     const mse_value = try mse.get(&[_]usize{0});

//     // Expected MSE loss
//     const expected_mse: f32 = (std.math.pow(f32, 4.0 - 1.0, 2) +
//         std.math.pow(f32, 1.0 - 2.0, 2) +
//         std.math.pow(f32, 2.0 - 3.0, 2) +
//         std.math.pow(f32, -1.0 - 4.0, 2)) / 4.0;

//     try std.testing.expectApproxEqAbs(mse_value.f32, expected_mse, 1e-6);
// }

// test "MSE Loss - Backward Pass" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensors
//     const pred = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 4.0, 1.0, 2.0, -1.0 }, true);
//     pred.label = try std.fmt.allocPrint(allocator, "pred", .{});
//     defer pred.deinit();

//     const target = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, false);
//     target.label = try std.fmt.allocPrint(allocator, "target", .{});
//     defer target.deinit();

//     // Compute gradient
//     const mse = try mse_loss.forward(pred, target);
//     defer mse.deinit();

//     try mse.backward(null);

//     // Expected gradient
//     const expected_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1.5, -0.5, -0.5, -2.5 });
//     defer expected_grad.deinit();

//     if (pred.grad) |computed_grad| {
//         const total_size = utils.compute_size(computed_grad.shape);
//         try computed_grad.print();

//         for (0..total_size) |i| {
//             const row = i / computed_grad.shape[1];
//             const col = i % computed_grad.shape[1];
//             const expected_value = try expected_grad.get(&[_]usize{ row, col });
//             const computed_value = try computed_grad.get(&[_]usize{ row, col });
//             try std.testing.expectEqual(expected_value.f32, computed_value.f32);
//         }
//     }
// }

// test "MSE Loss - Backward Pass with Different Batch Size" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensors
//     const pred = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, .f32, &[_]f32{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }, true);
//     defer pred.deinit();

//     const target = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, .f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, true);
//     defer target.deinit();

//     // Compute gradient
//     const mse = try mse_loss.forward(pred, target);
//     defer mse.deinit();

//     try mse.backward(null);

//     // Expected gradient
//     const expected_grad = try Tensor.from_data(allocator, &[_]usize{ 3, 2 }, .f32, &[_]f32{ 0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333 }, true);
//     defer expected_grad.deinit();

//     if (pred.grad) |computed_grad| {
//         const total_size = utils.compute_size(computed_grad.shape);
//         try computed_grad.print();

//         for (0..total_size) |i| {
//             const row = i / computed_grad.shape[1];
//             const col = i % computed_grad.shape[1];
//             const expected_value = try expected_grad.get(&[_]usize{ row, col });
//             const computed_value = try computed_grad.get(&[_]usize{ row, col });

//             const is_equal = std.math.approxEqAbs(f32, expected_value.f32, computed_value.f32, 1e-6);
//             try std.testing.expect(is_equal);
//         }
//     }
// }

// // ======================
// // Tests for loss layers
// // ======================

// test "Linear forward pass without activation" {
//     const allocator = std.heap.page_allocator;
//     const input_size = 3;
//     const output_size = 2;

//     // Initialize the linear layer without an activation function
//     var linear_layer = try Linear.init(allocator, input_size, output_size, null);
//     defer linear_layer.deinit();

//     // Create an input tensor
//     const input = try Tensor.from_value(allocator, &[_]usize{ 1, input_size }, dtypes.DataType.f32, 1.0, false);
//     defer input.deinit();

//     // Perform the forward pass
//     const output = try linear_layer.forward(input);
//     defer output.deinit();

//     // Check the output shape
//     try expect(output.shape[0] == 1);
//     try expect(output.shape[1] == output_size);

//     // Verify the output values
//     for (0..output_size) |i| {
//         const value = try output.data.get(&[_]usize{ 0, i });
//         const val_f32 = value.f32;

//         // The output should be the result of the linear transformation (no activation applied)
//         // Since weights and biases are initialized to 0, the output should be 0 + 0 = 0
//         try expect(val_f32 == 0.0);
//     }
// }

// test "Linear forward pass with ReLU activation" {
//     const allocator = std.heap.page_allocator;
//     const input_size = 3;
//     const output_size = 2;

//     var linear_layer = try Linear.init(allocator, input_size, output_size, F.relu);
//     defer linear_layer.deinit();

//     const input = try Tensor.from_value(allocator, &[_]usize{ 1, input_size }, dtypes.DataType.f32, -1.0, false);
//     defer input.deinit();

//     const output = try linear_layer.forward(input);
//     defer output.deinit();

//     try expect(output.shape[0] == 1);
//     try expect(output.shape[1] == output_size);

//     for (0..output_size) |i| {
//         const value = try output.data.get(&[_]usize{ 0, i });
//         const val_f32 = value.f32;
//         try expect(val_f32 == 0.0); // ReLU should zero out negative values
//     }
// }

// test "Linear forward pass with Tanh activation" {
//     const allocator = std.heap.page_allocator;
//     const input_size = 3;
//     const output_size = 2;

//     var linear_layer = try Linear.init(allocator, input_size, output_size, F.tanh);
//     defer linear_layer.deinit();

//     const in_val: f32 = 1.0;
//     const input = try Tensor.from_value(allocator, &[_]usize{ 1, input_size }, dtypes.DataType.f32, in_val, false);
//     defer input.deinit();

//     const output = try linear_layer.forward(input);
//     defer output.deinit();

//     try expect(output.shape[0] == 1);
//     try expect(output.shape[1] == output_size);
//     const out_val: f32 = 0.0;

//     for (0..output_size) |i| {
//         const value = try output.data.get(&[_]usize{ 0, i });
//         const val_f32 = value.f32;
//         try expect(val_f32 == std.math.tanh(out_val)); // Tanh of 1.0
//     }
// }

// test "Linear forward pass with Softmax activation" {
//     const allocator = std.heap.page_allocator;
//     const input_size = 3;
//     const output_size = 3;

//     var linear_layer = try Linear.init(allocator, input_size, output_size, F.softmax);
//     defer linear_layer.deinit();

//     const input = try Tensor.from_value(allocator, &[_]usize{ 1, input_size }, dtypes.DataType.f32, 1.0, false);
//     defer input.deinit();

//     const output = try linear_layer.forward(input);
//     defer output.deinit();

//     try expect(output.shape[0] == 1);
//     try expect(output.shape[1] == output_size);

//     // Ensure the output sums to 1 (property of softmax)
//     var sum: f32 = 0.0;
//     for (0..output_size) |i| {
//         const value = try output.data.get(&[_]usize{ 0, i });
//         const val_f32 = value.f32;
//         sum += val_f32;
//     }

//     const is_equal = std.math.approxEqAbs(f32, sum, 1.0, 1e-6);
//     try expect(is_equal);
// }
