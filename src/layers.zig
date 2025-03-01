const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const dtypes = @import("math/ndarray/dtypes.zig"); // Corrected path
const activations = @import("activations.zig");
const Activation = activations.Activation;

// Layer type
pub const LayerType = enum { Linear };

// Layer union
pub const Layer = union(LayerType) {
    Linear: *Linear,

    pub fn forward(self: *Layer, input: *Tensor) !*Tensor {
        return switch (self.*) {
            .Linear => |linear| linear.forward(input),
        };
    }

    pub fn parameters(self: *Layer, allocator: std.mem.Allocator) !std.ArrayList(*Tensor) {
        return switch (self.*) {
            .Linear => |linear| try linear.parameters(allocator),
        };
    }

    pub fn zero_grad(self: *Layer) void {
        switch (self.*) {
            .Linear => |linear| linear.zero_grad(),
        }
    }

    pub fn deinit(self: *Layer) void {
        switch (self.*) {
            .Linear => |linear| linear.deinit(),
        }
    }
};

// Linear layer
pub const Linear = struct {
    weights: *Tensor,
    biases: *Tensor,
    input_size: usize,
    output_size: usize,
    allocator: std.mem.Allocator,
    activation: ?Activation, // Optional activation function

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, activation: ?Activation) !Linear {
        // Define sizes for weights and biases
        const weights_shape = &[_]usize{ output_size, input_size }; // Shape: [output_size, input_size]
        const biases_shape = &[_]usize{ 1, output_size }; // Shape: [1, output_size]

        // Create weights and biases tensors
        const weights = try Tensor.from_value(allocator, weights_shape, dtypes.DataType.f32, 0.0, true);
        const biases = try Tensor.from_value(allocator, biases_shape, dtypes.DataType.f32, 0.0, true);

        // Label the tensors for debugging
        weights.label = try std.fmt.allocPrint(allocator, "ll.w", .{});
        biases.label = try std.fmt.allocPrint(allocator, "ll.b", .{});

        return Linear{
            .weights = weights,
            .biases = biases,
            .input_size = input_size,
            .output_size = output_size,
            .allocator = allocator,
            .activation = activation,
        };
    }

    pub fn parameters(self: *Linear, allocator: std.mem.Allocator) !std.ArrayList(*Tensor) {
        var params = std.ArrayList(*Tensor).init(allocator);
        try params.append(self.weights);
        try params.append(self.biases);
        return params;
    }

    pub fn forward(self: *Linear, input: *Tensor) !*Tensor {
        const weights_transposed = try self.weights.transpose();
        const matmul_result = try input.matmul(weights_transposed);
        var linear_output = try matmul_result.add(self.biases);
        weights_transposed.label = try std.fmt.allocPrint(self.allocator, "Wt", .{});
        matmul_result.label = try std.fmt.allocPrint(self.allocator, "x.Wt", .{});
        linear_output.label = try std.fmt.allocPrint(self.allocator, "x.Wt+b", .{});

        // Apply activation function if provided
        if (self.activation) |activation| {
            const activated_output = try activation(linear_output);
            linear_output.deinit(); // Clean up intermediate tensor
            activated_output.label = try std.fmt.allocPrint(self.allocator, "relu(x.Wt+b)", .{});
            return activated_output;
        }

        return linear_output;
    }

    pub fn zero_grad(self: *Linear) void {
        if (self.weights.grad) |weights_grad| {
            try weights_grad.fill(0.0);
        }
        if (self.biases.grad) |biases_grad| {
            try biases_grad.fill(0.0);
        }
    }

    pub fn deinit(self: *Linear) void {
        self.weights.deinit();
        self.biases.deinit();
    }
};

// // Tests
// const expect = std.testing.expect;

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

//     var linear_layer = try Linear.init(allocator, input_size, output_size, activations.relu);
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

//     var linear_layer = try Linear.init(allocator, input_size, output_size, activations.tanh);
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

//     var linear_layer = try Linear.init(allocator, input_size, output_size, activations.softmax);
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
