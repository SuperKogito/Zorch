// optimizers.zig
const std = @import("std");
const zorch = @import("zorch.zig");

const ops = zorch.ops;
const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;

const Tensor = zorch.Tensor;
const NdArray = zorch.NdArray;
pub const DataType = dtypes.DataType;
pub const NumericUnion = dtypes.NumericUnion;

const Model = zorch.nn.Model;
const Layer = zorch.nn.Layer;
const Linear = zorch.nn.Linear;

/// A structure representing the Stochastic Gradient Descent (SGD) optimizer.
///
/// This struct contains the following fields:
/// - `allocator`: The memory allocator used for optimizer operations.
/// - `layers`: A list of layers to optimize.
/// - `learning_rate`: The learning rate for the optimizer.
pub const SGD = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(*Layer),
    learning_rate: f32,

    /// Initializes a new SGD optimizer.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for optimizer operations.
    /// - `layers`: A list of layers to optimize.
    /// - `learning_rate`: The learning rate for the optimizer.
    ///
    /// # Returns
    /// A new `SGD` instance.
    ///
    /// # Errors
    /// Returns an error if initialization fails.
    pub fn init(allocator: std.mem.Allocator, layers: std.ArrayList(*Layer), learning_rate: f32) !SGD {
        std.debug.print("Inside optimizer.init()\n", .{});
        std.debug.print("Layers Count: {}\n", .{layers.items.len});

        return SGD{
            .allocator = allocator,
            .layers = layers,
            .learning_rate = learning_rate,
        };
    }

    /// Performs a single optimization step using SGD.
    ///
    /// This function updates the parameters of all layers in the optimizer by subtracting
    /// the gradient scaled by the learning rate.
    ///
    /// # Errors
    /// Returns an error if the optimization step fails.
    pub fn step(self: *SGD) !void {
        for (self.layers.items) |layer| {
            const params = try layer.parameters(self.allocator);
            for (params.items) |param| {
                if (param.grad) |grad| {
                    // Standard SGD update: param -= learning_rate * grad
                    const lr_times_grad = try grad.mul_scalar(self.learning_rate, false);
                    errdefer lr_times_grad.deinit();

                    _ = try param.data.sub(lr_times_grad, true);
                }
            }
        }
    }

    /// Resets the gradients of all parameters in the optimizer to zero.
    pub fn zero_grad(self: *SGD) void {
        for (self.layers.items) |layer| {
            switch (layer.*) {
                .Linear => |linear| {
                    linear.zero_grad();
                },
            }
        }
    }

    /// Deinitializes the SGD optimizer, freeing all associated resources.
    pub fn deinit(self: *SGD) void {
        self.layers.deinit();
    }
};

// pub const SGD = struct {
//     model: *Model,
//     learning_rate: f32,

//     pub fn init(allocator: std.mem.Allocator, model: *Model, learning_rate: f32) !SGD {
//         _ = allocator;
//         std.debug.print("Inside optimizer.init()\n", .{});

//         std.debug.print("Layers Count: {}\n", .{model.layers.items.len});

//         return SGD{
//             .model = model,
//             .learning_rate = learning_rate,
//         };
//     }

//     pub fn step(self: *SGD) !void {
//         std.debug.print("Inside optimizer.step()\n", .{});
//         const params = try self.model.parameters();
//         std.debug.print("Layers Count: {}\n", .{self.model.layers.items.len});

//         for (params.items) |param| {
//             if (param.grad) |grad| {
//                 // Standard SGD update: param -= learning_rate * grad
//                 const lr_times_grad = try grad.mul_scalar(self.learning_rate, false);
//                 _ = try param.data.sub(lr_times_grad, true);
//             } else {
//                 std.debug.print("Gradient is null for parameter\n", .{});
//             }
//         }
//         std.debug.print("Out of optimizer.step()\n", .{});
//     }
//     pub fn zero_grad(self: *SGD) void {
//         const params = self.model.parameters() catch return; // Handle potential error

//         for (params.items) |param| {
//             if (param.grad) |grad| {
//                 grad.fill(0.0); // Zero out the gradient
//             }
//         }
//     }

//     pub fn deinit(self: *SGD) void {
//         // Clean up velocities
//         self.model.deinit();
//     }
// };

// pub const SGD = struct {
//     model: *Model,
//     learning_rate: f32,
//     momentum: f32,
//     velocities: std.ArrayList(*NdArray),

//     pub fn init(allocator: std.mem.Allocator, model: *Model, learning_rate: f32, momentum: f32) !SGD {
//         var velocities = std.ArrayList(*NdArray).init(allocator);

//         // Initialize velocities for each parameter in the model
//         const params = try model.parameters();
//         for (params.items) |param| {
//             const velocity = try NdArray.from_value(allocator, param.shape, dtypes.DataType.f32, 0.0);
//             try velocities.append(velocity);
//         }

//         return SGD{
//             .model = model,
//             .learning_rate = learning_rate,
//             .momentum = momentum,
//             .velocities = velocities,
//         };
//     }

//     pub fn step(self: *SGD) !void {
//         const params = try self.model.parameters();
//         for (params.items, 0..) |param, i| {
//             if (param.grad) |grad| {
//                 // Update velocity if momentum is enabled
//                 if (self.momentum != 0.0) {
//                     const velocity = self.velocities.items[i];
//                     // velocity = momentum * velocity
//                     _ = try velocity.mul_scalar(self.momentum, true);
//                     // velocity += grad
//                     _ = try velocity.add(grad, true);
//                     // param -= learning_rate * velocity
//                     const lr_times_grad = try grad.mul_scalar(self.learning_rate, false);
//                     _ = try param.data.sub(lr_times_grad, true);
//                 } else {
//                     // Standard SGD update: param -= learning_rate * grad
//                     const lr_times_grad = try grad.mul_scalar(self.learning_rate, false);
//                     _ = try param.data.sub(lr_times_grad, true);
//                 }
//             }
//         }
//     }

//     pub fn zero_grad(self: *SGD) void {
//         const params = self.model.parameters() catch return; // Handle potential error

//         for (params.items) |param| {
//             if (param.grad) |grad| {
//                 grad.fill(0.0); // Zero out the gradient
//             }
//         }
//     }

//     pub fn deinit(self: *SGD) void {
//         // Clean up velocities
//         for (self.velocities.items) |velocity| {
//             velocity.deinit();
//         }
//         self.velocities.deinit();
//     }
// };

// // test
// const expect = std.testing.expect;

// test "SGD optimizer with Model" {
//     const allocator = std.testing.allocator;

//     // Initialize a simple model
//     var model = try Model.init(allocator);
//     defer model.deinit();

//     // Create a Linear layer
//     const input_size: usize = 5;
//     const output_size: usize = 2;
//     var linear_layer = try Linear.init(allocator, input_size, output_size);
//     defer linear_layer.deinit();

//     // Wrap the Linear layer in the Layer union
//     var layer = Layer{ .Linear = &linear_layer };

//     // Add the Linear layer to the model
//     try model.addLayer(&layer);

//     // Initialize the SGD optimizer
//     var sgd = try SGD.init(allocator, &model, 0.01, 0.9);
//     defer sgd.deinit();

//     // Create a dummy input tensor
//     const input_shape = &[_]usize{ 1, input_size };
//     const input = try Tensor.from_value(allocator, input_shape, dtypes.DataType.f32, 1.0, true);
//     defer input.deinit();

//     // Perform a forward pass
//     const output = try model.forward(input);
//     defer output.deinit();

//     // Compute gradients (dummy operation)
//     const weights_grad = try NdArray.from_value(allocator, &[_]usize{ input_size, output_size }, dtypes.DataType.f32, 3);
//     defer weights_grad.deinit();

//     const biases_grad = try NdArray.from_value(allocator, &[_]usize{ 1, output_size }, dtypes.DataType.f32, 3);
//     defer biases_grad.deinit();

//     // Set gradients for the Linear layer
//     linear_layer.weights.grad = weights_grad;
//     linear_layer.biases.grad = biases_grad;

//     // Update parameters using SGD
//     try sgd.step();

//     // Check if the parameters have been updated correctly
//     const params = try model.parameters();
//     defer params.deinit();

//     for (params.items) |param| {
//         const expected_value = 0.0 - 0.01 * 3; // param -= learning_rate * grad
//         const computed_value = try param.data.get(&[_]usize{ 0, 0 });
//         try expect(computed_value.f32 == expected_value);
//     }
// }
