const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Layer = @import("layers.zig").Layer;
const Linear = @import("layers.zig").Linear;
const dtypes = @import("math/ndarray/dtypes.zig");

pub const Model = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(*Layer),

    pub fn init(allocator: std.mem.Allocator) !Model {
        const model = Model{
            .allocator = allocator,
            .layers = std.ArrayList(*Layer).init(allocator),
        };
        return model;
    }

    pub fn add_layer(self: *Model, layer: *Layer) !void {
        try self.layers.append(layer);
    }

    pub fn forward(self: *Model, input: *Tensor) !*Tensor {
        var output = input;
        for (self.layers.items) |layer| {
            output = try layer.forward(output);
        }
        return output;
    }

    pub fn zero_grad(self: *Model) void {
        for (self.layers.items) |layer| {
            layer.zero_grad();
        }
    }

    pub fn parameters(self: *Model) !std.ArrayList(*Tensor) {
        var params = std.ArrayList(*Tensor).init(self.allocator);
        for (self.layers.items) |layer| {
            switch (layer.*) {
                .Linear => |linear| {
                    try params.append(linear.weights);
                    try params.append(linear.biases);
                },
            }
        }
        return params;
    }

    pub fn deinit(self: *Model) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit();
    }
};

const expect = std.testing.expect;

test "Model and Layer functionality" {
    const allocator = std.testing.allocator;

    // Initialize a model
    var model = try Model.init(allocator);
    defer model.deinit();

    // Create a Linear layer
    const input_size: usize = 2;
    const output_size: usize = 2;
    var linear_layer = try Linear.init(allocator, input_size, output_size, null);

    // Wrap the Linear layer in a Layer union
    var linear_layer_wrapper = Layer{ .Linear = &linear_layer };
    defer linear_layer_wrapper.deinit();

    // Add the Linear layer to the model
    try model.add_layer(&linear_layer_wrapper);

    // Create a dummy input tensor
    const input_shape = &[_]usize{ 1, input_size };
    const input = try Tensor.from_value(allocator, input_shape, dtypes.DataType.f32, 1.0, true);
    defer input.deinit();

    // Perform a forward pass
    const output = try model.forward(input);
    defer output.deinit();

    // Check the output shape
    try expect(output.shape[0] == 1);
    try expect(output.shape[1] == output_size);

    // Check the parameters
    const params = try model.parameters();
    defer params.deinit();

    // Linear layer has weights and biases
    try expect(params.items.len == 2);

    // Check that the weights and biases are initialized
    try expect(params.items[0].shape[0] == output_size);
    try expect(params.items[0].shape[1] == input_size);
    try expect(params.items[1].shape[0] == 1);
    try expect(params.items[1].shape[1] == output_size);

    // Zero out gradients
    model.zero_grad();

    // Check that gradients are zeroed out
    for (params.items) |param| {
        if (param.grad) |grad| {
            for (grad.data) |value| {
                try expect(value == 0.0);
            }
        }
    }
}
