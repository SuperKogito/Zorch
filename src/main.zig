const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const Layer = @import("layers.zig").Layer;
const Linear = @import("layers.zig").Linear;
const activations = @import("activations.zig");
const Model = @import("sequential.zig").Model;
const SGD = @import("optimizers.zig").SGD;
const MSE = @import("losses.zig").MSE;
const dtypes = @import("math/ndarray/dtypes.zig");
const TensorError = @import("errors.zig").TensorError;
const NdArray = @import("math/ndarray/ndarray.zig").NdArray;

pub const Network = struct {
    model: Model, // Use Model instead of Sequential
    optimizer: SGD, // Optimizer for updating parameters
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, layers_list: []*Layer, learning_rate: f32, momentum: f32) !Network {
        var model = try Model.init(allocator);
        errdefer model.deinit();

        // Add all layers to the model
        for (layers_list) |layer| {
            try model.add_layer(layer);
        }

        std.debug.print("{any}\n", .{model.layers.items});

        // Initialize the optimizer with a pointer to the model
        const optimizer = try SGD.init(allocator, &model, learning_rate);
        _ = momentum;
        return Network{
            .model = model, // Store a pointer to the model
            .optimizer = optimizer,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Network) void {
        self.optimizer.deinit();
        self.model.deinit(); // Deinitialize the model
    }
    pub fn train(self: *Network, inputs: *Tensor, targets: *Tensor, n_epochs: usize) ![]f32 {
        var losses = try self.allocator.alloc(f32, n_epochs);
        errdefer self.allocator.free(losses);

        const num_samples = inputs.shape[0];
        const in_grad = try NdArray.ones(self.allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32);

        for (0..n_epochs) |epoch| {
            var total_loss: f32 = 0.0;
            std.debug.print("Epoch: {}\n", .{epoch});

            for (2..num_samples) |i| {
                // Slice the input and target tensors for the current sample
                const input_tensor = try inputs.slice(0, i, i + 1);
                defer input_tensor.deinit();

                const target_tensor = try targets.slice(0, i, i + 1);
                defer target_tensor.deinit();

                // std.debug.print("X: ", .{});
                // try input_tensor.info();
                // input_tensor.label = "x";
                // std.debug.print("Y: ", .{});
                // try target_tensor.info();
                // target_tensor.label = "y";

                // Forward pass
                const output = try self.model.forward(input_tensor);
                defer output.deinit();
                // std.debug.print("y: ", .{});
                // try output.info();
                // std.debug.print("----\n", .{});
                std.debug.print("Layers Count when accessed via Model    : {}\n", .{self.model.layers.items.len});
                std.debug.print("Layers Count when accessed via Optimizer: {}\n", .{self.optimizer.model.layers.items.len});

                // Compute loss
                const loss = try MSE.forward(output, target_tensor);
                const val = try loss.get(&[_]usize{0});
                total_loss += val.f32; // Assuming loss is a scalar tensor
                std.debug.print("total_loss =  {any}\n", .{total_loss});

                // Backward pass (gradient computation)
                try loss.backward(in_grad);
                std.debug.print("Layers Count: {}\n", .{self.model.layers.items.len});

                // Update parameters
                try self.optimizer.step();

                // Zero gradients
                self.model.zero_grad();
            }
            std.debug.print("====== \n", .{});

            // Record average loss for this epoch
            losses[epoch] = total_loss / @as(f32, @floatFromInt(num_samples));
        }

        return losses;
    }
    pub fn forward(self: *Network, input_tensor: *Tensor) TensorError!*Tensor {

        // Forward pass
        const output = try self.model.forward(input_tensor);
        return output;
    }
};

pub fn create_xor_dataset(allocator: std.mem.Allocator) !struct { inputs: *Tensor, outputs: *Tensor } {
    const input_data = [_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    const input_shape = &[_]usize{ 4, 2 };
    const xor_inputs = try Tensor.from_data(allocator, input_shape, dtypes.DataType.f32, &input_data, true);

    const output_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };
    const output_shape = &[_]usize{ 4, 1 };
    const xor_outputs = try Tensor.from_data(allocator, output_shape, dtypes.DataType.f32, &output_data, false);

    return .{
        .inputs = xor_inputs,
        .outputs = xor_outputs,
    };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create XOR dataset as Tensors
    const dataset = try create_xor_dataset(allocator);
    defer {
        dataset.inputs.deinit();
        dataset.outputs.deinit();
    }

    // Create layers
    const input_size = 2;
    const hidden_size = 4;
    const output_size = 1;

    var linear1 = try Linear.init(allocator, input_size, hidden_size, activations.relu);
    var linear2 = try Linear.init(allocator, hidden_size, output_size, null);

    // Wrap layers in the `Layer` union
    var layer1 = Layer{ .Linear = &linear1 };
    var layer2 = Layer{ .Linear = &linear2 };

    // Create a mutable array of layers
    var layers = [_]*Layer{ &layer1, &layer2 };

    // Initialize the network
    var network = try Network.init(allocator, layers[0..], 0.01, 0.9);
    defer network.deinit();
    // std.debug.print("{any}\n", .{network.optimizer.model.layers.items});

    // Train the network
    std.debug.print("======== \n", .{});
    std.debug.print("TRAINING \n", .{});
    std.debug.print("======== \n", .{});
    const n_epochs = 10;
    const losses = try network.train(dataset.inputs, dataset.outputs, n_epochs);
    defer allocator.free(losses);

    // Print losses
    for (losses, 0..) |loss, epoch| {
        std.debug.print("Epoch {}: Loss = {}\n", .{ epoch, loss });
    }

    std.debug.print("======= \n", .{});
    std.debug.print("TESTING \n", .{});
    std.debug.print("======= \n", .{});
    // Test the trained network
    for (0..dataset.inputs.shape[0]) |i| {
        // Corrected slice call: pass usize values directly
        const input_tensor = try dataset.inputs.slice(0, i, i + 1);
        defer input_tensor.deinit();

        const output = try network.forward(input_tensor);
        defer output.deinit();

        std.debug.print("Input: {any}, Output: {any}\n", .{ input_tensor.data, output.data });
    }
}
