const std = @import("std");
const zorch = @import("zorch.zig");

const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;

const Tensor = zorch.Tensor;
const NdArray = zorch.NdArray;

const datasets = zorch.datasets;
const autograd = zorch.autograd;

const Layer = zorch.nn.Layer;
const Linear = zorch.nn.Linear;
const SGD = zorch.optim.SGD;
const MSE = zorch.nn.MSELoss;
const F = zorch.functional;

pub const Network = struct {
    allocator: std.mem.Allocator,
    layers: std.ArrayList(*Layer),
    optimizer: SGD,

    pub fn init(allocator: std.mem.Allocator, layers_list: []*Layer, learning_rate: f32, momentum: f32) !Network {
        var layers = std.ArrayList(*Layer).init(allocator);
        for (layers_list) |layer| {
            try layers.append(layer);
        }

        _ = momentum;
        const optimizer = try SGD.init(allocator, layers, learning_rate);

        return Network{
            .allocator = allocator,
            .layers = layers,
            .optimizer = optimizer,
        };
    }

    pub fn zero_grad(self: *Network) void {
        for (self.layers.items) |layer| {
            layer.zero_grad();
        }
    }

    pub fn parameters(self: *Network) !std.ArrayList(*Tensor) {
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

    pub fn deinit(self: *Network) void {
        for (self.layers.items) |layer| {
            layer.deinit();
        }
        self.layers.deinit();
        self.optimizer.deinit();
    }

    pub fn forward(self: *Network, input: *Tensor) !*Tensor {
        var output = input;
        for (self.layers.items) |layer| {
            output = try layer.forward(output);
        }
        return output;
    }

    pub fn train(self: *Network, inputs: *Tensor, targets: *Tensor, n_epochs: usize) ![]f32 {
        var losses = try self.allocator.alloc(f32, n_epochs);
        errdefer self.allocator.free(losses);

        const num_samples = inputs.shape[0];

        for (0..n_epochs) |epoch| {
            var total_loss: f32 = 0.0;
            std.debug.print("***\n", .{});
            std.debug.print("**\n", .{});
            std.debug.print("*\n", .{});

            for (0..num_samples) |i| {
                const input_tensor = try inputs.slice(0, i, i + 1);
                defer input_tensor.deinit();

                const target_tensor = try targets.slice(0, i, i + 1);
                defer target_tensor.deinit();

                // Forward pass
                const output = try self.forward(input_tensor);
                defer output.deinit(); // Deallocate output after backward pass

                // Compute loss
                const loss = try MSE.forward(output, target_tensor);

                std.debug.print(" * Target     :  ", .{});
                try target_tensor.info();
                std.debug.print(" * Prediction :  ", .{});
                try output.info();
                std.debug.print(" -> Loss      :  ", .{});
                try loss.print();

                // Get loss value
                const val = try loss.get(&[_]usize{0});
                total_loss += val.f32;
                // std.debug.print("-> loss val   : {any}\n", .{val.f32});
                // std.debug.print("-> total loss : {any}\n", .{total_loss});

                // Backward pass
                std.debug.print("--------------------------\n", .{});
                std.debug.print("LOSS BACKWARD\n", .{});
                std.debug.print("--------------------------\n", .{});
                try loss.backward(null);

                const params = try self.parameters();

                // std.debug.print("--------------------------\n", .{});
                // std.debug.print("OPTIMIZER STEP\n", .{});
                // std.debug.print("--------------------------\n", .{});
                // for (params.items) |param| {
                //     try param.info();
                // }

                try self.optimizer.step();

                // for (params.items) |param| {
                //     try param.info();
                // }

                // Zero gradients
                self.zero_grad();
                params.deinit();
                std.debug.print("**********************************\n", .{});
                defer loss.deinit(); // Deallocate loss after backward pass

            }

            losses[epoch] = total_loss / @as(f32, @floatFromInt(num_samples));
            std.debug.print("**********************************\n", .{});
            std.debug.print(" Epoch {}: Loss = {}  \n", .{ epoch, losses[epoch] });
            std.debug.print("**********************************\n", .{});
        }

        return losses;
    }
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create XOR dataset as Tensors
    const dataset = try datasets.create_xor_dataset(allocator);
    defer {
        dataset.inputs.deinit();
        dataset.outputs.deinit();
    }

    // Create layers
    const input_size = 2;
    const hidden_size = 3;
    const output_size = 1;

    var linear1 = try Linear.init(allocator, input_size, hidden_size, F.relu);
    var linear2 = try Linear.init(allocator, hidden_size, output_size, F.relu);
    linear1.weights.label = try std.fmt.allocPrint(allocator, "l1.w", .{});
    linear1.biases.label = try std.fmt.allocPrint(allocator, "l1.b", .{});
    linear2.weights.label = try std.fmt.allocPrint(allocator, "l2.w", .{});
    linear2.biases.label = try std.fmt.allocPrint(allocator, "l2.b", .{});

    std.debug.print("l1.w: {any}\n", .{linear1.weights.shape});
    std.debug.print("l1.b: {any}\n", .{linear1.biases.shape});
    std.debug.print("l2.w: {any}\n", .{linear2.weights.shape});
    std.debug.print("l2.b: {any}\n", .{linear2.biases.shape});

    // Wrap layers in the `Layer` union
    var layer1 = Layer{ .Linear = &linear1 };
    var layer2 = Layer{ .Linear = &linear2 };

    // Create a mutable array of layers
    var layers = [_]*Layer{ &layer1, &layer2 };

    // Initialize the network
    var network = try Network.init(allocator, layers[0..], 0.1, 0.9);
    defer network.deinit();

    // Train the network
    std.debug.print("======== \n", .{});
    std.debug.print("TRAINING \n", .{});
    std.debug.print("======== \n", .{});
    const n_epochs = 7;
    const losses = try network.train(dataset.inputs, dataset.outputs, n_epochs);
    defer allocator.free(losses);

    // Print losses
    for (losses, 0..) |loss, epoch| {
        std.debug.print("Epoch {}: Loss = {d:.9}\n", .{ epoch, loss });
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

        std.debug.print("+---------- INPUT :  ", .{});
        try input_tensor.info();

        std.debug.print("└── PRED  :  ", .{});
        try output.info();
    }
}
