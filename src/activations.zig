const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

// Define activation function type
pub const Activation = *const fn (input: *Tensor) anyerror!*Tensor;

// Activation functions
pub fn relu(input: *Tensor) !*Tensor {
    return input.relu();
}

pub fn tanh(input: *Tensor) !*Tensor {
    return input.tanh();
}

pub fn softmax(input: *Tensor) !*Tensor {
    return input.softmax();
}
