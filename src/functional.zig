const std = @import("std");
const zorch = @import("zorch.zig");
const Tensor = zorch.Tensor;

/// A function pointer type representing an activation function.
///
/// This type defines a function that takes a tensor as input and returns a new tensor
/// after applying the activation function. Activation functions are used to introduce
/// non-linearity into neural networks.
pub const Activation = *const fn (input: *Tensor) anyerror!*Tensor;

/// Applies the Rectified Linear Unit (ReLU) activation function to the input tensor.
///
/// ReLU is defined as `max(0, x)`, where `x` is the input tensor. It sets all negative
/// values in the tensor to zero and leaves positive values unchanged.
///
/// # Parameters
/// - `input`: The input tensor.
///
/// # Returns
/// A new tensor with the ReLU activation applied.
///
/// # Errors
/// Returns an error if the operation fails.
pub fn relu(input: *Tensor) !*Tensor {
    return input.relu();
}

/// Applies the Hyperbolic Tangent (tanh) activation function to the input tensor.
///
/// The tanh function is defined as `(e^x - e^(-x)) / (e^x + e^(-x))`. It squashes the
/// input values to the range `[-1, 1]`.
///
/// # Parameters
/// - `input`: The input tensor.
///
/// # Returns
/// A new tensor with the tanh activation applied.
///
/// # Errors
/// Returns an error if the operation fails.
pub fn tanh(input: *Tensor) !*Tensor {
    return input.tanh();
}

/// Applies the Softmax activation function to the input tensor.
///
/// The Softmax function is defined as `exp(x_i) / sum(exp(x_j))` for all `j`. It converts
/// the input tensor into a probability distribution, where the values sum to 1.
///
/// # Parameters
/// - `input`: The input tensor.
///
/// # Returns
/// A new tensor with the Softmax activation applied.
///
/// # Errors
/// Returns an error if the operation fails.
pub fn softmax(input: *Tensor) !*Tensor {
    return input.softmax();
}
