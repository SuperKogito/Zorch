const std = @import("std");
const zorch = @import("zorch.zig");

const dtypes = zorch.dtypes;
const Tensor = zorch.Tensor;

/// Creates a dataset for the XOR problem.
///
/// The XOR problem is a classic binary classification task where the inputs are pairs of binary values,
/// and the output is the logical XOR of the inputs. This function generates a dataset with 4 samples:
/// - Inputs: `[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]`
/// - Outputs: `[[0.0], [1.0], [1.0], [0.0]]`
///
/// # Parameters
/// - `allocator`: The memory allocator to use for tensor creation.
///
/// # Returns
/// A struct containing two tensors:
/// - `inputs`: A tensor of shape `[4, 2]` containing the input data.
/// - `outputs`: A tensor of shape `[4, 1]` containing the corresponding output labels.
///
/// # Errors
/// Returns an error if tensor creation fails.
///
/// # Example
/// ```zig
/// const allocator = std.heap.page_allocator;
/// const dataset = try create_xor_dataset(allocator);
/// defer dataset.inputs.deinit();
/// defer dataset.outputs.deinit();
///
/// // Use the dataset for training or testing
/// try dataset.inputs.print();
/// try dataset.outputs.print();
/// ```
pub fn create_xor_dataset(allocator: std.mem.Allocator) !struct { inputs: *Tensor, outputs: *Tensor } {
    const input_data = [_]f32{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
    const input_shape = &[_]usize{ 4, 2 };
    const xor_inputs = try Tensor.from_data(allocator, input_shape, .f32, &input_data, true);

    const output_data = [_]f32{ 0.0, 1.0, 1.0, 0.0 };
    const output_shape = &[_]usize{ 4, 1 };
    const xor_outputs = try Tensor.from_data(allocator, output_shape, .f32, &output_data, false);

    return .{
        .inputs = xor_inputs,
        .outputs = xor_outputs,
    };
}
