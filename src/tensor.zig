const std = @import("std");
const zorch = @import("zorch.zig");

const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;
const NdArray = zorch.NdArray;
const autograd = zorch.autograd;
const TensorError = zorch.errors.TensorError;

/// A tensor structure representing multi-dimensional arrays with support for automatic differentiation.
///
/// This struct contains the following fields:
/// - `allocator`: The memory allocator used for tensor operations.
/// - `shape`: The shape of the tensor (dimensions).
/// - `dtype`: The data type of the tensor elements.
/// - `data`: The underlying `NdArray` storing the tensor's data.
/// - `parents`: A list of parent tensors used in the computation graph.
/// - `operation`: The operation that produced this tensor (if any).
/// - `label`: A label for the tensor (useful for debugging).
/// - `requires_grad`: Whether the tensor requires gradient computation.
/// - `grad`: The gradient tensor (if `requires_grad` is true).
pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []const usize,
    dtype: dtypes.DataType,
    data: *NdArray,
    parents: []const *Tensor,
    operation: ?*autograd.Operation,
    label: []const u8,
    requires_grad: bool,
    grad: ?*NdArray,

    /// Initializes a new tensor with the given parameters.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for tensor operations.
    /// - `shape`: The shape of the tensor.
    /// - `dtype`: The data type of the tensor elements.
    /// - `data`: The underlying `NdArray` storing the tensor's data.
    /// - `parents`: A list of parent tensors used in the computation graph.
    /// - `operation`: The operation that produced this tensor (if any).
    /// - `label`: A label for the tensor (useful for debugging).
    /// - `requires_grad`: Whether the tensor requires gradient computation.
    ///
    /// # Returns
    /// A pointer to the newly created tensor.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, data: *NdArray, parents: []const *Tensor, operation: ?*autograd.Operation, label: []const u8, requires_grad: bool) TensorError!*Tensor {
        const tensor = try allocator.create(Tensor);
        errdefer allocator.destroy(tensor);

        tensor.* = .{
            .allocator = allocator,
            .shape = try allocator.dupe(usize, shape),
            .dtype = dtype,
            .data = data,
            .parents = try allocator.dupe(*Tensor, parents),
            .operation = operation,
            .label = try allocator.dupe(u8, label),
            .requires_grad = requires_grad,
            .grad = null,
        };

        errdefer {
            allocator.free(tensor.shape);
            allocator.free(tensor.parents);
            allocator.free(tensor.label);
        }

        if (requires_grad) {
            tensor.grad = try NdArray.zeros(allocator, shape, .f32);
        }

        return tensor;
    }

    /// Deinitializes the tensor, freeing all associated resources.
    ///
    /// This function frees the shape, label, parents, gradient, and underlying data.
    pub fn deinit(self: *Tensor) void {
        // Free the shape array
        if (self.shape.len > 0) {
            self.allocator.free(self.shape);
        }

        // Free the label if it's not an empty string
        if (self.label.len > 0) {
            self.allocator.free(self.label);
        }

        // Free the parents array if it exists
        if (self.parents.len > 0) {
            self.allocator.free(self.parents);
        }

        // Free the gradient tensor if it exists
        if (self.grad) |grad| {
            grad.deinit();
        }

        // Free the data (NdArray)
        self.data.deinit();

        // Free the Tensor itself
        self.allocator.destroy(self);
    }

    /// Slices the tensor along a specified dimension.
    ///
    /// # Parameters
    /// - `dim`: The dimension to slice.
    /// - `start`: The starting index of the slice.
    /// - `end`: The ending index of the slice.
    ///
    /// # Returns
    /// A new tensor representing the sliced data.
    ///
    /// # Errors
    /// Returns an error if the slice indices are invalid or allocation fails.
    pub fn slice(self: *Tensor, dim: usize, start: usize, end: usize) !*Tensor {
        // Ensure that the slice indices are valid
        if (start >= end or end > self.shape[0]) {
            return error.InvalidSliceIndices;
        }

        // Slice the NdArray data (same shape slicing logic as before)
        const sliced_data = try self.data.slice(dim, start, end);

        // Create a new tensor that shares the sliced data
        const new_tensor = try self.allocator.create(Tensor);

        // Adjust the shape for the new sliced tensor
        const new_shape = try self.allocator.alloc(usize, self.shape.len);
        new_shape[0] = end - start; // Only update the first dimension

        // Copy the remaining dimensions
        for (self.shape[1..], 1..) |d, i| {
            new_shape[i] = d;
        }

        // Assign values directly to the allocated `new_tensor`
        new_tensor.* = .{
            .allocator = self.allocator,
            .shape = new_shape,
            .dtype = self.dtype,
            .data = sliced_data,
            .parents = self.parents, // Share the same parents
            .operation = self.operation, // Share the same operation
            .label = self.label, // Share the same label
            .requires_grad = self.requires_grad, // Share the same requires_grad
            .grad = null, // No gradient for the sliced tensor
        };

        return new_tensor;
    }

    /// Creates a tensor from a scalar value.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for tensor operations.
    /// - `shape`: The shape of the tensor.
    /// - `dtype`: The data type of the tensor elements.
    /// - `value`: The scalar value to initialize the tensor with.
    /// - `requires_grad`: Whether the tensor requires gradient computation.
    ///
    /// # Returns
    /// A pointer to the newly created tensor.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_value(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, value: anytype, requires_grad: bool) TensorError!*Tensor {
        const ndarray = try NdArray.from_value(allocator, shape, dtype, value);
        const grad_array = if (requires_grad) try NdArray.zeros(allocator, shape, .f32) else null;
        const tensor = try allocator.create(Tensor);
        tensor.* = .{
            .allocator = allocator,
            .shape = try allocator.dupe(usize, shape),
            .dtype = dtype,
            .data = ndarray,
            .parents = &[_]*Tensor{},
            .operation = null,
            .label = "",
            .requires_grad = requires_grad,
            .grad = grad_array,
        };
        return tensor;
    }

    /// Creates a tensor from an existing `NdArray`.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for tensor operations.
    /// - `ndarray`: The `NdArray` to use as the tensor's data.
    /// - `parents`: A list of parent tensors used in the computation graph.
    /// - `operation`: The operation that produced this tensor (if any).
    /// - `label`: A label for the tensor (useful for debugging).
    /// - `requires_grad`: Whether the tensor requires gradient computation.
    ///
    /// # Returns
    /// A pointer to the newly created tensor.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_ndarray(allocator: std.mem.Allocator, ndarray: *NdArray, parents: []const *Tensor, operation: ?*autograd.Operation, label: []const u8, requires_grad: bool) TensorError!*Tensor {
        const grad_array = if (requires_grad) try NdArray.zeros(allocator, ndarray.shape, .f32) else null;
        const tensor = try allocator.create(Tensor);

        tensor.* = .{
            .allocator = allocator,
            .shape = try allocator.dupe(usize, ndarray.shape),
            .dtype = ndarray.dtype,
            .data = ndarray,
            .parents = try allocator.dupe(*Tensor, parents),
            .operation = operation,
            .label = try allocator.dupe(u8, label),
            .requires_grad = requires_grad,
            .grad = grad_array,
        };
        return tensor;
    }

    /// Creates a tensor from raw data.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for tensor operations.
    /// - `shape`: The shape of the tensor.
    /// - `dtype`: The data type of the tensor elements.
    /// - `data`: The raw data to initialize the tensor with.
    /// - `requires_grad`: Whether the tensor requires gradient computation.
    ///
    /// # Returns
    /// A pointer to the newly created tensor.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_data(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, data: anytype, requires_grad: bool) TensorError!*Tensor {
        const ndarray = try NdArray.from_data(allocator, shape, dtype, data);
        const grad_array = if (requires_grad) try NdArray.zeros(allocator, shape, .f32) else null;
        const tensor = try allocator.create(Tensor);
        tensor.* = .{
            .allocator = allocator,
            .shape = try allocator.dupe(usize, shape),
            .dtype = dtype,
            .data = ndarray,
            .parents = &[_]*Tensor{},
            .operation = null,
            .label = "",
            .requires_grad = requires_grad,
            .grad = grad_array,
        };
        return tensor;
    }

    /// Resets the gradient of the tensor to zero.
    ///
    /// # Errors
    /// Returns an error if gradient initialization fails.
    pub fn zero_grad(self: *Tensor) TensorError!void {
        if (self.requires_grad) {
            if (self.grad) |grad| {
                try grad.set_all(.{ .f32 = 0.0 });
            } else {
                self.grad = try NdArray.zeros(self.allocator, self.shape, .f32);
            }
        }
    }

    /// Retrieves the value at the specified indices.
    ///
    /// # Parameters
    /// - `indices`: The indices of the value to retrieve.
    ///
    /// # Returns
    /// The value at the specified indices as a `dtypes.NumericUnion`.
    ///
    /// # Errors
    /// Returns an error if the indices are invalid.
    pub fn get(self: *const Tensor, indices: []const usize) !dtypes.NumericUnion {
        return self.data.get(indices);
    }

    /// Sets the value at the specified indices.
    ///
    /// # Parameters
    /// - `indices`: The indices of the value to set.
    /// - `value`: The value to set.
    ///
    /// # Errors
    /// Returns an error if the indices are invalid or the value type is incompatible.
    pub fn set(self: *Tensor, indices: []const usize, value: dtypes.NumericUnion) !void {
        try self.data.set(indices, value);
    }

    /// Returns the total number of elements in the tensor.
    pub fn len(self: *const Tensor) usize {
        return self.data.len();
    }

    /// Performs backpropagation to compute gradients for the tensor.
    ///
    /// # Parameters
    /// - `grad`: The gradient to propagate (optional).
    ///
    /// # Errors
    /// Returns an error if gradient computation fails.
    pub fn backward(self: *Tensor, grad: ?*NdArray) TensorError!void {
        if (!self.requires_grad) return;

        if (grad == null) {
            self.grad = try NdArray.ones(self.allocator, self.shape, .f32);
            errdefer self.grad.deinit();
        } else {
            self.grad = grad;
        }

        var visited = std.AutoHashMap(*const Tensor, void).init(self.allocator);
        defer visited.deinit();

        var topo_order = std.ArrayList(*const Tensor).init(self.allocator);
        defer topo_order.deinit();

        try self.topological_sort(&visited, &topo_order);

        var i: usize = topo_order.items.len;
        while (i > 0) {
            i -= 1;
            const tensor = topo_order.items[i];
            std.debug.print("T.backward: Tensor = {s} | requires grad ? {any} -> Grad = ", .{ tensor.label, tensor.requires_grad });
            if (tensor.grad) |g| {
                try g.info();
            }

            if (tensor.operation) |op| {
                // std.debug.print("Backward operation: {s}\n", .{op.notation});
                // Unwrap `grad` before passing it to `op.backward`
                if (tensor.grad) |g| {
                    // try g.info();
                    try op.backward(tensor, g); // `g` is now `*NdArray`, not `?*NdArray`
                } else {
                    std.debug.print("Grad: None\n", .{});
                }
            }
            // std.debug.print("======== \n", .{});
        }
    }

    /// Performs a topological sort of the computation graph.
    ///
    /// # Parameters
    /// - `visited`: A hash map to track visited tensors.
    /// - `topo_order`: A list to store the sorted tensors.
    ///
    /// # Errors
    /// Returns an error if sorting fails.
    fn topological_sort(
        self: *const Tensor,
        visited: *std.AutoHashMap(*const Tensor, void),
        topo_order: *std.ArrayList(*const Tensor),
    ) TensorError!void {
        if (!visited.contains(self)) {
            try visited.put(self, {});
            for (self.parents) |parent| {
                try parent.topological_sort(visited, topo_order);
            }
            try topo_order.append(self);
        }
    }

    /// Updates the gradient of the tensor.
    ///
    /// # Parameters
    /// - `grad`: The gradient to update.
    ///
    /// # Errors
    /// Returns an error if gradient computation fails.
    pub fn update_grad(self: *Tensor, grad: *NdArray) !void {
        if (!self.requires_grad) return;

        // Ensure gradient is an NdArray
        var grad_scaled = try grad.scale(self.data.allocator, 1.0);

        // Broadcast gradient to match tensor shape
        grad_scaled = try grad_scaled.broadcast_to(self.data.allocator, self.data.shape);

        // If broadcasting fails, attempt sum reduction
        if (grad_scaled.shape.len != self.data.shape.len) {
            while (grad_scaled.shape.len > self.data.shape.len) {
                grad_scaled = try grad_scaled.sum(self.data.allocator, 0);
            }

            for (grad_scaled.shape, 0..) |grad_dim, axis| {
                if (grad_dim != self.data.shape[axis]) {
                    grad_scaled = try grad_scaled.sum(self.data.allocator, axis);
                }
            }
        }

        // Accumulate or initialize gradient
        if (self.grad) |existing_grad| {
            self.grad = try existing_grad.add(self.data.allocator, grad_scaled);
        } else {
            self.grad = grad_scaled;
        }
    }

    /// Prints the tensor's data.
    ///
    /// # Errors
    /// Returns an error if printing fails.
    pub fn print(self: *Tensor) !void {
        try self.data.print();
    }

    /// Prints detailed information about the tensor.
    ///
    /// # Errors
    /// Returns an error if printing fails.
    pub fn info(self: *Tensor) !void {
        std.debug.print("Tensor: [ label: {s}  ", .{self.label});
        try utils.print_ndarray_info(self.data, self.allocator);
        std.debug.print("]\n", .{});
    }

    /// Performs an element-wise addition with another tensor or scalar.
    ///
    /// # Parameters
    /// - `other`: The tensor or scalar to add.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn add(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &autograd.add_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    /// Performs an element-wise multiplication with another tensor or scalar.
    ///
    /// # Parameters
    /// - `other`: The tensor or scalar to multiply.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn mul(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &autograd.mul_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    /// Performs an element-wise subtraction with another tensor or scalar.
    ///
    /// # Parameters
    /// - `other`: The tensor or scalar to subtract.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sub(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &autograd.sub_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    /// Performs an element-wise division with another tensor or scalar.
    ///
    /// # Parameters
    /// - `other`: The tensor or scalar to divide by.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn div(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &autograd.div_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    /// Raises the tensor to the power of another tensor or scalar.
    ///
    /// # Parameters
    /// - `n`: The tensor or scalar exponent.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn pow(self: *Tensor, n: anytype) TensorError!*Tensor {
        const op = &autograd.pow_op;
        if (@TypeOf(n) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, n });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, n, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    /// Performs matrix multiplication with another tensor.
    ///
    /// # Parameters
    /// - `other`: The tensor to multiply with.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn matmul(self: *Tensor, other: *Tensor) TensorError!*Tensor {
        const op = &autograd.gemm_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, other });
    }

    /// Applies the hyperbolic tangent (tanh) function element-wise.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn tanh(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.tanh_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Applies the sigmoid function element-wise.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sigmoid(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.sigmoid_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Applies the softmax function element-wise.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn softmax(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.softmax_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Applies the rectified linear unit (ReLU) function element-wise.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn relu(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.relu_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Computes the minimum value along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the minimum (optional).
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn min(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &autograd.min_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, axis });
    }

    /// Computes the maximum value along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the maximum (optional).
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn max(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &autograd.max_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, axis });
    }

    /// Computes the mean value along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the mean (optional).
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn mean(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &autograd.mean_op;
        _ = axis;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Transposes the tensor.
    ///
    /// # Returns
    /// A new tensor representing the result of the operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn transpose(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.transpose_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    /// Creates a clone of the tensor.
    ///
    /// # Returns
    /// A new tensor with the same data and properties.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn clone(self: *Tensor) TensorError!*Tensor {
        const op = &autograd.clone_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }
};

// ============================
// Tests for the Tensor struct
// ============================
const expect = std.testing.expect;

test "Tensor.init()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    // defer ndarray.deinit(); // Remove this line

    const tensor = try Tensor.init(allocator, shape, .f32, ndarray, &[_]*Tensor{}, null, "", true);
    defer tensor.deinit(); // Tensor's deinit will handle the NdArray deallocation

    // Assert that the tensor's shape matches the expected shape
    std.debug.assert(tensor.shape.len == shape.len);
    for (tensor.shape, shape) |a, b| {
        std.debug.assert(a == b);
    }

    // Assert that the tensor's data matches the expected data
    const expected_value = @as(f32, 1.0);
    for (0..shape[0]) |i| {
        for (0..shape[1]) |j| {
            const indices = &[_]usize{ i, j };
            const actual_value = try ndarray.get(indices);
            std.debug.assert(actual_value.f32 == expected_value);
        }
    }

    // Assert that the gradient is initialized if requires_grad is true
    if (tensor.requires_grad) {
        std.debug.assert(tensor.grad != null);
        const grad_value = @as(f32, 0.0);
        for (0..shape[0]) |i| {
            for (0..shape[1]) |j| {
                const indices = &[_]usize{ i, j };
                const actual_grad_value = try tensor.grad.?.get(indices);
                std.debug.assert(actual_grad_value.f32 == grad_value);
            }
        }
    }
}

test "Tensor.from_value()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Test with all data types
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const value = switch (dtype) {
            .f32 => @as(f32, 3.5),
            .f64 => @as(f64, 3.5),
            .i32 => @as(i32, 3),
            .i64 => @as(i64, 3),
            .i16 => @as(i16, 3),
            .i8 => @as(i8, 3),
            .u32 => @as(u32, 3),
            .u64 => @as(u64, 3),
            .u16 => @as(u16, 3),
            .u8 => @as(u8, 3),
        };

        const shape = &[_]usize{ 2, 2 };
        const tensor = try Tensor.from_value(allocator, shape, dtype, value, true);
        defer tensor.deinit();

        std.debug.assert(tensor.shape.len == shape.len);
    }
}

test "Tensor.from_data()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };
    const tensor = try Tensor.from_data(allocator, shape, .f32, data_a, true);
    defer tensor.deinit();

    // try tensor.print();
    // try tensor.info();

    std.debug.assert(tensor.shape.len == shape.len);
    for (tensor.shape, shape) |a, b| {
        std.debug.assert(a == b);
    }
}

test "Tensor.len()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    const tensor = try Tensor.init(allocator, shape, .f32, ndarray, &[_]*Tensor{}, null, "", true);
    defer tensor.deinit();

    try expect(tensor.len() == 6); // Check length
}

test "Tensor.init and deinit" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));

    const tensor = try Tensor.init(allocator, shape, .f32, ndarray, &[_]*Tensor{}, null, "", true);
    defer tensor.deinit(); // Ensure deinit is called

    // Perform operations with the tensor
    const transposed = try tensor.transpose();
    defer transposed.deinit();
}

test "Tensor.from_ndarray()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Test with all data types
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const value = switch (dtype) {
            .f32 => @as(f32, 3.5),
            .f64 => @as(f64, 3.5),
            .i32 => @as(i32, 3),
            .i64 => @as(i64, 3),
            .i16 => @as(i16, 3),
            .i8 => @as(i8, 3),
            .u32 => @as(u32, 3),
            .u64 => @as(u64, 3),
            .u16 => @as(u16, 3),
            .u8 => @as(u8, 3),
        };

        const shape = &[_]usize{ 2, 2 };
        const ndarray = try NdArray.from_value(allocator, shape, dtype, value);
        const tensor = try Tensor.from_ndarray(allocator, ndarray, &[_]*Tensor{}, null, "", true);
        defer tensor.deinit();

        std.debug.assert(tensor.shape.len == shape.len);
    }
}

test "Tensor.set()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, true);
    defer a.deinit();

    try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

    const updated_val = try a.get(&[_]usize{ 0, 0 });
    std.debug.assert(updated_val.f32 == 42.0);
}

test "Tensor.get()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, true);
    defer a.deinit();

    const value = try a.get(&[_]usize{ 0, 0 });
    std.debug.assert(value.f32 == 1.0);
}

test "Tensor.transpose()" {
    const allocator = std.testing.allocator;
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };

    const tensor = try Tensor.from_data(allocator, shape, .f32, data_a, false);
    defer tensor.deinit();

    const transposed = try tensor.transpose();
    defer transposed.deinit();

    std.debug.assert(tensor.shape[0] == transposed.shape[1]);
    std.debug.assert(tensor.shape[1] == transposed.shape[0]);

    for (0..tensor.shape[0]) |i| {
        for (0..tensor.shape[1]) |j| {
            const original_val = try tensor.get(&[_]usize{ i, j });
            const transposed_val = try transposed.get(&[_]usize{ j, i });
            std.debug.assert(original_val.f32 == transposed_val.f32);
        }
    }
}

test "Tensor.add()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, false);
    defer a.deinit();
    const b = try Tensor.from_value(allocator, shape, .f32, 2.0, false);
    defer b.deinit();

    const sum = try a.add(b);
    defer sum.deinit();

    const sum_val = try sum.get(&[_]usize{ 0, 0 });
    std.debug.assert(sum_val.f32 == 3.0);
}

test "Tensor.sub()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, false);
    defer a.deinit();
    const b = try Tensor.from_value(allocator, shape, .f32, 2.0, false);
    defer b.deinit();

    const sub = try a.sub(b);
    defer sub.deinit();

    const sub_val = try sub.get(&[_]usize{ 0, 0 });
    std.debug.assert(sub_val.f32 == -1.0);
}

test "Tensor.mul()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, false);
    defer a.deinit();
    const b = try Tensor.from_value(allocator, shape, .f32, 2.0, false);
    defer b.deinit();

    const mul = try a.mul(b);
    defer mul.deinit();

    const mul_val = try mul.get(&[_]usize{ 0, 0 });
    std.debug.assert(mul_val.f32 == 2.0);
}

test "Tensor.div()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 5, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 1.0, false);
    defer a.deinit();
    const b = try Tensor.from_value(allocator, shape, .f32, 2.0, false);
    defer b.deinit();

    const div = try a.div(b);
    defer div.deinit();

    const div_val = try div.get(&[_]usize{ 0, 0 });
    std.debug.assert(div_val.f32 == 0.5);
}

test "Tensor.pow()" {
    const allocator = std.testing.allocator;
    const shape = &[_]usize{ 2, 3 };
    const a = try Tensor.from_value(allocator, shape, .f32, 3.0, false);
    defer a.deinit();
    const b = try Tensor.from_value(allocator, shape, .f32, 2.0, false);
    defer b.deinit();

    const pow = try a.pow(b);
    defer pow.deinit();

    const pow_val = try pow.get(&[_]usize{ 0, 0 });
    std.debug.assert(pow_val.f32 == 9.0);
}
