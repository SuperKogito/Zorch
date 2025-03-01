const std = @import("std");
const utils = @import("math/ndarray/utils.zig");

const ops = @import("autograd.zig");
const dtypes = @import("math/ndarray/dtypes.zig");
const NdArray = @import("math/ndarray/ndarray.zig").NdArray;

pub const Operation = ops.Operation;
pub const DataType = dtypes.DataType;
pub const NumericUnion = dtypes.NumericUnion;

pub const add_op = ops.add_op;
pub const mul_op = ops.mul_op;
pub const sub_op = ops.sub_op;
pub const div_op = ops.div_op;
pub const gemm_op = ops.gemm_op;
pub const transpose_op = ops.transpose_op;
pub const tanh_op = ops.tanh_op;
pub const sigmoid_op = ops.sigmoid_op;
pub const softmax_op = ops.softmax_op;
pub const relu_op = ops.relu_op;
pub const pow_op = ops.pow_op;
pub const min_op = ops.min_op;
pub const max_op = ops.max_op;
pub const mean_op = ops.mean_op;

// Define a custom error set for Tensor operations
const TensorError = @import("errors.zig").TensorError;

pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []const usize,
    dtype: DataType,
    data: *NdArray,
    parents: []const *Tensor,
    operation: ?*Operation,
    label: []const u8,
    requires_grad: bool,
    grad: ?*NdArray,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType, data: *NdArray, parents: []const *Tensor, operation: ?*Operation, label: []const u8, requires_grad: bool) TensorError!*Tensor {
        const tensor = try allocator.create(Tensor);
        const grad_array = if (requires_grad) try NdArray.zeros(allocator, shape, .f32) else null;

        tensor.* = .{
            .allocator = allocator,
            .shape = try allocator.dupe(usize, shape),
            .dtype = dtype,
            .data = data,
            .parents = try allocator.dupe(*Tensor, parents),
            .operation = operation,
            .label = try allocator.dupe(u8, label),
            .requires_grad = requires_grad,
            .grad = grad_array,
        };

        return tensor;
    }

    pub fn deinit(self: *Tensor) void {
        std.debug.print("Freeing Tensor {s}\n", .{self.label});

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

    pub fn from_value(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType, value: anytype, requires_grad: bool) TensorError!*Tensor {
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

    pub fn from_ndarray(
        allocator: std.mem.Allocator,
        ndarray: *NdArray,
        parents: []const *Tensor,
        operation: ?*Operation,
        label: []const u8,
        requires_grad: bool,
    ) TensorError!*Tensor {
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

    pub fn from_data(
        allocator: std.mem.Allocator,
        shape: []const usize,
        dtype: DataType,
        data: anytype,
        requires_grad: bool,
    ) TensorError!*Tensor {
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

    pub fn zero_grad(self: *Tensor) TensorError!void {
        if (self.requires_grad) {
            if (self.grad) |grad| {
                try grad.set_all(.{ .f32 = 0.0 });
            } else {
                self.grad = try NdArray.zeros(self.allocator, self.shape, .f32);
            }
        }
    }

    pub fn get(self: *const Tensor, indices: []const usize) !NumericUnion {
        return self.data.get(indices);
    }

    pub fn set(self: *Tensor, indices: []const usize, value: NumericUnion) !void {
        try self.data.set(indices, value);
    }

    pub fn len(self: *const Tensor) usize {
        return self.data.len();
    }

    pub fn backward(self: *Tensor, grad: ?*NdArray) TensorError!void {
        if (!self.requires_grad) return;

        self.grad = grad;

        var visited = std.AutoHashMap(*const Tensor, void).init(self.allocator);
        defer visited.deinit();

        var topo_order = std.ArrayList(*const Tensor).init(self.allocator);
        defer topo_order.deinit();

        try self.buildTopo(&visited, &topo_order);

        var i: usize = topo_order.items.len;
        while (i > 0) {
            i -= 1;
            const tensor = topo_order.items[i];
            // std.debug.print("Tensor = {s} | {any} -> ", .{ tensor.label, tensor.requires_grad });
            // try @constCast(tensor).info();

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

    fn buildTopo(
        self: *const Tensor,
        visited: *std.AutoHashMap(*const Tensor, void),
        topo_order: *std.ArrayList(*const Tensor),
    ) TensorError!void {
        if (!visited.contains(self)) {
            try visited.put(self, {});
            for (self.parents) |parent| {
                try parent.buildTopo(visited, topo_order);
            }
            try topo_order.append(self);
        }
    }

    pub fn print(self: *Tensor) !void {
        try self.data.print();
    }

    pub fn info(self: *Tensor) !void {
        std.debug.print("Tensor: [", .{});
        try utils.print_ndarray_info(self.data, self.allocator);
        std.debug.print("]\n", .{});
    }

    pub fn add(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &add_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    pub fn mul(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &mul_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    pub fn sub(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &sub_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    pub fn div(self: *Tensor, other: anytype) TensorError!*Tensor {
        const op = &div_op;
        if (@TypeOf(other) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, other });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, other, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    pub fn pow(self: *Tensor, n: anytype) TensorError!*Tensor {
        const op = &pow_op;
        if (@TypeOf(n) == *Tensor) {
            return try op.forward(self.allocator, &[_]*Tensor{ self, n });
        } else {
            const scalar_tensor = try Tensor.from_value(self.allocator, self.shape, self.dtype, n, false);
            return try op.forward(self.allocator, &[_]*Tensor{ self, scalar_tensor });
        }
    }

    pub fn matmul(self: *Tensor, other: *Tensor) TensorError!*Tensor {
        const op = &gemm_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, other });
    }

    pub fn tanh(self: *Tensor) TensorError!*Tensor {
        const op = &tanh_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    pub fn sigmoid(self: *Tensor) TensorError!*Tensor {
        const op = &sigmoid_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    pub fn softmax(self: *Tensor) TensorError!*Tensor {
        const op = &softmax_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    pub fn relu(self: *Tensor) TensorError!*Tensor {
        const op = &relu_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    pub fn min(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &min_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, axis });
    }

    pub fn max(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &max_op;
        return try op.forward(self.allocator, &[_]*Tensor{ self, axis });
    }
    pub fn mean(self: *Tensor, axis: ?usize) !*Tensor {
        const op = &mean_op;
        _ = axis;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }

    pub fn transpose(self: *Tensor) TensorError!*Tensor {
        const op = &transpose_op;
        return try op.forward(self.allocator, &[_]*Tensor{self});
    }
};

// Helper function
fn compute_size(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| size *= dim;
    return size;
}

// // ============================
// // Tests for the Tensor struct
// // ============================
// const expect = std.testing.expect;

// test "Tensor.init()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 3 };
//     const ndarray = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     // defer ndarray.deinit(); // Remove this line

//     const tensor = try Tensor.init(allocator, shape, dtypes.DataType.f32, ndarray, &[_]*Tensor{}, null, "", true);
//     defer tensor.deinit(); // Tensor's deinit will handle the NdArray deallocation

//     // Assert that the tensor's shape matches the expected shape
//     std.debug.assert(tensor.shape.len == shape.len);
//     for (tensor.shape, shape) |a, b| {
//         std.debug.assert(a == b);
//     }

//     // Assert that the tensor's data matches the expected data
//     const expected_value = @as(f32, 1.0);
//     for (0..shape[0]) |i| {
//         for (0..shape[1]) |j| {
//             const indices = &[_]usize{ i, j };
//             const actual_value = try ndarray.get(indices);
//             std.debug.assert(actual_value.f32 == expected_value);
//         }
//     }

//     // Assert that the gradient is initialized if requires_grad is true
//     if (tensor.requires_grad) {
//         std.debug.assert(tensor.grad != null);
//         const grad_value = @as(f32, 0.0);
//         for (0..shape[0]) |i| {
//             for (0..shape[1]) |j| {
//                 const indices = &[_]usize{ i, j };
//                 const actual_grad_value = try tensor.grad.?.get(indices);
//                 std.debug.assert(actual_grad_value.f32 == grad_value);
//             }
//         }
//     }
// }

// test "Tensor.from_value()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     // Test with all data types
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const value = switch (dtype) {
//             .f32 => @as(f32, 3.5),
//             .f64 => @as(f64, 3.5),
//             .i32 => @as(i32, 3),
//             .i64 => @as(i64, 3),
//             .i16 => @as(i16, 3),
//             .i8 => @as(i8, 3),
//             .u32 => @as(u32, 3),
//             .u64 => @as(u64, 3),
//             .u16 => @as(u16, 3),
//             .u8 => @as(u8, 3),
//         };

//         const shape = &[_]usize{ 2, 2 };
//         const tensor = try Tensor.from_value(allocator, shape, dtype, value, true);
//         defer tensor.deinit();

//         std.debug.assert(tensor.shape.len == shape.len);
//     }
// }

// test "Tensor.from_data()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };
//     const tensor = try Tensor.from_data(allocator, shape, dtypes.DataType.f32, data_a, true);
//     defer tensor.deinit();

//     // try tensor.print();
//     // try tensor.info();

//     std.debug.assert(tensor.shape.len == shape.len);
//     for (tensor.shape, shape) |a, b| {
//         std.debug.assert(a == b);
//     }
// }

// test "Tensor.len()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 3 };
//     const ndarray = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     const tensor = try Tensor.init(allocator, shape, dtypes.DataType.f32, ndarray, &[_]*Tensor{}, null, "", true);
//     defer tensor.deinit();

//     try expect(tensor.len() == 6); // Check length
// }

// test "Tensor.init and deinit" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 3 };
//     const ndarray = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));

//     const tensor = try Tensor.init(allocator, shape, dtypes.DataType.f32, ndarray, &[_]*Tensor{}, null, "", true);
//     defer tensor.deinit(); // Ensure deinit is called

//     // Perform operations with the tensor
//     const transposed = try tensor.transpose();
//     defer transposed.deinit();
// }

// test "Tensor.from_ndarray()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     // Test with all data types
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const value = switch (dtype) {
//             .f32 => @as(f32, 3.5),
//             .f64 => @as(f64, 3.5),
//             .i32 => @as(i32, 3),
//             .i64 => @as(i64, 3),
//             .i16 => @as(i16, 3),
//             .i8 => @as(i8, 3),
//             .u32 => @as(u32, 3),
//             .u64 => @as(u64, 3),
//             .u16 => @as(u16, 3),
//             .u8 => @as(u8, 3),
//         };

//         const shape = &[_]usize{ 2, 2 };
//         const ndarray = try NdArray.from_value(allocator, shape, dtype, value);
//         const tensor = try Tensor.from_ndarray(allocator, ndarray, &[_]*Tensor{}, null, "", true);
//         defer tensor.deinit();

//         std.debug.assert(tensor.shape.len == shape.len);
//     }
// }

// test "Tensor.set()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, true);
//     defer a.deinit();

//     try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

//     const updated_val = try a.get(&[_]usize{ 0, 0 });
//     std.debug.assert(updated_val.f32 == 42.0);
// }

// test "Tensor.get()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, true);
//     defer a.deinit();

//     const value = try a.get(&[_]usize{ 0, 0 });
//     std.debug.assert(value.f32 == 1.0);
// }

// test "Tensor.transpose()" {
//     const allocator = std.testing.allocator;
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };

//     const tensor = try Tensor.from_data(allocator, shape, dtypes.DataType.f32, data_a, false);
//     defer tensor.deinit();

//     const transposed = try tensor.transpose();
//     defer transposed.deinit();

//     std.debug.assert(tensor.shape[0] == transposed.shape[1]);
//     std.debug.assert(tensor.shape[1] == transposed.shape[0]);

//     for (0..tensor.shape[0]) |i| {
//         for (0..tensor.shape[1]) |j| {
//             const original_val = try tensor.get(&[_]usize{ i, j });
//             const transposed_val = try transposed.get(&[_]usize{ j, i });
//             std.debug.assert(original_val.f32 == transposed_val.f32);
//         }
//     }
// }

// test "Tensor.add()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, false);
//     defer a.deinit();
//     const b = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 2.0, false);
//     defer b.deinit();

//     const sum = try a.add(b);
//     defer sum.deinit();

//     const sum_val = try sum.get(&[_]usize{ 0, 0 });
//     std.debug.assert(sum_val.f32 == 3.0);
// }

// test "Tensor.sub()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, false);
//     defer a.deinit();
//     const b = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 2.0, false);
//     defer b.deinit();

//     const sub = try a.sub(b);
//     defer sub.deinit();

//     const sub_val = try sub.get(&[_]usize{ 0, 0 });
//     std.debug.assert(sub_val.f32 == -1.0);
// }

// test "Tensor.mul()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, false);
//     defer a.deinit();
//     const b = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 2.0, false);
//     defer b.deinit();

//     const mul = try a.mul(b);
//     defer mul.deinit();

//     const mul_val = try mul.get(&[_]usize{ 0, 0 });
//     std.debug.assert(mul_val.f32 == 2.0);
// }

// test "Tensor.div()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 5, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 1.0, false);
//     defer a.deinit();
//     const b = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 2.0, false);
//     defer b.deinit();

//     const div = try a.div(b);
//     defer div.deinit();

//     const div_val = try div.get(&[_]usize{ 0, 0 });
//     std.debug.assert(div_val.f32 == 0.5);
// }

// test "Tensor.pow()" {
//     const allocator = std.testing.allocator;
//     const shape = &[_]usize{ 2, 3 };
//     const a = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 3.0, false);
//     defer a.deinit();
//     const b = try Tensor.from_value(allocator, shape, dtypes.DataType.f32, 2.0, false);
//     defer b.deinit();

//     const pow = try a.pow(b);
//     defer pow.deinit();

//     const pow_val = try pow.get(&[_]usize{ 0, 0 });
//     std.debug.assert(pow_val.f32 == 9.0);
// }
