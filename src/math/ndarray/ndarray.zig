const std = @import("std");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");
const converters = @import("converters.zig");

pub const DataType = dtypes.DataType;
pub const NumericUnion = dtypes.NumericUnion;
pub const ScalarValue = dtypes.ScalarValue;

const nd_bops = @import("operations_binary_ndarray.zig");
const sc_bops = @import("operations_binary_skalar.zig");
const nd_uops = @import("operations_unary_ndarray.zig");
const nd_rops = @import("operations_reduction_ndarray.zig");

const NdNdBinOp = @import("operations_binary_ndarray.zig").NdArrayBinaryOperation;
const NdScBinOp = @import("operations_binary_skalar.zig").NdArrayScalarBinaryOperation;
const NdUnaOp = @import("operations_unary_ndarray.zig").NdArrayUnaryOperation;
const NdRedOp = @import("operations_reduction_ndarray.zig").NdArrayReductionOperation;

pub const NdArray = struct {
    shape: []usize,
    dtype: DataType,
    data: []u8,
    strides: []usize,
    allocator: std.mem.Allocator,
    owns_data: bool, // Flag to track ownership of data

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*NdArray {
        const ndarray = try allocator.create(NdArray);
        errdefer allocator.destroy(ndarray);

        // Allocate and copy the shape
        ndarray.shape = try allocator.alloc(usize, shape.len);
        @memcpy(ndarray.shape, shape); // Copy the input shape into ndarray.shape
        errdefer allocator.free(ndarray.shape);

        const size = utils.compute_size(shape);
        const bytes_needed = size * dtype.sizeInBytes();
        ndarray.data = try allocator.alloc(u8, bytes_needed);
        errdefer allocator.free(ndarray.data);

        ndarray.strides = try allocator.alloc(usize, shape.len);
        errdefer allocator.free(ndarray.strides);
        compute_strides(shape, ndarray.strides);

        ndarray.dtype = dtype;
        ndarray.allocator = allocator;
        ndarray.owns_data = true;

        return ndarray;
    }
    pub fn deinit(self: *NdArray) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        if (self.owns_data) {
            self.allocator.free(self.data); // Only free data if owned
        }
        self.allocator.destroy(self);
    }

    pub fn compute_strides(shape: []const usize, strides: []usize) void {
        var stride: usize = 1;
        for (0..shape.len) |i| {
            strides[shape.len - 1 - i] = stride;
            stride *= shape[shape.len - 1 - i];
        }
    }

    pub fn slice(self: *const NdArray, dim: usize, start: usize, end: usize) !*NdArray {
        // Validate the dimension
        if (dim >= self.shape.len) {
            return error.InvalidDimension;
        }

        // Validate the slice indices
        if (start >= end or end > self.shape[dim]) {
            return error.InvalidSliceIndices;
        }

        // Calculate the new shape
        const new_shape = try self.allocator.alloc(usize, self.shape.len);
        for (self.shape, 0..) |dim_size, i| {
            new_shape[i] = if (i == dim) end - start else dim_size;
        }

        // Calculate the offset in the underlying data
        const offset = start * self.strides[dim];

        // Create a new NdArray that shares the underlying data
        const sliced_ndarray = try self.allocator.create(NdArray);
        sliced_ndarray.* = .{
            .shape = new_shape,
            .dtype = self.dtype,
            .data = self.data[offset * self.dtype.sizeInBytes() ..], // Share the underlying data
            .strides = try self.allocator.dupe(usize, self.strides),
            .allocator = self.allocator,
            .owns_data = false, // Sliced array does not own the data
        };

        return sliced_ndarray;
    }

    pub fn from_value(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType, value: anytype) !*NdArray {
        const size = utils.compute_size(shape);
        const element_size = dtype.sizeInBytes();
        const total_bytes = size * element_size;

        const data = try allocator.alloc(u8, total_bytes);
        errdefer allocator.free(data);

        const numeric_value = convert_value_to_dtype(value, dtype);
        const value_bytes = std.mem.asBytes(&numeric_value);

        for (0..size) |i| {
            const offset = i * element_size;
            @memcpy(data[offset .. offset + element_size], value_bytes[0..element_size]);
        }

        return NdArray.from_bytes(allocator, shape, dtype, data);
    }

    pub fn zeros(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*NdArray {
        return NdArray.from_value(allocator, shape, dtype, 0.0);
    }

    pub fn ones(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType) !*NdArray {
        return NdArray.from_value(allocator, shape, dtype, 1.0);
    }

    pub fn from_bytes(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType, data: []u8) !*NdArray {
        const ndarray = try allocator.create(NdArray);
        ndarray.* = .{
            .shape = try allocator.dupe(usize, shape),
            .dtype = dtype,
            .data = data,
            .strides = try allocator.alloc(usize, shape.len),
            .allocator = allocator,
            .owns_data = true,
        };
        compute_strides(shape, ndarray.strides);
        return ndarray;
    }

    pub fn from_data(allocator: std.mem.Allocator, shape: []const usize, dtype: DataType, data: anytype) !*NdArray {
        const Self = @This();
        const ndarray = try allocator.create(Self);
        errdefer allocator.destroy(ndarray);

        // Allocate and copy the shape
        ndarray.shape = try allocator.alloc(usize, shape.len);
        @memcpy(ndarray.shape, shape);

        const element_size = dtype.sizeInBytes();
        const total_bytes = data.len * element_size;

        // Allocate memory for the data
        ndarray.data = try allocator.alloc(u8, total_bytes);

        // Convert the input data into a slice and copy it
        const data_slice = data[0..data.len]; // Convert pointer to array into a slice
        @memcpy(ndarray.data, std.mem.sliceAsBytes(data_slice));

        // Compute strides
        ndarray.strides = try allocator.alloc(usize, shape.len);
        compute_strides(shape, ndarray.strides);

        ndarray.dtype = dtype;
        ndarray.allocator = allocator;
        ndarray.owns_data = true; // The array owns its data

        return ndarray;
    }

    pub fn clone(self: *NdArray) !*NdArray {
        const cloned = try NdArray.init(self.allocator, self.shape, self.dtype);
        @memcpy(cloned.data, self.data);
        return cloned;
    }

    pub fn get(self: *const NdArray, indices: []const usize) !NumericUnion {
        for (indices, self.shape) |idx, dim_size| {
            if (idx >= dim_size) return error.IndexOutOfBounds;
        }
        const flat_idx = self.flatten_idx(indices);
        const byte_offset = flat_idx * self.dtype.sizeInBytes();
        return converters.bytes_to_val(self, byte_offset);
    }

    pub fn set(self: *NdArray, indices: []const usize, value: NumericUnion) !void {
        for (indices, self.shape) |idx, dim_size| {
            if (idx >= dim_size) return error.IndexOutOfBounds;
        }
        const flat_idx = self.flatten_idx(indices);
        const byte_offset = flat_idx * self.dtype.sizeInBytes();
        converters.val_to_bytes(self, byte_offset, value);
    }

    pub fn len(self: *NdArray) usize {
        var size: usize = 1;
        for (self.shape) |dim| size *= dim;
        return size;
    }

    pub fn set_all(self: *NdArray, value: anytype) void {
        const total_size = utils.compute_size(self.shape);
        const element_size = self.dtype.sizeInBytes();
        const numeric_value = convert_value_to_dtype(value, self.dtype) catch |err| {
            @panic(@errorName(err)); // Handle errors if needed
        };

        // Extract the bytes of the specific type stored in `numeric_value`
        const value_bytes = switch (self.dtype) {
            .f32 => std.mem.asBytes(&numeric_value.f32),
            .f64 => std.mem.asBytes(&numeric_value.f64),
            .i8 => std.mem.asBytes(&numeric_value.i8),
            .i16 => std.mem.asBytes(&numeric_value.i16),
            .i32 => std.mem.asBytes(&numeric_value.i32),
            .i64 => std.mem.asBytes(&numeric_value.i64),
            .u8 => std.mem.asBytes(&numeric_value.u8),
            .u16 => std.mem.asBytes(&numeric_value.u16),
            .u32 => std.mem.asBytes(&numeric_value.u32),
            .u64 => std.mem.asBytes(&numeric_value.u64),
        };

        // Ensure the size of `value_bytes` matches `element_size`
        if (value_bytes.len != element_size) {
            @panic("Size of value_bytes does not match element_size");
        }

        for (0..total_size) |i| {
            const offset = i * element_size;
            @memcpy(self.data[offset .. offset + element_size], value_bytes);
        }
    }

    pub fn fill(self: *NdArray, value: anytype) !void {
        self.set_all(value);
    }

    fn flatten_idx(self: *const NdArray, indices: []const usize) usize {
        var idx: usize = 0;
        for (indices, self.strides) |i, stride| {
            idx += i * stride;
        }
        return idx;
    }

    pub fn reshape(self: *NdArray, new_shape: []const usize) !*NdArray {
        const original_size = utils.compute_size(self.shape);
        const new_size = utils.compute_size(new_shape);
        if (original_size != new_size) {
            return error.ReshapeError;
        }

        const reshaped = try NdArray.init(self.allocator, new_shape, self.dtype);
        @memcpy(reshaped.data, self.data);
        return reshaped;
    }

    pub fn print(self: *NdArray) !void {
        try utils.print_ndarray(self, self.allocator);
    }

    pub fn info(self: *NdArray) !void {
        std.debug.print("NdArray: [", .{});
        try utils.print_ndarray_info(self, self.allocator);
        std.debug.print("]\n", .{});
    }

    pub fn cast(self: *NdArray, target_dtype: DataType) !*NdArray {
        return try nd_uops.cast_elements(self, target_dtype);
    }

    // ===========================
    // Reduction Operations
    // ===========================
    pub fn get_size(self: *NdArray) usize {
        var size: usize = 1;
        for (self.shape) |dim| size *= dim;
        return size;
    }

    // pub fn argmin(self: *NdArray) ![]const usize {
    //     var min_index: usize = 0;
    //     var min_value = self.data[0];

    //     for (0.., self.data) |i, value| {
    //         if (value < min_value) {
    //             min_value = value;
    //             min_index = i;
    //         }
    //     }

    //     // Convert flat index to multi-dimensional indices
    //     const indices = try utils.unravel_index(self.allocator, min_index, self.shape);
    //     return indices;
    // }

    // pub fn argmax(self: *NdArray) ![]const usize {
    //     var max_index: usize = 0;
    //     var max_value = self.data[0];

    //     for (0.., self.data) |i, value| {
    //         if (value > max_value) {
    //             max_value = value;
    //             max_index = i;
    //         }
    //     }

    //     // Convert flat index to multi-dimensional indices
    //     const indices = try utils.unravel_index(self.allocator, max_index, self.shape);
    //     return indices;
    // }

    // pub fn unravel_index(allocator: std.mem.Allocator, flat_index: usize, shape: []const usize) ![]const usize {
    //     const n_dims = shape.len;
    //     var indices = try allocator.alloc(usize, n_dims);
    //     errdefer allocator.free(indices);

    //     var remaining_index = flat_index;
    //     for (0.., shape) |i, dim_size| {
    //         indices[i] = remaining_index % dim_size;
    //         remaining_index /= dim_size;
    //     }

    //     return indices;
    // }
    // pub fn ravel_multi_index(indices: []const usize, shape: []const usize) usize {
    //     var flat_index: usize = 0;
    //     var stride: usize = 1;

    //     for (0.., shape) |i, dim_size| {
    //         flat_index += indices[i] * stride;
    //         stride *= dim_size;
    //     }

    //     return flat_index;
    // }

    // pub fn min_backward(self: *NdArray, grad: *NdArray, axis: ?usize) !*NdArray {
    //     const min_indices = try self.argmin();
    //     defer self.allocator.free(min_indices);

    //     const grad_output = try NdArray.zeros(self.allocator, self.shape, self.dtype);
    //     defer grad_output.deinit();

    //     for (min_indices) |index| {
    //         const flat_idx = utils.ravel_multi_index(index, self.shape);
    //         grad_output.data[flat_idx] = grad.data[0];
    //     }

    //     return grad_output;
    // }

    // pub fn mean_backward(self: *NdArray, grad: *NdArray, axis: ?usize) !*NdArray {
    //     const total_elements = utils.compute_size(self.shape);
    //     const grad_per_element = grad.data[0] / @intFromFloat(f32, total_elements);

    //     const grad_output = try NdArray.full(self.allocator, self.shape, grad_per_element, self.dtype);
    //     return grad_output;
    // }

    // pub fn max_backward(self: *NdArray, grad: *NdArray, axis: ?usize) !*NdArray {
    //     const max_indices = try self.argmax();
    //     defer self.allocator.free(max_indices);

    //     const grad_output = try NdArray.zeros(self.allocator, self.shape, self.dtype);
    //     defer grad_output.deinit();

    //     for (max_indices) |index| {
    //         const flat_idx = utils.ravel_multi_index(index, self.shape);
    //         grad_output.data[flat_idx] = grad.data[0];
    //     }

    //     return grad_output;
    // }

    pub fn min(self: *NdArray, axis: ?usize) !*NdArray {
        return try nd_rops.reduce(self, NdRedOp.MIN, axis);
    }

    pub fn max(self: *NdArray, axis: ?usize) !*NdArray {
        return try nd_rops.reduce(self, NdRedOp.MAX, axis);
    }

    pub fn sum(self: *NdArray, axis: ?usize) !*NdArray {
        return try nd_rops.reduce(self, NdRedOp.SUM, axis);
    }

    pub fn mean(self: *NdArray, axis: ?usize) !*NdArray {
        return try nd_rops.reduce(self, NdRedOp.MEAN, axis);
    }

    pub fn prod(self: *NdArray, axis: ?usize) !*NdArray {
        return try nd_rops.reduce(self, NdRedOp.PROD, axis);
    }

    // ===========================
    // Unary Operations
    // ===========================
    pub fn log(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.LOG, in_place);
    }

    pub fn exp(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.EXP, in_place);
    }

    pub fn tanh(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.TANH, in_place);
    }

    pub fn relu(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.RELU, in_place);
    }

    pub fn softmax(self: *NdArray, axis: ?usize, in_place: bool) !*NdArray {
        // Step 1: Compute max along the axis
        const max_x = try self.max(axis);
        defer max_x.deinit();

        // Reshape max_x to match the dimensionality of self
        const reshaped_max_x = try max_x.reshape(&[_]usize{ max_x.shape[0], 1 });
        defer reshaped_max_x.deinit();

        // Step 2: Subtract reshaped_max_x from self (stabilize the values)
        const x_stable = try self.sub(reshaped_max_x, in_place);
        defer x_stable.deinit();

        // Step 3: Compute exponentials
        const exp_x = try x_stable.exp(in_place);
        defer exp_x.deinit();

        // Step 4: Compute sum of exponentials along the given axis
        const sum_exp_x = try exp_x.sum(axis);
        defer sum_exp_x.deinit();

        // Reshape sum_exp_x to match the dimensionality of exp_x
        const reshaped_sum_exp_x = try sum_exp_x.reshape(&[_]usize{ sum_exp_x.shape[0], 1 });
        defer reshaped_sum_exp_x.deinit();

        // Step 5: Normalize by dividing exp_x by reshaped_sum_exp_x
        const z = try nd_bops.apply_elementwise(exp_x, reshaped_sum_exp_x, self.allocator, NdNdBinOp.DIV, in_place);
        // Do NOT deinit z here! The caller is responsible for deallocating the result.
        return z;
    }

    pub fn sigmoid(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.SIGMOID, in_place);
    }

    pub fn neg(self: *NdArray, in_place: bool) !*NdArray {
        return try nd_uops.apply_unary(self, NdUnaOp.NEG, in_place);
    }

    // ===============================
    // Binary Operations (with Scalar)
    // ===============================
    pub fn add_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.ADD, in_place);
    }

    pub fn sub_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.SUB, in_place);
    }

    pub fn mul_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.MUL, in_place);
    }

    pub fn div_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.DIV, in_place);
    }

    pub fn pow_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.POW, in_place);
    }

    pub fn greater_than_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.GT, in_place);
    }
    pub fn smaller_than_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.LT, in_place);
    }
    pub fn equal_to_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try sc_bops.apply_binary(self, scalar, NdScBinOp.EQ, in_place);
    }
    // ===============================
    // Binary Operations (with NdArray)
    // ===============================
    pub fn add(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.ADD, in_place);
    }

    pub fn sub(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.SUB, in_place);
    }

    pub fn mul(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.MUL, in_place);
    }

    pub fn div(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.DIV, in_place);
    }

    pub fn pow(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.POW, in_place);
    }

    pub fn equal(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const equal_ndarray = try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.EQ, in_place);
        return equal_ndarray.cast(DataType.i32);
    }

    pub fn greater_than(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const gt_ndarray = try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.GT, in_place);
        return gt_ndarray.cast(DataType.i32);
    }

    pub fn less_than(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const lt_ndarray = try nd_bops.apply_elementwise(self, other, self.allocator, NdNdBinOp.LT, in_place);
        return lt_ndarray.cast(DataType.i32);
    }

    pub fn matmul(self: *NdArray, b: *const NdArray, alpha: f32, beta: f32, transA: bool, transB: bool) !*NdArray {
        return try nd_bops.gemm(self, b, alpha, beta, transA, transB, self.allocator);
    }

    pub fn transpose(self: *NdArray) !*NdArray {
        const rank = self.shape.len;

        // Allocate memory for the new shape and strides
        const new_shape = try self.allocator.alloc(usize, rank);
        errdefer self.allocator.free(new_shape);

        const new_strides = try self.allocator.alloc(usize, rank);
        errdefer self.allocator.free(new_strides);

        // Reverse the shape and strides
        for (0..rank) |i| {
            new_shape[i] = self.shape[rank - 1 - i];
            new_strides[i] = self.strides[rank - 1 - i];
        }

        // Create a new NdArray with the transposed shape and strides
        const transposed = try self.allocator.create(NdArray);
        transposed.* = .{
            .shape = new_shape,
            .dtype = self.dtype,
            .data = self.data, // Share the same underlying data
            .strides = new_strides,
            .allocator = self.allocator,
            .owns_data = false, // Transposed array does not own the data
        };

        return transposed;
    }
};

fn convert_value_to_dtype(value: anytype, dtype: DataType) !NumericUnion {
    return switch (dtype) {
        .f32 => NumericUnion{ .f32 = @floatCast(value) },
        .f64 => NumericUnion{ .f64 = @floatCast(value) },
        .i8 => NumericUnion{ .i8 = @intFromFloat(@floor(@as(f64, value))) },
        .i16 => NumericUnion{ .i16 = @intFromFloat(@floor(@as(f64, value))) },
        .i32 => NumericUnion{ .i32 = @intFromFloat(@floor(@as(f64, value))) },
        .i64 => NumericUnion{ .i64 = @intFromFloat(@floor(@as(f64, value))) },
        .u8 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return NumericUnion{ .u8 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u16 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return NumericUnion{ .u16 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u32 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return NumericUnion{ .u32 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u64 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return NumericUnion{ .u64 = @intFromFloat(@floor(@as(f64, value))) };
        },
    };
}

// // ============================
// // Tests for the NdArray struct
// // ============================
// const expect = std.testing.expect;

// test "ndarray.init()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 3 };
//     const ndarray = try NdArray.init(allocator, shape, .f32);
//     defer ndarray.deinit();

//     // Verify the shape
//     std.debug.assert(ndarray.shape[0] == 2);
//     std.debug.assert(ndarray.shape[1] == 3);

//     // Verify the strides
//     std.debug.assert(ndarray.strides[0] == 3);
//     std.debug.assert(ndarray.strides[1] == 1);
// }

// test "ndarray.from_value()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

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
//         defer ndarray.deinit();
//         std.debug.assert(ndarray.shape.len == shape.len);
//     }
// }

// test "ndarray.from_data()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };
//     const ndarray = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer ndarray.deinit();

//     std.debug.assert(ndarray.shape.len == shape.len);
//     for (ndarray.shape, shape) |a, b| {
//         std.debug.assert(a == b);
//     }
// }

// test "ndarray.len()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 3 };
//     const ndarray = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer ndarray.deinit();

//     try expect(ndarray.len() == 6); // Check length
// }

// test "ndarray.clone()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 2, 2 };
//     const original = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 7.0));
//     defer original.deinit();

//     const cloned = try original.clone();
//     defer cloned.deinit();

//     const value = try cloned.get(&[_]usize{ 1, 1 });
//     std.debug.assert(value.f32 == 7.0);
// }

// test "ndarray.ones()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const ones = try NdArray.ones(allocator, shape, dtypes.DataType.f32);
//     defer ones.deinit();
// }

// test "ndarray.zeros()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const zeros = try NdArray.zeros(allocator, shape, dtypes.DataType.f32);
//     defer zeros.deinit();
// }

// test "ndarray.transpose()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();
//     const debug = false;

//     // Original data for the ndarray
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };

//     // Create the ndarray from data
//     const ndarray = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer ndarray.deinit();

//     // Perform transpose operation
//     const transposed = try ndarray.transpose();
//     defer transposed.deinit();

//     if (debug) {
//         try ndarray.print();
//         try transposed.info();
//         try transposed.print();
//     }

//     // Assert the shapes are transposed correctly
//     try expect(ndarray.shape[0] == transposed.shape[1]);
//     try expect(ndarray.shape[1] == transposed.shape[0]);

//     // Verify that transposed values match the expected values
//     for (0..ndarray.shape[0]) |i| {
//         for (0..ndarray.shape[1]) |j| {
//             const original_val = try ndarray.get(&[_]usize{ i, j });
//             const transposed_val = try transposed.get(&[_]usize{ j, i });
//             if (debug) {
//                 std.debug.print("x[{}, {}]={} ? x_t[{}, {}]={}\n", .{ i, j, original_val.f32, i, j, transposed_val.f32 });
//             }
//             try expect(original_val.f32 == transposed_val.f32);
//         }
//     }
// }

// test "ndarray.tanh()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 5.0));
//     defer a.deinit();
//     const res = try a.tanh(false);
//     defer res.deinit();

//     try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

//     const updated_val = try a.get(&[_]usize{ 0, 0 });
//     try expect(updated_val.f32 == 42.0);
// }

// test "ndarray.relu()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     // Create a 1D array with both negative and positive values.
//     const shape = &[_]usize{4};
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, 0.0);
//     defer a.deinit();
//     // Manually set values.
//     try a.set(&[_]usize{0}, .{ .f32 = -3.0 });
//     try a.set(&[_]usize{1}, .{ .f32 = -0.5 });
//     try a.set(&[_]usize{2}, .{ .f32 = 0.0 });
//     try a.set(&[_]usize{3}, .{ .f32 = 2.0 });

//     // Apply ReLU (non in-place).
//     const res = try a.relu(false);
//     defer res.deinit();

//     // ReLU(x) = max(0, x)
//     const expected = [_]f32{ 0.0, 0.0, 0.0, 2.0 };
//     for (0.., expected) |i, exp_val| {
//         const out = try res.get(&[_]usize{i});
//         try expect(@abs(out.f32 - exp_val) < 0.0001);
//     }
// }

// test "ndarray.sigmoid()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();

//     var arr = try NdArray.init(allocator, &[_]usize{3}, .f32); // Shape is [3]
//     defer arr.deinit();

//     // Fill the array with some values
//     try arr.set(&[_]usize{0}, .{ .f32 = 0.0 }); // Set the first element
//     try arr.set(&[_]usize{1}, .{ .f32 = 1.0 }); // Set the second element
//     try arr.set(&[_]usize{2}, .{ .f32 = -1.0 }); // Set the third element

//     const result = try nd_uops.apply_unary(arr, .SIGMOID, false);
//     defer result.deinit();

//     // Check the results
//     const val_1 = try result.get(&[_]usize{0});
//     const val_2 = try result.get(&[_]usize{1});
//     const val_3 = try result.get(&[_]usize{2});
//     try expect(@abs(val_1.f32 - 0.5) < 0.0001); // Check the first element
//     try expect(@abs(val_2.f32 - 0.73105857863) < 0.0001); // Check the second element
//     try expect(@abs(val_3.f32 - 0.26894142137) < 0.0001); // Check the third element
// }

// test "ndarray.reshape()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();

//     // Create a 2D array with shape {3, 4} and values:
//     // [[1,  2,  3,  4 ],
//     //  [5,  6,  7,  8 ],
//     //  [9, 10, 11, 12 ]]
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 3, 4 };

//     // Initialize the NdArray
//     const arr = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer arr.deinit();

//     // Define the new shape {6, 2}
//     const new_shape = &[_]usize{ 6, 2 };

//     // Reshape the array
//     const reshaped_arr = try arr.reshape(new_shape);
//     defer reshaped_arr.deinit();

//     // Assertions to verify the reshape operation
//     std.debug.assert(reshaped_arr.shape.len == 2);
//     std.debug.assert(reshaped_arr.shape[0] == 6);
//     std.debug.assert(reshaped_arr.shape[1] == 2);
// }

// test "ndarray.softmax()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     // Create a 2D array with shape {2, 3} and values: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const arr = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer arr.deinit();

//     // Apply softmax (non in-place).
//     const res = try arr.softmax(1, false);
//     defer res.deinit();

//     // Compute expected softmax values for each row.
//     const exp1_row1 = @exp(@as(f32, 1.0));
//     const exp2_row1 = @exp(@as(f32, 2.0));
//     const exp3_row1 = @exp(@as(f32, 3.0));
//     const sum_row1 = exp1_row1 + exp2_row1 + exp3_row1;
//     const expected_row1 = [_]f32{ exp1_row1 / sum_row1, exp2_row1 / sum_row1, exp3_row1 / sum_row1 };

//     const exp1_row2 = @exp(@as(f32, 4.0));
//     const exp2_row2 = @exp(@as(f32, 5.0));
//     const exp3_row2 = @exp(@as(f32, 6.0));
//     const sum_row2 = exp1_row2 + exp2_row2 + exp3_row2;
//     const expected_row2 = [_]f32{ exp1_row2 / sum_row2, exp2_row2 / sum_row2, exp3_row2 / sum_row2 };

//     // Verify the results for the first row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 0, i });
//         try std.testing.expectApproxEqAbs(expected_row1[i], out.f32, 0.0001);
//     }

//     // Verify the results for the second row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 1, i });
//         try std.testing.expectApproxEqAbs(expected_row2[i], out.f32, 0.0001);
//     }
// }

// test "ndarray.reduce.axis(null)" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();
//     var arr = try NdArray.init(allocator, &[_]usize{3}, .f32); // Shape is [3]
//     defer arr.deinit();

//     // Fill the array with some values
//     try arr.set(&[_]usize{0}, .{ .f32 = 1.0 });
//     try arr.set(&[_]usize{1}, .{ .f32 = 2.0 });
//     try arr.set(&[_]usize{2}, .{ .f32 = 3.0 });

//     // Test SUM
//     const sum_result = try nd_rops.reduce(arr, .SUM, null);
//     defer sum_result.deinit();
//     const sum_val = try sum_result.get(&[_]usize{0});
//     try expect(sum_val.f32 == 6.0);

//     // Test MIN
//     const min_result = try nd_rops.reduce(arr, .MIN, null);
//     defer min_result.deinit();
//     const min_val = try min_result.get(&[_]usize{0});
//     try expect(min_val.f32 == 1.0);

//     // Test MAX
//     const max_result = try nd_rops.reduce(arr, .MAX, null);
//     defer max_result.deinit();
//     const max_val = try max_result.get(&[_]usize{0});
//     try expect(max_val.f32 == 3.0);

//     // Test MEAN
//     const mean_result = try nd_rops.reduce(arr, .MEAN, null);
//     defer mean_result.deinit();
//     const mean_val = try mean_result.get(&[_]usize{0});
//     try expect(mean_val.f32 == 2.0);

//     // Test PROD
//     const prod_result = try nd_rops.reduce(arr, .PROD, null);
//     defer prod_result.deinit();
//     const prod_val = try prod_result.get(&[_]usize{0});
//     try expect(prod_val.f32 == 6.0);
// }

// test "ndarray.reduce.axis(0)" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const arr = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer arr.deinit();

//     // Test SUM along axis 1
//     const sum_result = try nd_rops.reduce(arr, .SUM, 0);
//     defer sum_result.deinit();

//     const sum_0 = try sum_result.get(&[_]usize{0});
//     const sum_1 = try sum_result.get(&[_]usize{1});
//     const sum_2 = try sum_result.get(&[_]usize{2});
//     try expect(sum_0.f32 == 5.0);
//     try expect(sum_1.f32 == 7.0);
//     try expect(sum_2.f32 == 9.0);

//     // Test MAX along axis 1
//     const max_result = try nd_rops.reduce(arr, .MAX, 0);
//     defer max_result.deinit();

//     const max_0 = try max_result.get(&[_]usize{0});
//     const max_1 = try max_result.get(&[_]usize{1});
//     const max_2 = try max_result.get(&[_]usize{2});
//     try expect(max_0.f32 == 4.0);
//     try expect(max_1.f32 == 5.0);
//     try expect(max_2.f32 == 6.0);

//     // Test MIN along axis 1
//     const min_result = try nd_rops.reduce(arr, .MIN, 0);
//     defer min_result.deinit();

//     const min_0 = try min_result.get(&[_]usize{0});
//     const min_1 = try min_result.get(&[_]usize{1});
//     const min_2 = try min_result.get(&[_]usize{2});
//     try expect(min_0.f32 == 1.0);
//     try expect(min_1.f32 == 2.0);
//     try expect(min_2.f32 == 3.0);
// }

// test "ndarray.reduce.axis(1)" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const arr = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, data_a);
//     defer arr.deinit();

//     // Test SUM along axis 1
//     const sum_result = try nd_rops.reduce(arr, .SUM, 1);
//     defer sum_result.deinit();

//     const sum_0 = try sum_result.get(&[_]usize{0});
//     const sum_1 = try sum_result.get(&[_]usize{1});
//     try expect(sum_0.f32 == 6.0);
//     try expect(sum_1.f32 == 15.0);

//     // Test MAX along axis 1
//     const max_result = try nd_rops.reduce(arr, .MAX, 1);
//     defer max_result.deinit();

//     const max_0 = try max_result.get(&[_]usize{0});
//     const max_1 = try max_result.get(&[_]usize{1});
//     try expect(max_0.f32 == 3.0);
//     try expect(max_1.f32 == 6.0);

//     // Test MIN along axis 1
//     const min_result = try nd_rops.reduce(arr, .MIN, 1);
//     defer min_result.deinit();

//     const min_0 = try min_result.get(&[_]usize{0});
//     const min_1 = try min_result.get(&[_]usize{1});
//     try expect(min_0.f32 == 1.0);
//     try expect(min_1.f32 == 4.0);
// }

// test "ndarray.set()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();

//     try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

//     const updated_val = try a.get(&[_]usize{ 0, 0 });
//     try expect(updated_val.f32 == 42.0);
// }

// test "ndarray.get()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();

//     const value = try a.get(&[_]usize{ 0, 0 });
//     try expect(value.f32 == 1.0);
// }

// test "ndarray.fill()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 0.0));
//     defer a.deinit();

//     try a.fill(42.0);

//     const updated_val = try a.get(&[_]usize{ 0, 0 });
//     try expect(updated_val.f32 == 42.0);
// }

// test "ndarray.add()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();
//     const b = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer b.deinit();

//     const sum = try a.add(b, false);
//     defer sum.deinit();

//     const sum_val = try sum.get(&[_]usize{ 0, 0 });
//     try expect(sum_val.f32 == 3.0);
// }

// test "ndarray.sub()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();
//     const b = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer b.deinit();

//     const sub = try a.sub(b, false);
//     defer sub.deinit();

//     const sub_val = try sub.get(&[_]usize{ 0, 0 });
//     try expect(sub_val.f32 == -1.0);
// }

// test "ndarray.mul()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();
//     const b = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer b.deinit();

//     const mul = try a.mul(b, false);
//     defer mul.deinit();

//     const mul_val = try mul.get(&[_]usize{ 0, 0 });
//     try expect(mul_val.f32 == 2.0);
// }

// test "ndarray.div()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();
//     const b = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer b.deinit();

//     const div = try a.div(b, false);
//     defer div.deinit();

//     const div_val = try div.get(&[_]usize{ 0, 0 });
//     try expect(div_val.f32 == 0.5);
// }

// test "ndarray.pow()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const shape = &[_]usize{ 5, 3 };
//     const a = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer a.deinit();
//     const b = try NdArray.from_value(allocator, shape, dtypes.DataType.f32, @as(f32, 3.0));
//     defer b.deinit();

//     const pow = try a.pow(b, false);
//     defer pow.deinit();

//     const pow_val = try pow.get(&[_]usize{ 0, 0 });
//     try expect(pow_val.f32 == 8.0);
// }

// test "ndarray.add_scalar()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const test_value = 5;
//     const test_scalar = 3;
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const shape = &[_]usize{ 5, 3 };
//         const value = switch (dtype) {
//             .f32 => @as(f32, test_value),
//             .f64 => @as(f64, test_value),
//             .i32 => @as(i32, test_value),
//             .i64 => @as(i64, test_value),
//             .i16 => @as(i16, test_value),
//             .i8 => @as(i8, test_value),
//             .u32 => @as(u32, test_value),
//             .u64 => @as(u64, test_value),
//             .u16 => @as(u16, test_value),
//             .u8 => @as(u8, test_value),
//         };

//         const a = try NdArray.from_value(allocator, shape, dtype, value);
//         defer a.deinit();

//         // Test non-inplace operation
//         const result = try a.add_scalar(test_scalar, false);
//         defer result.deinit();

//         if (debug) {
//             try result.info();
//         }

//         const result_val = try result.get(&[_]usize{ 0, 0 });
//         const expected_result = switch (dtype) {
//             .f32 => ScalarValue{ .f32 = @as(f32, test_value) + @as(f32, test_scalar) },
//             .f64 => ScalarValue{ .f64 = @as(f64, test_value) + @as(f64, test_scalar) },
//             .i32 => ScalarValue{ .i32 = @as(i32, test_value) + @as(i32, test_scalar) },
//             .i64 => ScalarValue{ .i64 = @as(i64, test_value) + @as(i64, test_scalar) },
//             .i16 => ScalarValue{ .i16 = @as(i16, test_value) + @as(i16, test_scalar) },
//             .i8 => ScalarValue{ .i8 = @as(i8, test_value) + @as(i8, test_scalar) },
//             .u32 => ScalarValue{ .u32 = @as(u32, test_value) + @as(u32, test_scalar) },
//             .u64 => ScalarValue{ .u64 = @as(u64, test_value) + @as(u64, test_scalar) },
//             .u16 => ScalarValue{ .u16 = @as(u16, test_value) + @as(u16, test_scalar) },
//             .u8 => ScalarValue{ .u8 = @as(u8, test_value) + @as(u8, test_scalar) },
//         };

//         switch (dtype) {
//             .f32 => try expect(result_val.f32 == expected_result.f32),
//             .f64 => try expect(result_val.f64 == expected_result.f64),
//             .i32 => try expect(result_val.i32 == expected_result.i32),
//             .i64 => try expect(result_val.i64 == expected_result.i64),
//             .i16 => try expect(result_val.i16 == expected_result.i16),
//             .i8 => try expect(result_val.i8 == expected_result.i8),
//             .u32 => try expect(result_val.u32 == expected_result.u32),
//             .u64 => try expect(result_val.u64 == expected_result.u64),
//             .u16 => try expect(result_val.u16 == expected_result.u16),
//             .u8 => try expect(result_val.u8 == expected_result.u8),
//         }

//         // Test inplace operation
//         const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
//         defer a_copy.deinit();

//         const inplace_result = try a_copy.add_scalar(test_scalar, true);
//         defer if (inplace_result != a_copy) inplace_result.deinit();

//         if (debug) {
//             try inplace_result.info();
//         }

//         const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
//         switch (dtype) {
//             .f32 => try expect(inplace_val.f32 == expected_result.f32),
//             .f64 => try expect(inplace_val.f64 == expected_result.f64),
//             .i32 => try expect(inplace_val.i32 == expected_result.i32),
//             .i64 => try expect(inplace_val.i64 == expected_result.i64),
//             .i16 => try expect(inplace_val.i16 == expected_result.i16),
//             .i8 => try expect(inplace_val.i8 == expected_result.i8),
//             .u32 => try expect(inplace_val.u32 == expected_result.u32),
//             .u64 => try expect(inplace_val.u64 == expected_result.u64),
//             .u16 => try expect(inplace_val.u16 == expected_result.u16),
//             .u8 => try expect(inplace_val.u8 == expected_result.u8),
//         }

//         // Ensure that the inplace operation modified the original array
//         try expect(inplace_result == a_copy);
//     }
// }

// test "ndarray.sub_scalar()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const test_value = 5;
//     const test_scalar = 3;
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const shape = &[_]usize{ 5, 3 };
//         const value = switch (dtype) {
//             .f32 => @as(f32, test_value),
//             .f64 => @as(f64, test_value),
//             .i32 => @as(i32, test_value),
//             .i64 => @as(i64, test_value),
//             .i16 => @as(i16, test_value),
//             .i8 => @as(i8, test_value),
//             .u32 => @as(u32, test_value),
//             .u64 => @as(u64, test_value),
//             .u16 => @as(u16, test_value),
//             .u8 => @as(u8, test_value),
//         };

//         const a = try NdArray.from_value(allocator, shape, dtype, value);
//         defer a.deinit();

//         // Test non-inplace operation
//         const result = try a.sub_scalar(test_scalar, false);
//         defer result.deinit();

//         if (debug) {
//             try result.info();
//         }

//         const result_val = try result.get(&[_]usize{ 0, 0 });
//         const expected_result = switch (dtype) {
//             .f32 => ScalarValue{ .f32 = @as(f32, test_value) - @as(f32, test_scalar) },
//             .f64 => ScalarValue{ .f64 = @as(f64, test_value) - @as(f64, test_scalar) },
//             .i32 => ScalarValue{ .i32 = @as(i32, test_value) - @as(i32, test_scalar) },
//             .i64 => ScalarValue{ .i64 = @as(i64, test_value) - @as(i64, test_scalar) },
//             .i16 => ScalarValue{ .i16 = @as(i16, test_value) - @as(i16, test_scalar) },
//             .i8 => ScalarValue{ .i8 = @as(i8, test_value) - @as(i8, test_scalar) },
//             .u32 => blk: {
//                 if (test_value >= test_scalar) {
//                     break :blk ScalarValue{ .u32 = @as(u32, test_value) - @as(u32, test_scalar) };
//                 } else {
//                     std.debug.print("Warning: Skipping u32 subtraction test because test_value < test_scalar\n", .{});
//                     break :blk ScalarValue{ .u32 = 0 }; // Return a dummy value
//                 }
//             },
//             .u64 => blk: {
//                 if (test_value >= test_scalar) {
//                     break :blk ScalarValue{ .u64 = @as(u64, test_value) - @as(u64, test_scalar) };
//                 } else {
//                     std.debug.print("Warning: Skipping u64 subtraction test because test_value < test_scalar\n", .{});
//                     break :blk ScalarValue{ .u64 = 0 }; // Return a dummy value
//                 }
//             },
//             .u16 => blk: {
//                 if (test_value >= test_scalar) {
//                     break :blk ScalarValue{ .u16 = @as(u16, test_value) - @as(u16, test_scalar) };
//                 } else {
//                     std.debug.print("Warning: Skipping u16 subtraction test because test_value < test_scalar\n", .{});
//                     break :blk ScalarValue{ .u16 = 0 }; // Return a dummy value
//                 }
//             },
//             .u8 => blk: {
//                 if (test_value >= test_scalar) {
//                     break :blk ScalarValue{ .u8 = @as(u8, test_value) - @as(u8, test_scalar) };
//                 } else {
//                     std.debug.print("Warning: Skipping u8 subtraction test because test_value < test_scalar\n", .{});
//                     break :blk ScalarValue{ .u8 = 0 }; // Return a dummy value
//                 }
//             },
//         };

//         switch (dtype) {
//             .f32 => try expect(result_val.f32 == expected_result.f32),
//             .f64 => try expect(result_val.f64 == expected_result.f64),
//             .i32 => try expect(result_val.i32 == expected_result.i32),
//             .i64 => try expect(result_val.i64 == expected_result.i64),
//             .i16 => try expect(result_val.i16 == expected_result.i16),
//             .i8 => try expect(result_val.i8 == expected_result.i8),
//             .u32 => try expect(result_val.u32 == expected_result.u32),
//             .u64 => try expect(result_val.u64 == expected_result.u64),
//             .u16 => try expect(result_val.u16 == expected_result.u16),
//             .u8 => try expect(result_val.u8 == expected_result.u8),
//         }

//         // Test inplace operation
//         const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
//         defer a_copy.deinit();

//         const inplace_result = try a_copy.sub_scalar(test_scalar, true);
//         defer if (inplace_result != a_copy) inplace_result.deinit();

//         if (debug) {
//             try inplace_result.info();
//         }

//         const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
//         switch (dtype) {
//             .f32 => try expect(inplace_val.f32 == expected_result.f32),
//             .f64 => try expect(inplace_val.f64 == expected_result.f64),
//             .i32 => try expect(inplace_val.i32 == expected_result.i32),
//             .i64 => try expect(inplace_val.i64 == expected_result.i64),
//             .i16 => try expect(inplace_val.i16 == expected_result.i16),
//             .i8 => try expect(inplace_val.i8 == expected_result.i8),
//             .u32 => try expect(inplace_val.u32 == expected_result.u32),
//             .u64 => try expect(inplace_val.u64 == expected_result.u64),
//             .u16 => try expect(inplace_val.u16 == expected_result.u16),
//             .u8 => try expect(inplace_val.u8 == expected_result.u8),
//         }

//         // Ensure that the inplace operation modified the original array
//         try expect(inplace_result == a_copy);
//     }
// }

// test "ndarray.mul_scalar(3.0)" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const test_value = 5.0;
//     const test_scalar = 3.0;
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const shape = &[_]usize{ 5, 3 };
//         const value = switch (dtype) {
//             .f32 => @as(f32, test_value),
//             .f64 => @as(f64, test_value),
//             .i32 => @as(i32, test_value),
//             .i64 => @as(i64, test_value),
//             .i16 => @as(i16, test_value),
//             .i8 => @as(i8, test_value),
//             .u32 => @as(u32, test_value),
//             .u64 => @as(u64, test_value),
//             .u16 => @as(u16, test_value),
//             .u8 => @as(u8, test_value),
//         };

//         const a = try NdArray.from_value(allocator, shape, dtype, value);
//         defer a.deinit();

//         // Test non-inplace operation
//         const result = try a.mul_scalar(test_scalar, false);
//         defer result.deinit();

//         if (debug) {
//             try result.info();
//         }

//         const result_val = try result.get(&[_]usize{ 0, 0 });
//         const expected_result = switch (dtype) {
//             .f32 => ScalarValue{ .f32 = @as(f32, test_value) * @as(f32, test_scalar) },
//             .f64 => ScalarValue{ .f64 = @as(f64, test_value) * @as(f64, test_scalar) },
//             .i32 => ScalarValue{ .i32 = @as(i32, test_value) * @as(i32, test_scalar) },
//             .i64 => ScalarValue{ .i64 = @as(i64, test_value) * @as(i64, test_scalar) },
//             .i16 => ScalarValue{ .i16 = @as(i16, test_value) * @as(i16, test_scalar) },
//             .i8 => ScalarValue{ .i8 = @as(i8, test_value) * @as(i8, test_scalar) },
//             .u32 => ScalarValue{ .u32 = @as(u32, test_value) * @as(u32, test_scalar) },
//             .u64 => ScalarValue{ .u64 = @as(u64, test_value) * @as(u64, test_scalar) },
//             .u16 => ScalarValue{ .u16 = @as(u16, test_value) * @as(u16, test_scalar) },
//             .u8 => ScalarValue{ .u8 = @as(u8, test_value) * @as(u8, test_scalar) },
//         };

//         switch (dtype) {
//             .f32 => try expect(result_val.f32 == expected_result.f32),
//             .f64 => try expect(result_val.f64 == expected_result.f64),
//             .i32 => try expect(result_val.i32 == expected_result.i32),
//             .i64 => try expect(result_val.i64 == expected_result.i64),
//             .i16 => try expect(result_val.i16 == expected_result.i16),
//             .i8 => try expect(result_val.i8 == expected_result.i8),
//             .u32 => try expect(result_val.u32 == expected_result.u32),
//             .u64 => try expect(result_val.u64 == expected_result.u64),
//             .u16 => try expect(result_val.u16 == expected_result.u16),
//             .u8 => try expect(result_val.u8 == expected_result.u8),
//         }

//         // Test inplace operation
//         const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
//         defer a_copy.deinit();

//         const inplace_result = try a_copy.mul_scalar(test_scalar, true);
//         defer if (inplace_result != a_copy) inplace_result.deinit();

//         if (debug) {
//             try inplace_result.info();
//         }

//         const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
//         switch (dtype) {
//             .f32 => try expect(inplace_val.f32 == expected_result.f32),
//             .f64 => try expect(inplace_val.f64 == expected_result.f64),
//             .i32 => try expect(inplace_val.i32 == expected_result.i32),
//             .i64 => try expect(inplace_val.i64 == expected_result.i64),
//             .i16 => try expect(inplace_val.i16 == expected_result.i16),
//             .i8 => try expect(inplace_val.i8 == expected_result.i8),
//             .u32 => try expect(inplace_val.u32 == expected_result.u32),
//             .u64 => try expect(inplace_val.u64 == expected_result.u64),
//             .u16 => try expect(inplace_val.u16 == expected_result.u16),
//             .u8 => try expect(inplace_val.u8 == expected_result.u8),
//         }

//         // Ensure that the inplace operation modified the original array
//         try expect(inplace_result == a_copy);
//     }
// }

// test "ndarray.mul_scalar(0.005)" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const test_value = 5.0;
//     const test_scalar = 0.005;
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const shape = &[_]usize{ 5, 3 };
//         const value = switch (dtype) {
//             .f32 => @as(f32, test_value),
//             .f64 => @as(f64, test_value),
//             .i32 => @as(i32, test_value),
//             .i64 => @as(i64, test_value),
//             .i16 => @as(i16, test_value),
//             .i8 => @as(i8, test_value),
//             .u32 => @as(u32, test_value),
//             .u64 => @as(u64, test_value),
//             .u16 => @as(u16, test_value),
//             .u8 => @as(u8, test_value),
//         };

//         const a = try NdArray.from_value(allocator, shape, dtype, value);
//         defer a.deinit();

//         // Test non-inplace operation
//         const result = try a.mul_scalar(test_scalar, false);
//         defer result.deinit();

//         if (debug) {
//             try result.info();
//         }

//         const result_val = try result.get(&[_]usize{ 0, 0 });
//         const expected_result = switch (dtype) {
//             .f32 => ScalarValue{ .f32 = @as(f32, test_value) * @as(f32, test_scalar) },
//             .f64 => ScalarValue{ .f64 = @as(f64, test_value) * @as(f64, test_scalar) },
//             .i32 => ScalarValue{ .i32 = @as(i32, test_value) * @as(i32, @intFromFloat(test_scalar)) },
//             .i64 => ScalarValue{ .i64 = @as(i64, test_value) * @as(i64, @intFromFloat(test_scalar)) },
//             .i16 => ScalarValue{ .i16 = @as(i16, test_value) * @as(i16, @intFromFloat(test_scalar)) },
//             .i8 => ScalarValue{ .i8 = @as(i8, test_value) * @as(i8, @intFromFloat(test_scalar)) },
//             .u32 => ScalarValue{ .u32 = @as(u32, test_value) * @as(u32, @intFromFloat(test_scalar)) },
//             .u64 => ScalarValue{ .u64 = @as(u64, test_value) * @as(u64, @intFromFloat(test_scalar)) },
//             .u16 => ScalarValue{ .u16 = @as(u16, test_value) * @as(u16, @intFromFloat(test_scalar)) },
//             .u8 => ScalarValue{ .u8 = @as(u8, test_value) * @as(u8, @intFromFloat(test_scalar)) },
//         };

//         switch (dtype) {
//             .f32 => try expect(result_val.f32 == expected_result.f32),
//             .f64 => try expect(result_val.f64 == expected_result.f64),
//             .i32 => try expect(result_val.i32 == expected_result.i32),
//             .i64 => try expect(result_val.i64 == expected_result.i64),
//             .i16 => try expect(result_val.i16 == expected_result.i16),
//             .i8 => try expect(result_val.i8 == expected_result.i8),
//             .u32 => try expect(result_val.u32 == expected_result.u32),
//             .u64 => try expect(result_val.u64 == expected_result.u64),
//             .u16 => try expect(result_val.u16 == expected_result.u16),
//             .u8 => try expect(result_val.u8 == expected_result.u8),
//         }

//         // Test inplace operation
//         const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
//         defer a_copy.deinit();

//         const inplace_result = try a_copy.mul_scalar(test_scalar, true);
//         defer if (inplace_result != a_copy) inplace_result.deinit();

//         if (debug) {
//             try inplace_result.info();
//         }

//         const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
//         switch (dtype) {
//             .f32 => try expect(inplace_val.f32 == expected_result.f32),
//             .f64 => try expect(inplace_val.f64 == expected_result.f64),
//             .i32 => try expect(inplace_val.i32 == expected_result.i32),
//             .i64 => try expect(inplace_val.i64 == expected_result.i64),
//             .i16 => try expect(inplace_val.i16 == expected_result.i16),
//             .i8 => try expect(inplace_val.i8 == expected_result.i8),
//             .u32 => try expect(inplace_val.u32 == expected_result.u32),
//             .u64 => try expect(inplace_val.u64 == expected_result.u64),
//             .u16 => try expect(inplace_val.u16 == expected_result.u16),
//             .u8 => try expect(inplace_val.u8 == expected_result.u8),
//         }

//         // Ensure that the inplace operation modified the original array
//         try expect(inplace_result == a_copy);
//     }
// }

// test "ndarray.div_scalar()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const test_value = 5;
//     const test_scalar = 3;
//     const data_types = &[_]dtypes.DataType{
//         .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
//     };

//     for (data_types) |dtype| {
//         const shape = &[_]usize{ 5, 3 };
//         const value = switch (dtype) {
//             .f32 => @as(f32, test_value),
//             .f64 => @as(f64, test_value),
//             .i32 => @as(i32, test_value),
//             .i64 => @as(i64, test_value),
//             .i16 => @as(i16, test_value),
//             .i8 => @as(i8, test_value),
//             .u32 => @as(u32, test_value),
//             .u64 => @as(u64, test_value),
//             .u16 => @as(u16, test_value),
//             .u8 => @as(u8, test_value),
//         };

//         const a = try NdArray.from_value(allocator, shape, dtype, value);
//         defer a.deinit();

//         // Test non-inplace operation
//         const result = try a.div_scalar(test_scalar, false);
//         defer result.deinit();

//         if (debug) {
//             try result.info();
//         }

//         const result_val = try result.get(&[_]usize{ 0, 0 });
//         const expected_result = switch (dtype) {
//             .f32 => ScalarValue{ .f32 = @as(f32, test_value) / @as(f32, test_scalar) },
//             .f64 => ScalarValue{ .f64 = @as(f64, test_value) / @as(f64, test_scalar) },
//             .i32 => ScalarValue{ .i32 = @divTrunc(@as(i32, test_value), @as(i32, test_scalar)) },
//             .i64 => ScalarValue{ .i64 = @divTrunc(@as(i64, test_value), @as(i64, test_scalar)) },
//             .i16 => ScalarValue{ .i16 = @divTrunc(@as(i16, test_value), @as(i16, test_scalar)) },
//             .i8 => ScalarValue{ .i8 = @divTrunc(@as(i8, test_value), @as(i8, test_scalar)) },
//             .u32 => ScalarValue{ .u32 = @as(u32, test_value) / @as(u32, test_scalar) },
//             .u64 => ScalarValue{ .u64 = @as(u64, test_value) / @as(u64, test_scalar) },
//             .u16 => ScalarValue{ .u16 = @as(u16, test_value) / @as(u16, test_scalar) },
//             .u8 => ScalarValue{ .u8 = @as(u8, test_value) / @as(u8, test_scalar) },
//         };

//         switch (dtype) {
//             .f32 => try expect(result_val.f32 == expected_result.f32),
//             .f64 => try expect(result_val.f64 == expected_result.f64),
//             .i32 => try expect(result_val.i32 == expected_result.i32),
//             .i64 => try expect(result_val.i64 == expected_result.i64),
//             .i16 => try expect(result_val.i16 == expected_result.i16),
//             .i8 => try expect(result_val.i8 == expected_result.i8),
//             .u32 => try expect(result_val.u32 == expected_result.u32),
//             .u64 => try expect(result_val.u64 == expected_result.u64),
//             .u16 => try expect(result_val.u16 == expected_result.u16),
//             .u8 => try expect(result_val.u8 == expected_result.u8),
//         }

//         // Test inplace operation
//         const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
//         defer a_copy.deinit();

//         const inplace_result = try a_copy.div_scalar(test_scalar, true);
//         defer if (inplace_result != a_copy) inplace_result.deinit();

//         if (debug) {
//             try inplace_result.info();
//         }

//         const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
//         switch (dtype) {
//             .f32 => try expect(inplace_val.f32 == expected_result.f32),
//             .f64 => try expect(inplace_val.f64 == expected_result.f64),
//             .i32 => try expect(inplace_val.i32 == expected_result.i32),
//             .i64 => try expect(inplace_val.i64 == expected_result.i64),
//             .i16 => try expect(inplace_val.i16 == expected_result.i16),
//             .i8 => try expect(inplace_val.i8 == expected_result.i8),
//             .u32 => try expect(inplace_val.u32 == expected_result.u32),
//             .u64 => try expect(inplace_val.u64 == expected_result.u64),
//             .u16 => try expect(inplace_val.u16 == expected_result.u16),
//             .u8 => try expect(inplace_val.u8 == expected_result.u8),
//         }

//         // Ensure that the inplace operation modified the original array
//         try expect(inplace_result == a_copy);
//     }
// }

// test "ndarray.matmul()" {
//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     const allocator = gpa.allocator();
//     defer _ = gpa.deinit();

//     const debug = false;
//     const a_shape = &[_]usize{ 3, 2 };
//     const a = try NdArray.from_value(allocator, a_shape, dtypes.DataType.f32, @as(f32, 1.0));
//     defer a.deinit();

//     const b_shape = &[_]usize{ 2, 4 };
//     const b = try NdArray.from_value(allocator, b_shape, dtypes.DataType.f32, @as(f32, 2.0));
//     defer b.deinit();

//     const matmul_result = try a.matmul(b, 1.0, 1.0, false, false);
//     defer matmul_result.deinit();

//     if (debug) {
//         try a.print();
//         try b.print();
//         try matmul_result.print();
//     }

//     const matmul_val = try matmul_result.get(&[_]usize{ 0, 0 });
//     try expect(matmul_val.f32 == 4.0);
// }

// test "NdArray.cast: convert f32 to i32" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarray
//     const data = [_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const ndarray = try NdArray.from_data(allocator, shape, DataType.f32, &data);
//     defer ndarray.deinit();

//     // Cast the ndarray to i32
//     const casted = try ndarray.cast(DataType.i32);
//     defer casted.deinit();

//     // Expected result after casting to i32 (truncates floating-point values)
//     const expected_data = &[_]i32{ 1, 2, 3, 4, 5, 6 };

//     // Verify the result using a loop
//     const total_size = utils.compute_size(shape);
//     for (0..total_size) |i| {
//         const row = i / shape[1]; // Calculate the row index
//         const col = i % shape[1]; // Calculate the column index
//         const casted_value = try casted.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.i32);
//     }
// }

// test "NdArray.cast: convert f32 to f64" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarray
//     const data = [_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const ndarray = try NdArray.from_data(allocator, shape, DataType.f32, &data);
//     defer ndarray.deinit();

//     // Cast the ndarray to f64
//     const casted = try ndarray.cast(DataType.f64);
//     defer casted.deinit();

//     // Expected result after casting to f64
//     const expected_data = &[_]f64{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };

//     // Verify the result using a loop
//     const total_size = utils.compute_size(shape);
//     for (0..total_size) |i| {
//         const row = i / shape[1]; // Calculate the row index
//         const col = i % shape[1]; // Calculate the column index
//         const casted_value = try casted.get(&[_]usize{ row, col });

//         // Use approxEqAbs to compare floating-point values with a tolerance
//         const tolerance = 1e-6; // Adjust the tolerance as needed
//         const is_equal = std.math.approxEqAbs(f64, expected_data[i], casted_value.f64, tolerance);

//         // Log the values for debugging
//         // std.debug.print("Expected: {d}, Got: {d}, Tolerance: {d}\n", .{ expected_data[i], casted_value.f64, tolerance });

//         // Check if the values are approximately equal
//         try std.testing.expect(is_equal);
//     }
// }

// test "NdArray.cast: convert i32 to i64" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarray
//     const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };

//     // Create the ndarray from data
//     const ndarray = try NdArray.from_data(allocator, shape, DataType.i32, &data);
//     defer ndarray.deinit();

//     // Cast the ndarray to i64
//     const casted = try ndarray.cast(DataType.i64);
//     defer casted.deinit();

//     // Expected result after casting to i64
//     const expected_data = &[_]i64{ 1, 2, 3, 4, 5, 6 };

//     // Verify the result using a loop
//     const total_size = utils.compute_size(shape);
//     for (0..total_size) |i| {
//         const row = i / shape[1]; // Calculate the row index
//         const col = i % shape[1]; // Calculate the column index
//         const casted_value = try casted.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.i64);
//     }
// }

// test "NdArray.cast: convert f64 to f32" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const data = [_]f64{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
//     const shape = &[_]usize{ 2, 3 };

//     const ndarray = try NdArray.from_data(allocator, shape, .f64, &data);
//     defer ndarray.deinit();

//     const casted = try ndarray.cast(.f32);
//     defer casted.deinit();

//     const expected_data = &[_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
//     const total_size = utils.compute_size(shape);

//     for (0..total_size) |i| {
//         const row = i / shape[1];
//         const col = i % shape[1];
//         const casted_value = try casted.get(&[_]usize{ row, col });
//         const tolerance = 1e-6;
//         const is_equal = std.math.approxEqAbs(f32, expected_data[i], casted_value.f32, tolerance);
//         try std.testing.expect(is_equal);
//     }
// }

// test "NdArray.cast: convert i32 to f64" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const data = [_]i32{ 1, -2, 3, 4, -5, 6 };
//     const shape = &[_]usize{ 2, 3 };

//     const ndarray = try NdArray.from_data(allocator, shape, .i32, &data);
//     defer ndarray.deinit();

//     const casted = try ndarray.cast(.f64);
//     defer casted.deinit();

//     const expected_data = &[_]f64{ 1.0, -2.0, 3.0, 4.0, -5.0, 6.0 };
//     const total_size = utils.compute_size(shape);

//     for (0..total_size) |i| {
//         const row = i / shape[1];
//         const col = i % shape[1];
//         const casted_value = try casted.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.f64);
//     }
// }

// test "NdArray.slice: slice rows" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };

//     const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
//     defer ndarray.deinit();

//     const sliced = try ndarray.slice(0, 1, 3); // Slice rows 1 to 3
//     defer sliced.deinit();

//     const expected_data = &[_]f32{ 4, 5, 6, 7, 8, 9 };
//     const sliced_shape = &[_]usize{ 2, 3 };
//     const total_size = utils.compute_size(sliced_shape);

//     for (0..total_size) |i| {
//         const row = i / sliced_shape[1];
//         const col = i % sliced_shape[1];
//         const casted_value = try sliced.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.f32);
//     }
// }

// test "NdArray.slice: slice columns" {
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const shape = &[_]usize{ 4, 3 };

//     const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
//     defer ndarray.deinit();

//     const sliced = try ndarray.slice(1, 1, 2); // Slice columns 1 to 2
//     defer sliced.deinit();

//     const expected_data = &[_]f32{ 2, 5, 8, 11 };
//     const sliced_shape = &[_]usize{ 4, 1 };
//     const total_size = utils.compute_size(sliced_shape);

//     for (0..total_size) |i| {
//         const row = i / sliced_shape[1];
//         const col = i % sliced_shape[1];
//         const casted_value = try sliced.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.f32);
//     }
// }

// test "NdArray.equal: element-wise equality comparison" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarrays
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
//     const data_b = [_]f32{ 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0 };
//     const shape = &[_]usize{ 4, 3 };

//     // Create the ndarrays from data
//     const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
//     const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
//     defer ndarray_a.deinit();
//     defer ndarray_b.deinit();

//     // Perform the equality comparison
//     const result = try ndarray_a.equal(ndarray_b, false);
//     defer result.deinit();

//     // Expected result (1 where elements are equal, 0 otherwise)
//     const expected_data = &[_]i32{ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
//     const total_size = utils.compute_size(shape);

//     for (0..total_size) |i| {
//         const row = i / result.shape[1];
//         const col = i % result.shape[1];
//         const casted_value = try result.get(&[_]usize{ row, col });
//         try std.testing.expectEqual(expected_data[i], casted_value.i32);
//     }
// }

// test "NdArray.greater_than: element-wise greater than comparison" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarrays
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 0, 11 };
//     const data_b = [_]f32{ 1, 1, 4, 3, 6, 5, 1, 9 };
//     const shape = &[_]usize{ 4, 2 };

//     // Create the ndarrays from data
//     const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
//     const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
//     defer ndarray_a.deinit();
//     defer ndarray_b.deinit();

//     // Perform the greater-than comparison
//     const result = try ndarray_a.greater_than(ndarray_b, false);
//     defer result.deinit();

//     // Expected result (1 where elements in A are greater than B, 0 otherwise)
//     const expected_data = &[_]i32{
//         0, 1, 0,
//         1, 0, 1,
//         0, 1, 0,
//         1, 1, 1,
//     };

//     const total_size = utils.compute_size(result.shape);
//     for (0..total_size) |i| {
//         const row = i / result.shape[1];
//         const col = i % result.shape[1];
//         const casted_value = try result.get(&[_]usize{ row, col });
//         // std.debug.print("{} ? {} \n", .{ expected_data[i], casted_value.i32 });
//         try std.testing.expectEqual(expected_data[i], casted_value.i32);
//     }
// }

// test "NdArray.less_than: element-wise less than comparison" {
//     // Initialize an allocator for the test
//     var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     // Original data for the ndarrays
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 8 };
//     const data_b = [_]f32{ 1, 3, 2, 5, 8, 8.5 };
//     const shape = &[_]usize{ 3, 2 };

//     // Create the ndarrays from data
//     const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
//     const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
//     defer ndarray_a.deinit();
//     defer ndarray_b.deinit();

//     // Perform the less-than comparison
//     const result = try ndarray_a.less_than(ndarray_b, false);
//     defer result.deinit();

//     // Expected result (1 where elements in A are less than B, 0 otherwise)
//     const expected_data = &[_]i32{ 0, 1, 0, 1, 1, 1, 1 };

//     const total_size = utils.compute_size(result.shape);
//     for (0..total_size) |i| {
//         const row = i / result.shape[1];
//         const col = i % result.shape[1];
//         const casted_value = try result.get(&[_]usize{ row, col });
//         // std.debug.print("{} ? {} \n", .{ expected_data[i], casted_value.i32 });
//         try std.testing.expectEqual(expected_data[i], casted_value.i32);
//     }
// }
