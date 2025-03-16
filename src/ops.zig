const std = @import("std");
const zorch = @import("zorch.zig");

const ops = zorch.ops;
const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;
const NdArray = zorch.NdArray;
pub const DataType = dtypes.DataType;
pub const NumericUnion = dtypes.NumericUnion;

///=============================================================================
/// Binary operations with ndarray
///=============================================================================
pub const NdArrayBinaryOperation = enum {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    MIN,
    MAX,
    MOD,
    EQ, // Equal to
    GT, // Greater than
    LT, // Less than
};

pub fn compute_strides(shape: []const usize, strides: []usize) void {
    var stride: usize = 1;
    for (0..shape.len) |i| {
        strides[shape.len - 1 - i] = stride;
        stride *= shape[shape.len - 1 - i];
    }
}

pub fn is_compatible_for_broadcast(shape1: []const usize, shape2: []const usize) bool {
    const max_rank = @max(shape1.len, shape2.len);

    for (0..max_rank) |i| {
        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;

        if (dim1 != dim2 and dim1 != 1 and dim2 != 1) {
            return false;
        }
    }
    return true;
}

pub fn compute_broadcast_shape(shape1: []const usize, shape2: []const usize, allocator: std.mem.Allocator) ![]usize {
    const max_rank = @max(shape1.len, shape2.len);
    var result = try allocator.alloc(usize, max_rank);
    errdefer allocator.free(result);

    for (0..max_rank) |i| {
        const dim1 = if (i < shape1.len) shape1[shape1.len - 1 - i] else 1;
        const dim2 = if (i < shape2.len) shape2[shape2.len - 1 - i] else 1;
        result[max_rank - 1 - i] = @max(dim1, dim2);
    }

    return result;
}

fn apply_binary_op(op: NdArrayBinaryOperation, a: anytype, b: anytype) @TypeOf(a) {
    // std.debug.print(" {} {} {}\n", .{ a, op, b });
    return switch (op) {
        .ADD => a + b,
        .SUB => a - b,
        .MUL => a * b,
        .DIV => switch (@TypeOf(a)) {
            i8, i16, i32, i64 => @divTrunc(a, b), // Use @divTrunc for signed integers
            else => a / b, // Use regular division for unsigned integers and floats
        },
        .POW => std.math.pow(@TypeOf(a), a, b),
        .MIN => if (a < b) a else b,
        .MAX => if (a > b) a else b,
        .MOD => switch (@TypeOf(a)) {
            i8, i16, i32, i64 => @mod(a, b), // Use @mod for signed integers
            else => @rem(a, b), // Use regular modulo for unsigned integers and floats
        },
        .EQ => if (a == b) @as(i32, 1) else @as(i32, 0), // Equal to
        .GT => if (a > b) @as(i32, 1) else @as(i32, 0), // Greater than
        .LT => if (a < b) @as(i32, 1) else @as(i32, 0), // Less than
    };
}

fn compute_indices_and_apply_op(
    self: *const NdArray,
    other: *const NdArray,
    result: *NdArray,
    broadcasted_shape: []const usize,
    self_strides: []const usize,
    other_strides: []const usize,
    result_strides: []const usize,
    op: NdArrayBinaryOperation,
) void {
    const total_size = utils.compute_size(broadcasted_shape);

    for (0..total_size) |i| {
        var self_idx: usize = 0;
        var other_idx: usize = 0;
        var temp_i = i;

        for (0..broadcasted_shape.len) |axis| {
            const coord = temp_i / result_strides[axis];
            temp_i %= result_strides[axis];

            self_idx += if (axis >= self.shape.len or self.shape[axis] == 1) 0 else coord * self_strides[axis];
            other_idx += if (axis >= other.shape.len or other.shape[axis] == 1) 0 else coord * other_strides[axis];
        }

        const self_value = utils.bytes_to_val(self, self_idx * self.dtype.sizeInBytes());
        const other_value = utils.bytes_to_val(other, other_idx * other.dtype.sizeInBytes());

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => .{ .f32 = apply_binary_op(op, self_value.f32, other_value.f32) },
            .f64 => .{ .f64 = apply_binary_op(op, self_value.f64, other_value.f64) },
            .i8 => .{ .i8 = apply_binary_op(op, self_value.i8, other_value.i8) },
            .i16 => .{ .i16 = apply_binary_op(op, self_value.i16, other_value.i16) },
            .i32 => .{ .i32 = apply_binary_op(op, self_value.i32, other_value.i32) },
            .i64 => .{ .i64 = apply_binary_op(op, self_value.i64, other_value.i64) },
            .u8, .u16, .u32, .u64 => .{ .u32 = apply_binary_op(op, self_value.u32, other_value.u32) },
        };

        utils.val_to_bytes(result, i * result.dtype.sizeInBytes(), result_value);
    }
}

pub fn apply_elementwise(
    self: *const NdArray,
    other: *const NdArray,
    allocator: std.mem.Allocator,
    op: NdArrayBinaryOperation,
    in_place: bool,
) !*NdArray {
    // Check if shapes are compatible for broadcasting
    if (!is_compatible_for_broadcast(self.shape, other.shape)) {
        return error.ShapeMismatch;
    }

    // Compute the broadcasted shape
    const broadcasted_shape = try compute_broadcast_shape(self.shape, other.shape, allocator);
    defer allocator.free(broadcasted_shape);

    // Initialize the result tensor
    var result: *NdArray = undefined;
    if (in_place) {
        // If in-place, use `self` as the result tensor
        if (self.shape.len != broadcasted_shape.len or !std.mem.eql(usize, self.shape, broadcasted_shape)) {
            return error.ShapeMismatchForInplace;
        }
        result = @constCast(self); // Safe to cast since we're operating in-place
    } else {
        // Otherwise, create a new tensor with the broadcasted shape
        result = try allocator.create(NdArray);
        result.* = NdArray{
            .shape = try allocator.alloc(usize, broadcasted_shape.len),
            .dtype = self.dtype,
            .data = try allocator.alloc(u8, utils.compute_size(broadcasted_shape) * self.dtype.sizeInBytes()),
            .strides = try allocator.alloc(usize, broadcasted_shape.len), // Allocate strides
            .allocator = allocator,
            .owns_data = true,
        };
        @memcpy(result.shape, broadcasted_shape);

        // Compute strides for the result tensor
        compute_strides(broadcasted_shape, result.strides);

        errdefer {
            allocator.free(result.shape);
            allocator.free(result.data);
            allocator.free(result.strides); // Free strides on error
            allocator.destroy(result);
        }
    }

    // Perform the operation and store the result
    compute_indices_and_apply_op(
        self,
        other,
        result,
        broadcasted_shape,
        self.strides,
        other.strides,
        result.strides,
        op,
    );

    return result;
}
pub fn gemm(
    self: *const NdArray, // Represents matrix A
    b: *const NdArray, // Represents matrix B
    alpha: f32, // Scaling factor for A * B
    beta: f32, // Scaling factor for C
    transA: bool, // Whether to transpose A
    transB: bool, // Whether to transpose B
    allocator: std.mem.Allocator, // Memory allocator
) !*NdArray {
    // Check if the inputs are matrices (2D arrays)
    if (self.shape.len != 2 or b.shape.len != 2) {
        return error.InvalidShape;
    }

    // Get dimensions of A and B
    const a_rows = if (transA) self.shape[1] else self.shape[0];
    const a_cols = if (transA) self.shape[0] else self.shape[1];
    const b_rows = if (transB) b.shape[1] else b.shape[0];
    const b_cols = if (transB) b.shape[0] else b.shape[1];
    // std.debug.print("transA: {}\n", .{transA});
    // std.debug.print("transB: {}\n", .{transB});

    // std.debug.print("A: {}x{}\n", .{ a_rows, a_cols });
    // std.debug.print("B: {}x{}\n", .{ b_rows, b_cols });
    // Check if the matrices can be multiplied
    if (a_cols != b_rows) {
        return error.InvalidShape;
    }

    // Create the result matrix C with shape [a_rows, b_cols]
    const c_shape = &[_]usize{ a_rows, b_cols };
    const c = try NdArray.init(allocator, c_shape, self.dtype);
    errdefer c.deinit(allocator);

    // Perform the matrix multiplication
    for (0..a_rows) |i| {
        for (0..b_cols) |j| {
            var sum_val: NumericUnion = switch (self.dtype) {
                .f32 => .{ .f32 = 0.0 },
                .f64 => .{ .f64 = 0.0 },
                .i8 => .{ .i8 = 0 },
                .i16 => .{ .i16 = 0 },
                .i32 => .{ .i32 = 0 },
                .i64 => .{ .i64 = 0 },
                .u8 => .{ .u8 = 0 },
                .u16 => .{ .u16 = 0 },
                .u32 => .{ .u32 = 0 },
                .u64 => .{ .u64 = 0 },
            };

            for (0..a_cols) |k| {
                const a_idx = if (transA) k * a_rows + i else i * a_cols + k;
                const b_idx = if (transB) j * b_rows + k else k * b_cols + j;

                // std.debug.print("a_idx: {}, b_idx: {}\n", .{ a_idx, b_idx });

                const a_value = utils.bytes_to_val(self, a_idx * self.dtype.sizeInBytes());
                const b_value = utils.bytes_to_val(b, b_idx * b.dtype.sizeInBytes());

                // Perform multiplication and accumulate the result
                sum_val = switch (self.dtype) {
                    .f32 => .{ .f32 = sum_val.f32 + a_value.f32 * b_value.f32 },
                    .f64 => .{ .f64 = sum_val.f64 + a_value.f64 * b_value.f64 },
                    .i8 => .{ .i8 = sum_val.i8 + a_value.i8 * b_value.i8 },
                    .i16 => .{ .i16 = sum_val.i16 + a_value.i16 * b_value.i16 },
                    .i32 => .{ .i32 = sum_val.i32 + a_value.i32 * b_value.i32 },
                    .i64 => .{ .i64 = sum_val.i64 + a_value.i64 * b_value.i64 },
                    .u8 => .{ .u8 = sum_val.u8 + a_value.u8 * b_value.u8 },
                    .u16 => .{ .u16 = sum_val.u16 + a_value.u16 * b_value.u16 },
                    .u32 => .{ .u32 = sum_val.u32 + a_value.u32 * b_value.u32 },
                    .u64 => .{ .u64 = sum_val.u64 + a_value.u64 * b_value.u64 },
                };
            }

            // Apply alpha scaling to the sum_val
            sum_val = switch (self.dtype) {
                .f32 => .{ .f32 = sum_val.f32 * alpha },
                .f64 => .{ .f64 = sum_val.f64 * @as(f64, alpha) },
                .i8 => .{ .i8 = @as(i8, @intFromFloat(@as(f32, @floatFromInt(sum_val.i8)) * alpha)) },
                .i16 => .{ .i16 = @as(i16, @intFromFloat(@as(f32, @floatFromInt(sum_val.i16)) * alpha)) },
                .i32 => .{ .i32 = @as(i32, @intFromFloat(@as(f32, @floatFromInt(sum_val.i32)) * alpha)) },
                .i64 => .{ .i64 = @as(i64, @intFromFloat(@as(f64, @floatFromInt(sum_val.i64)) * @as(f64, alpha))) },
                .u8 => .{ .u8 = @as(u8, @intFromFloat(@as(f32, @floatFromInt(sum_val.u8)) * alpha)) },
                .u16 => .{ .u16 = @as(u16, @intFromFloat(@as(f32, @floatFromInt(sum_val.u16)) * alpha)) },
                .u32 => .{ .u32 = @as(u32, @intFromFloat(@as(f32, @floatFromInt(sum_val.u32)) * alpha)) },
                .u64 => .{ .u64 = @as(u64, @intFromFloat(@as(f64, @floatFromInt(sum_val.u64)) * @as(f64, alpha))) },
            };

            // If beta is non-zero, add beta * C_ij to the result
            if (beta != 0.0) {
                const c_idx = i * b_cols + j;
                const c_value = utils.bytes_to_val(c, c_idx * c.dtype.sizeInBytes());

                sum_val = switch (self.dtype) {
                    .f32 => .{ .f32 = sum_val.f32 + c_value.f32 * beta },
                    .f64 => .{ .f64 = sum_val.f64 + c_value.f64 * @as(f64, beta) },
                    .i8 => .{ .i8 = sum_val.i8 + @as(i8, @intFromFloat(@as(f32, @floatFromInt(c_value.i8)) * beta)) },
                    .i16 => .{ .i16 = sum_val.i16 + @as(i16, @intFromFloat(@as(f32, @floatFromInt(c_value.i16)) * beta)) },
                    .i32 => .{ .i32 = sum_val.i32 + @as(i32, @intFromFloat(@as(f32, @floatFromInt(c_value.i32)) * beta)) },
                    .i64 => .{ .i64 = sum_val.i64 + @as(i64, @intFromFloat(@as(f64, @floatFromInt(c_value.i64)) * @as(f64, beta))) },
                    .u8 => .{ .u8 = sum_val.u8 + @as(u8, @intFromFloat(@as(f32, @floatFromInt(c_value.u8)) * beta)) },
                    .u16 => .{ .u16 = sum_val.u16 + @as(u16, @intFromFloat(@as(f32, @floatFromInt(c_value.u16)) * beta)) },
                    .u32 => .{ .u32 = sum_val.u32 + @as(u32, @intFromFloat(@as(f32, @floatFromInt(c_value.u32)) * beta)) },
                    .u64 => .{ .u64 = sum_val.u64 + @as(u64, @intFromFloat(@as(f64, @floatFromInt(c_value.u64)) * @as(f64, beta))) },
                };
            }

            // Store the result in the output matrix
            const c_idx = i * b_cols + j;
            utils.val_to_bytes(c, c_idx * c.dtype.sizeInBytes(), sum_val);
        }
    }

    return c;
}

///=============================================================================
/// Binary operations with scalars
///=============================================================================
pub const NdArrayScalarBinaryOperation = enum {
    // binary operations
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    MIN,
    MAX,
    MOD,
    GT, // Greater than
    LT, // Less than
    EQ, // Equal to
};

pub fn apply_binary(self: *NdArray, scalar: anytype, op: NdArrayScalarBinaryOperation, in_place: bool) !*NdArray {
    const total_size = utils.compute_size(self.shape);
    const element_size = self.dtype.sizeInBytes();

    var result: *NdArray = undefined;
    if (in_place) {
        result = self; // Use the existing array for in-place operations
    } else {
        // Allocate a new array for non-in-place operations
        result = try NdArray.init(self.allocator, self.shape, self.dtype);
        errdefer result.deinit(); // Correct: Call `deinit` without arguments
    }

    for (0..total_size) |i| {
        const byte_offset = i * element_size;
        const value = utils.bytes_to_val(self, byte_offset);

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => .{ .f32 = apply_binary_op_sc(op, value.f32, @as(f32, scalar)) },
            .f64 => .{ .f64 = apply_binary_op_sc(op, value.f64, @as(f64, scalar)) },
            .i8 => .{ .i8 = apply_binary_op_sc(op, value.i8, @as(i8, @intFromFloat(scalar))) },
            .i16 => .{ .i16 = apply_binary_op_sc(op, value.i16, @as(i16, @intFromFloat(scalar))) },
            .i32 => .{ .i32 = apply_binary_op_sc(op, value.i32, @as(i32, @intFromFloat(scalar))) },
            .i64 => .{ .i64 = apply_binary_op_sc(op, value.i64, @as(i64, @intFromFloat(scalar))) },
            .u8 => .{ .u8 = apply_binary_op_sc(op, value.u8, @as(u8, @intFromFloat(scalar))) },
            .u16 => .{ .u16 = apply_binary_op_sc(op, value.u16, @as(u16, @intFromFloat(scalar))) },
            .u32 => .{ .u32 = apply_binary_op_sc(op, value.u32, @as(u32, @intFromFloat(scalar))) },
            .u64 => .{ .u64 = apply_binary_op_sc(op, value.u64, @as(u64, @intFromFloat(scalar))) },
        };

        utils.val_to_bytes(result, byte_offset, result_value);
    }

    return result;
}

fn apply_binary_op_sc(op: NdArrayScalarBinaryOperation, a: anytype, b: anytype) @TypeOf(a) {
    return switch (op) {
        .ADD => a + b,
        .SUB => a - b,
        .MUL => a * b,
        .DIV => switch (@TypeOf(a)) {
            i8, i16, i32, i64 => @divTrunc(a, b),
            else => a / b,
        },
        .POW => std.math.pow(@TypeOf(a), a, b),
        .MIN => if (a < b) a else b,
        .MAX => if (a > b) a else b,
        .MOD => switch (@TypeOf(a)) {
            i8, i16, i32, i64 => @mod(a, b),
            else => @rem(a, b),
        },
        .GT => if (a > b) @as(i32, 1) else @as(i32, 0), // Greater than
        .LT => if (a < b) @as(i32, 1) else @as(i32, 0), // Less than
        .EQ => if (a == b) @as(i32, 1) else @as(i32, 0), // Equal to
    };
}

///=============================================================================
/// Reduction operations with ndarray
///=============================================================================
pub const NdArrayReductionOperation = enum {
    MIN,
    MAX,
    SUM,
    MEAN,
    PROD,
    ARGMIN,
    ARGMAX,
};

pub fn reduce(
    self: *NdArray,
    op: NdArrayReductionOperation,
    axis: ?usize, // Optional axis to reduce along
    keepdims: bool, // Whether to keep the reduced dimensions
) !*NdArray {
    const total_size = utils.compute_size(self.shape);

    // If no axis is provided, reduce the entire array to a scalar
    if (axis == null) {
        const result_value = try reduce_to_scalar(self, op, total_size);
        const result_shape = if (keepdims) blk: {
            const shape = try self.allocator.alloc(usize, 1);
            shape[0] = 1;
            break :blk shape;
        } else &[_]usize{1};

        defer if (keepdims) self.allocator.free(result_shape);

        // Initialize the result tensor with the correct data type
        const result = switch (op) {
            .ARGMIN, .ARGMAX => try NdArray.init(self.allocator, result_shape, .u64),
            else => try NdArray.init(self.allocator, result_shape, self.dtype),
        };
        errdefer result.deinit();

        // Set the result value
        try result.set(&[_]usize{0}, result_value);
        return result;
    }

    // If an axis is provided, reduce along that axis
    const new_shape = try compute_reduced_shape(self.shape, axis.?, keepdims, self.allocator);
    defer self.allocator.free(new_shape);

    const result = switch (op) {
        .ARGMIN, .ARGMAX => blk: {
            const res = try NdArray.init(self.allocator, new_shape, .u64);
            errdefer res.deinit();
            break :blk res;
        },
        else => blk: {
            const res = try NdArray.init(self.allocator, new_shape, self.dtype);
            errdefer res.deinit();
            break :blk res;
        },
    };

    try reduce_along_axis(self, result, op, axis.?, keepdims, self.allocator);

    return result;
}
fn reduce_to_scalar(self: *NdArray, op: NdArrayReductionOperation, total_size: usize) !NumericUnion {
    var result_value: NumericUnion = undefined;
    switch (self.dtype) {
        .f32 => {
            var acc: f32 = switch (op) {
                .MIN => std.math.inf(f32),
                .MAX => -std.math.inf(f32),
                .SUM => 0,
                .MEAN => 0,
                .PROD => 1,
                .ARGMIN, .ARGMAX => 0, // Placeholder, will be replaced
            };
            var arg_index: usize = 0; // For ARGMIN and ARGMAX
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).f32;
                switch (op) {
                    .MIN => acc = @min(acc, value),
                    .MAX => acc = @max(acc, value),
                    .SUM => acc += value,
                    .MEAN => acc += value,
                    .PROD => acc *= value,
                    .ARGMIN => {
                        if (value < acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                    .ARGMAX => {
                        if (value > acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                }
            }
            if (op == .MEAN) {
                acc /= @as(f32, @floatFromInt(total_size));
            }
            result_value = switch (op) {
                .ARGMIN, .ARGMAX => .{ .u64 = @as(u64, @intCast(arg_index)) }, // Return the index for ARGMIN/ARGMAX
                else => .{ .f32 = acc },
            };
        },
        .f64 => {
            var acc: f64 = switch (op) {
                .MIN => std.math.inf(f64),
                .MAX => -std.math.inf(f64),
                .SUM => 0,
                .MEAN => 0,
                .PROD => 1,
                .ARGMIN, .ARGMAX => 0, // Placeholder, will be replaced
            };
            var arg_index: usize = 0; // For ARGMIN and ARGMAX
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).f64;
                switch (op) {
                    .MIN => acc = @min(acc, value),
                    .MAX => acc = @max(acc, value),
                    .SUM => acc += value,
                    .MEAN => acc += value,
                    .PROD => acc *= value,
                    .ARGMIN => {
                        if (value < acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                    .ARGMAX => {
                        if (value > acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                }
            }
            if (op == .MEAN) {
                acc /= @as(f64, @floatFromInt(total_size));
            }
            result_value = switch (op) {
                .ARGMIN, .ARGMAX => .{ .u64 = @as(u64, @intCast(arg_index)) }, // Return the index for ARGMIN/ARGMAX
                else => .{ .f64 = acc },
            };
        },
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |_| {
            var acc: i64 = switch (op) {
                .MIN => std.math.maxInt(i64),
                .MAX => std.math.minInt(i64),
                .SUM => 0,
                .MEAN => 0,
                .PROD => 1,
                .ARGMIN, .ARGMAX => 0, // Placeholder, will be replaced
            };
            var arg_index: usize = 0; // For ARGMIN and ARGMAX
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).i64;
                switch (op) {
                    .MIN => acc = @min(acc, value),
                    .MAX => acc = @max(acc, value),
                    .SUM => acc += value,
                    .MEAN => acc += value,
                    .PROD => acc *= value,
                    .ARGMIN => {
                        if (value < acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                    .ARGMAX => {
                        if (value > acc) {
                            acc = value;
                            arg_index = i;
                        }
                    },
                }
            }
            if (op == .MEAN) {
                acc = @divTrunc(acc, @as(i64, @intCast(total_size)));
            }
            result_value = switch (op) {
                .ARGMIN, .ARGMAX => .{ .u64 = @as(u64, @intCast(arg_index)) }, // Return the index for ARGMIN/ARGMAX
                else => .{ .i64 = acc },
            };
        },
    }
    return result_value;
}
fn compute_reduced_shape(shape: []const usize, axis: usize, keepdims: bool, allocator: std.mem.Allocator) ![]usize {
    var new_shape = std.ArrayList(usize).init(allocator);
    defer new_shape.deinit();
    for (shape, 0..) |dim, i| {
        if (i == axis) {
            if (keepdims) {
                try new_shape.append(1); // Keep the reduced dimension as size 1
            }
        } else {
            try new_shape.append(dim);
        }
    }
    return try new_shape.toOwnedSlice();
}

fn reduce_along_axis(self: *NdArray, result: *NdArray, op: NdArrayReductionOperation, axis: usize, keepdims: bool, allocator: std.mem.Allocator) !void {
    const outer_size = self.shape[if (axis == 0) 1 else 0];
    const inner_size = self.shape[axis];

    for (0..outer_size) |outer_idx| {
        var acc: f32 = switch (op) {
            .MIN => std.math.inf(f32),
            .MAX => -std.math.inf(f32),
            .SUM => 0,
            .MEAN => 0,
            .PROD => 1,
            .ARGMIN, .ARGMAX => 0, // Placeholder, will be replaced
        };
        var arg_index: usize = 0; // For ARGMIN and ARGMAX

        for (0..inner_size) |inner_idx| {
            const indices = if (axis == 0) &[_]usize{ inner_idx, outer_idx } else &[_]usize{ outer_idx, inner_idx };
            const value = (try self.get(indices)).f32;

            switch (op) {
                .MIN => acc = @min(acc, value),
                .MAX => acc = @max(acc, value),
                .SUM => acc += value,
                .MEAN => acc += value,
                .PROD => acc *= value,
                .ARGMIN => {
                    if (value < acc) {
                        acc = value;
                        arg_index = inner_idx;
                    }
                },
                .ARGMAX => {
                    if (value > acc) {
                        acc = value;
                        arg_index = inner_idx;
                    }
                },
            }
        }

        if (op == .MEAN) {
            acc /= @as(f32, @floatFromInt(inner_size));
        }

        // Calculate the correct result indices based on the axis and keepdims
        const result_indices = if (keepdims) blk: {
            const indices = try allocator.alloc(usize, result.shape.len);
            for (indices, 0..) |*idx, i| {
                if (i == axis) {
                    idx.* = 0; // Reduced axis is set to 0
                } else if (i < axis) {
                    idx.* = outer_idx; // Copy outer_idx for dimensions before the axis
                } else {
                    idx.* = outer_idx; // Copy outer_idx for dimensions after the axis
                }
            }
            break :blk indices;
        } else &[_]usize{outer_idx};

        defer if (keepdims) allocator.free(result_indices);

        const result_value: NumericUnion = switch (op) {
            .ARGMIN, .ARGMAX => .{ .u64 = @as(u64, @intCast(arg_index)) }, // Return the index for ARGMIN/ARGMAX
            else => .{ .f32 = acc }, // Return the computed value for other operations
        };
        try result.set(result_indices, result_value);
    }
}

// This is for column major order
fn _compute_indices_from_flat_index(flat_idx: usize, shape: []const usize, allocator: std.mem.Allocator) ![]usize {
    const indices = try allocator.alloc(usize, shape.len);
    var remaining = flat_idx;
    for (shape, 0..) |dim, i| {
        indices[i] = remaining % dim;
        remaining /= dim;
    }
    return indices;
}

// This is for row major order
fn compute_indices_from_flat_index(flat_idx: usize, shape: []const usize, allocator: std.mem.Allocator) ![]usize {
    const indices = try allocator.alloc(usize, shape.len);
    var remaining = flat_idx;
    for (0..shape.len) |i| {
        const dim = shape[shape.len - 1 - i]; // Iterate from the last dimension to the first
        indices[shape.len - 1 - i] = remaining % dim;
        remaining /= dim;
    }
    return indices;
}

///=============================================================================
/// Unary operations with ndarray
/// This file contains unary operations with ndarray
/// Unary operations are operations that involve only one operand.
///
/// Unary operations with ndarray
/// This file contains unary operations with ndarray
/// Unary operations are operations that involve only one operand.
///=============================================================================
pub const NdArrayUnaryOperation = enum {
    NEG,
    EXP,
    LOG,
    LOG10,
    SQRT,
    SIN,
    COS,
    ABS,
    FLOOR,
    CEIL,
    ROUND,
    TANH,
    SIGN,
    RELU,
    SOFTMAX,
    SWISH,
    SIGMOID,
};

pub fn apply_unary(self: *NdArray, op: NdArrayUnaryOperation, in_place: bool) !*NdArray {
    const total_size = utils.compute_size(self.shape);
    const element_size = self.dtype.sizeInBytes();

    var result: *NdArray = undefined;
    if (in_place) {
        result = self; // Use the existing array for in-place operations
    } else {
        // Allocate a new array for non-in-place operations
        result = try NdArray.init(self.allocator, self.shape, self.dtype);
        errdefer result.deinit();
    }

    for (0..total_size) |i| {
        const byte_offset = i * element_size;
        const value = utils.bytes_to_val(self, byte_offset);

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => switch (op) {
                .NEG => .{ .f32 = -value.f32 },
                .EXP => .{ .f32 = @exp(value.f32) },
                .LOG => if (value.f32 <= 0) return error.InvalidInput else .{ .f32 = @log(value.f32) },
                .LOG10 => if (value.f32 <= 0) return error.InvalidInput else .{ .f32 = @log10(value.f32) },
                .SQRT => if (value.f32 < 0) return error.InvalidInput else .{ .f32 = @sqrt(value.f32) },
                .SIN => .{ .f32 = @sin(value.f32) },
                .COS => .{ .f32 = @cos(value.f32) },
                .TANH => .{ .f32 = std.math.tanh(value.f32) },
                .ABS => .{ .f32 = @abs(value.f32) },
                .FLOOR => .{ .f32 = @floor(value.f32) },
                .CEIL => .{ .f32 = @ceil(value.f32) },
                .ROUND => .{ .f32 = @round(value.f32) },
                .SIGN => .{ .f32 = if (value.f32 > 0) @as(f32, 1) else if (value.f32 < 0) @as(f32, -1) else @as(f32, 0) },
                .RELU => .{ .f32 = if (value.f32 > 0) value.f32 else 0 },
                .SWISH => .{ .f32 = value.f32 * (1 / (1 + @exp(-value.f32))) },
                .SOFTMAX => .{ .f32 = @exp(value.f32) }, // Softmax is just exp(x) for normalized input
                .SIGMOID => blk: {
                    const x = value.f32;
                    const exp_neg_x = @exp(-x);
                    const sigmoid_x = 1.0 / (1.0 + exp_neg_x);
                    break :blk .{ .f32 = sigmoid_x };
                },
            },
            .f64 => switch (op) {
                .NEG => .{ .f64 = -value.f64 },
                .EXP => .{ .f64 = @exp(value.f64) },
                .LOG => if (value.f64 <= 0) return error.InvalidInput else .{ .f64 = @log(value.f64) },
                .LOG10 => if (value.f64 <= 0) return error.InvalidInput else .{ .f64 = @log10(value.f64) },
                .SQRT => if (value.f64 < 0) return error.InvalidInput else .{ .f64 = @sqrt(value.f64) },
                .SIN => .{ .f64 = @sin(value.f64) },
                .COS => .{ .f64 = @cos(value.f64) },
                .TANH => .{ .f64 = std.math.tanh(value.f64) },
                .ABS => .{ .f64 = @abs(value.f64) },
                .FLOOR => .{ .f64 = @floor(value.f64) },
                .CEIL => .{ .f64 = @ceil(value.f64) },
                .ROUND => .{ .f64 = @round(value.f64) },
                .SIGN => .{ .f64 = if (value.f64 > 0) @as(f64, 1) else if (value.f64 < 0) @as(f64, -1) else @as(f64, 0) },
                .RELU => .{ .f64 = if (value.f64 > 0) value.f64 else 0 },
                .SWISH => .{ .f64 = value.f64 * (1 / (1 + @exp(-value.f64))) },
                .SOFTMAX => .{ .f64 = @exp(value.f64) }, // Softmax is just exp(x) for normalized input
                .SIGMOID => blk: {
                    const x = value.f64;
                    const exp_neg_x = @exp(-x);
                    const sigmoid_x = 1.0 / (1.0 + exp_neg_x);
                    break :blk .{ .f64 = sigmoid_x };
                },
            },
            inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |T| switch (op) {
                .NEG => blk: {
                    const min_val = switch (T) {
                        .i8 => std.math.minInt(i8),
                        .i16 => std.math.minInt(i16),
                        .i32 => std.math.minInt(i32),
                        .i64 => std.math.minInt(i64),
                        else => unreachable, // Unsigned types cannot be negated
                    };
                    break :blk .{ .i32 = if (value.i32 == min_val) return error.Overflow else -value.i32 };
                },
                .ABS => blk: {
                    const min_val = switch (T) {
                        .i8 => std.math.minInt(i8),
                        .i16 => std.math.minInt(i16),
                        .i32 => std.math.minInt(i32),
                        .i64 => std.math.minInt(i64),
                        else => unreachable, // Unsigned types cannot be negated
                    };
                    break :blk .{ .i32 = if (value.i32 == min_val) return error.Overflow else if (value.i32 < 0) -value.i32 else value.i32 };
                },
                .SIGN => .{ .i32 = if (value.i32 > 0) 1 else if (value.i32 < 0) -1 else 0 },
                else => blk: {
                    const float_val = @as(f64, @floatFromInt(value.i32));
                    const computed_result = switch (op) {
                        .EXP => @exp(float_val),
                        .LOG => if (float_val <= 0) return error.InvalidInput else @log(float_val),
                        .LOG10 => if (float_val <= 0) return error.InvalidInput else @log10(float_val),
                        .SQRT => if (float_val < 0) return error.InvalidInput else @sqrt(float_val),
                        .SIN => @sin(float_val),
                        .COS => @cos(float_val),
                        .TANH => std.math.tanh(float_val),
                        .FLOOR => @floor(float_val),
                        .CEIL => @ceil(float_val),
                        .ROUND => @round(float_val),
                        .RELU => if (float_val > 0) float_val else 0,
                        .SWISH => float_val * (1 / (1 + @exp(-float_val))),
                        .SOFTMAX => @exp(float_val), // Softmax is just exp(x) for normalized input
                        .SIGMOID => 1 / (1 + @exp(-float_val)),
                        else => unreachable,
                    };
                    break :blk .{ .i32 = @as(i32, @intFromFloat(computed_result)) };
                },
            },
        };

        utils.val_to_bytes(result, byte_offset, result_value);
    }

    return result;
}

pub fn cast_elements(self: *NdArray, target_dtype: DataType) !*NdArray {
    const total_size = utils.compute_size(self.shape);
    const element_size = target_dtype.sizeInBytes();

    // Allocate a new array for the result
    const result = try NdArray.init(self.allocator, self.shape, target_dtype);
    errdefer result.deinit();

    for (0..total_size) |i| {
        const byte_offset = i * self.dtype.sizeInBytes();
        const value = utils.bytes_to_val(self, byte_offset);

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => switch (target_dtype) {
                .f32 => .{ .f32 = value.f32 }, // No-op for f32
                .f64 => .{ .f64 = @as(f64, value.f32) },
                .i32 => .{ .i32 = @as(i32, @intFromFloat(value.f32)) },
                .i64 => .{ .i64 = @as(i64, @intFromFloat(value.f32)) },
                else => return error.UnsupportedDataType,
            },
            .f64 => switch (target_dtype) {
                .f32 => .{ .f32 = @as(f32, @floatCast(value.f64)) },
                .f64 => .{ .f64 = value.f64 }, // No-op for f64
                .i32 => .{ .i32 = @as(i32, @intFromFloat(value.f64)) },
                .i64 => .{ .i64 = @as(i64, @intFromFloat(value.f64)) },
                else => return error.UnsupportedDataType,
            },
            inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |_| switch (target_dtype) {
                .f32 => .{ .f32 = @as(f32, @floatFromInt(value.i32)) },
                .f64 => .{ .f64 = @as(f64, @floatFromInt(value.i32)) },
                .i32 => .{ .i32 = value.i32 }, // No-op for i32
                .i64 => .{ .i64 = @as(i64, value.i32) },
                else => return error.UnsupportedDataType,
            },
        };

        const result_byte_offset = i * element_size;
        utils.val_to_bytes(result, result_byte_offset, result_value);
    }

    return result;
}
