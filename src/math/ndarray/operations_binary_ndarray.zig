const std = @import("std");
const utils = @import("utils.zig");
const converters = @import("converters.zig");

const NdArray = @import("ndarray.zig").NdArray;
pub const NumericUnion = @import("dtypes.zig").NumericUnion;

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

        const self_value = converters.bytes_to_val(self, self_idx * self.dtype.sizeInBytes());
        const other_value = converters.bytes_to_val(other, other_idx * other.dtype.sizeInBytes());

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => .{ .f32 = apply_binary_op(op, self_value.f32, other_value.f32) },
            .f64 => .{ .f64 = apply_binary_op(op, self_value.f64, other_value.f64) },
            .i8 => .{ .i8 = apply_binary_op(op, self_value.i8, other_value.i8) },
            .i16 => .{ .i16 = apply_binary_op(op, self_value.i16, other_value.i16) },
            .i32 => .{ .i32 = apply_binary_op(op, self_value.i32, other_value.i32) },
            .i64 => .{ .i64 = apply_binary_op(op, self_value.i64, other_value.i64) },
            .u8, .u16, .u32, .u64 => .{ .u32 = apply_binary_op(op, self_value.u32, other_value.u32) },
        };

        converters.val_to_bytes(result, i * result.dtype.sizeInBytes(), result_value);
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

                const a_value = converters.bytes_to_val(self, a_idx * self.dtype.sizeInBytes());
                const b_value = converters.bytes_to_val(b, b_idx * b.dtype.sizeInBytes());

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
                const c_value = converters.bytes_to_val(c, c_idx * c.dtype.sizeInBytes());

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
            converters.val_to_bytes(c, c_idx * c.dtype.sizeInBytes(), sum_val);
        }
    }

    return c;
}
