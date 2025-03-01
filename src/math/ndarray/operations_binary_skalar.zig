const std = @import("std");
const utils = @import("utils.zig");
const converters = @import("converters.zig");

const NdArray = @import("ndarray.zig").NdArray;
pub const NumericUnion = @import("dtypes.zig").NumericUnion;

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
        const value = converters.bytes_to_val(self, byte_offset);

        const result_value: NumericUnion = switch (self.dtype) {
            .f32 => .{ .f32 = apply_binary_op(op, value.f32, @as(f32, scalar)) },
            .f64 => .{ .f64 = apply_binary_op(op, value.f64, @as(f64, scalar)) },
            .i8 => .{ .i8 = apply_binary_op(op, value.i8, @as(i8, @intFromFloat(scalar))) },
            .i16 => .{ .i16 = apply_binary_op(op, value.i16, @as(i16, @intFromFloat(scalar))) },
            .i32 => .{ .i32 = apply_binary_op(op, value.i32, @as(i32, @intFromFloat(scalar))) },
            .i64 => .{ .i64 = apply_binary_op(op, value.i64, @as(i64, @intFromFloat(scalar))) },
            .u8 => .{ .u8 = apply_binary_op(op, value.u8, @as(u8, @intFromFloat(scalar))) },
            .u16 => .{ .u16 = apply_binary_op(op, value.u16, @as(u16, @intFromFloat(scalar))) },
            .u32 => .{ .u32 = apply_binary_op(op, value.u32, @as(u32, @intFromFloat(scalar))) },
            .u64 => .{ .u64 = apply_binary_op(op, value.u64, @as(u64, @intFromFloat(scalar))) },
        };

        converters.val_to_bytes(result, byte_offset, result_value);
    }

    return result;
}

fn apply_binary_op(op: NdArrayScalarBinaryOperation, a: anytype, b: anytype) @TypeOf(a) {
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
