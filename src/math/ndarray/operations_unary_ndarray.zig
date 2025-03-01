const std = @import("std");
const utils = @import("utils.zig");
const dtypes = @import("dtypes.zig");
const converters = @import("converters.zig");

pub const DataType = dtypes.DataType;
const NdArray = @import("ndarray.zig").NdArray;
pub const NumericUnion = @import("dtypes.zig").NumericUnion;

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
        const value = converters.bytes_to_val(self, byte_offset);

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

        converters.val_to_bytes(result, byte_offset, result_value);
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
        const value = converters.bytes_to_val(self, byte_offset);

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
        converters.val_to_bytes(result, result_byte_offset, result_value);
    }

    return result;
}
