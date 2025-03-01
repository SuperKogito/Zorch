const std = @import("std");
const utils = @import("utils.zig");
const converters = @import("converters.zig");

const NdArray = @import("ndarray.zig").NdArray;
pub const NumericUnion = @import("dtypes.zig").NumericUnion;

pub const NdArrayReductionOperation = enum {
    MIN,
    MAX,
    SUM,
    MEAN,
    PROD,
};

pub fn reduce(
    self: *NdArray,
    op: NdArrayReductionOperation,
    axis: ?usize, // Optional axis to reduce along
) !*NdArray {
    const total_size = utils.compute_size(self.shape);

    // If no axis is provided, reduce the entire array to a scalar
    if (axis == null) {
        const result_value = try reduce_to_scalar(self, op, total_size);
        const result = try NdArray.init(self.allocator, &[_]usize{1}, self.dtype);
        try result.set(&[_]usize{0}, result_value);
        return result;
    }

    // If an axis is provided, reduce along that axis
    const new_shape = try compute_reduced_shape(self.shape, axis.?, self.allocator);
    defer self.allocator.free(new_shape);

    const result = try NdArray.init(self.allocator, new_shape, self.dtype);
    errdefer result.deinit();

    try reduce_along_axis(self, result, op, axis.?, self.allocator);

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
            };
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).f32;
                acc = switch (op) {
                    .MIN => @min(acc, value),
                    .MAX => @max(acc, value),
                    .SUM => acc + value,
                    .MEAN => acc + value,
                    .PROD => acc * value,
                };
            }
            if (op == .MEAN) {
                acc /= @as(f32, @floatFromInt(total_size));
            }
            result_value = .{ .f32 = acc };
        },
        .f64 => {
            var acc: f64 = switch (op) {
                .MIN => std.math.inf(f64),
                .MAX => -std.math.inf(f64),
                .SUM => 0,
                .MEAN => 0,
                .PROD => 1,
            };
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).f64;
                acc = switch (op) {
                    .MIN => @min(acc, value),
                    .MAX => @max(acc, value),
                    .SUM => acc + value,
                    .MEAN => acc + value,
                    .PROD => acc * value,
                };
            }
            if (op == .MEAN) {
                acc /= @as(f64, @floatFromInt(total_size));
            }
            result_value = .{ .f64 = acc };
        },
        inline .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64 => |_| {
            var acc: i64 = switch (op) {
                .MIN => std.math.maxInt(i64),
                .MAX => std.math.minInt(i64),
                .SUM => 0,
                .MEAN => 0,
                .PROD => 1,
            };
            for (0..total_size) |i| {
                const indices = try compute_indices_from_flat_index(i, self.shape, self.allocator);
                defer self.allocator.free(indices);
                const value = (try self.get(indices)).i64;
                acc = switch (op) {
                    .MIN => @min(acc, value),
                    .MAX => @max(acc, value),
                    .SUM => acc + value,
                    .MEAN => acc + value,
                    .PROD => acc * value,
                };
            }
            if (op == .MEAN) {
                acc = @divTrunc(acc, @as(i64, @intCast(total_size)));
            }
            result_value = .{ .i64 = acc };
        },
    }
    return result_value;
}

fn compute_reduced_shape(shape: []const usize, axis: usize, allocator: std.mem.Allocator) ![]usize {
    var new_shape = std.ArrayList(usize).init(allocator);
    defer new_shape.deinit();
    for (shape, 0..) |dim, i| {
        if (i != axis) {
            try new_shape.append(dim);
        }
    }
    return try new_shape.toOwnedSlice();
}

fn reduce_along_axis(self: *NdArray, result: *NdArray, op: NdArrayReductionOperation, axis: usize, allocator: std.mem.Allocator) !void {
    _ = allocator;
    const outer_size = self.shape[if (axis == 0) 1 else 0];
    const inner_size = self.shape[axis];

    for (0..outer_size) |outer_idx| {
        var acc: f32 = switch (op) {
            .MIN => std.math.inf(f32),
            .MAX => -std.math.inf(f32),
            .SUM => 0,
            .MEAN => 0,
            .PROD => 1,
        };

        for (0..inner_size) |inner_idx| {
            const indices = if (axis == 0) &[_]usize{ inner_idx, outer_idx } else &[_]usize{ outer_idx, inner_idx };
            const value = (try self.get(indices)).f32;

            acc = switch (op) {
                .MIN => @min(acc, value),
                .MAX => @max(acc, value),
                .SUM => acc + value,
                .MEAN => acc + value,
                .PROD => acc * value,
            };
        }

        if (op == .MEAN) {
            acc /= @as(f32, @floatFromInt(inner_size));
        }

        try result.set(&[_]usize{outer_idx}, .{ .f32 = acc });
    }
}

fn compute_indices_from_flat_index(flat_idx: usize, shape: []const usize, allocator: std.mem.Allocator) ![]usize {
    const indices = try allocator.alloc(usize, shape.len);
    var remaining = flat_idx;
    for (shape, 0..) |dim, i| {
        indices[i] = remaining % dim;
        remaining /= dim;
    }
    return indices;
}
