const std = @import("std");
const utils = @import("zig");
const dtypes = @import("dtypes.zig");

const DType = dtypes.DType;
const NdArray = @import("ndarray.zig").NdArray;

pub fn alloc_indices(rank: usize, allocator: std.mem.Allocator) ![]usize {
    const indices = try allocator.alloc(usize, rank);
    for (0..rank) |i| {
        indices[i] = 0;
    }
    return indices;
}

pub fn print_ndarray(ndarray: *const NdArray, allocator: std.mem.Allocator) !void {
    const writer = std.io.getStdOut().writer();
    const indices = try alloc_indices(ndarray.shape.len, allocator);
    defer allocator.free(indices);
    try print_ndarray_recursive(
        ndarray,
        writer,
        0,
        0,
        allocator,
        indices,
    );
    // Print a new line after the tensor output
    std.debug.print("\n", .{});
}

pub fn print_ndarray_recursive(
    ndarray: *const NdArray,
    writer: anytype,
    dim_level: usize,
    indent: usize,
    allocator: std.mem.Allocator,
    indices: []usize, // Pass the indices array as a parameter
) !void {
    const rank = ndarray.shape.len;
    const current_dim = ndarray.shape[dim_level];

    var spaces: [256]u8 = undefined;
    for (0..indent) |i| spaces[i] = ' ';

    if (dim_level == rank - 1) {
        try writer.print("{s}[", .{spaces[0..indent]});
        for (0..current_dim) |i| {
            indices[dim_level] = i; // Update the current dimension index
            const val = try ndarray.get(indices);
            switch (ndarray.dtype) {
                .f32 => try writer.print("{d:.3}", .{val.f32}),
                .f64 => try writer.print("{d:.6}", .{val.f64}),
                .i8 => try writer.print("{d}", .{val.i8}),
                .i16 => try writer.print("{d}", .{val.i16}),
                .i32 => try writer.print("{d}", .{val.i32}),
                .i64 => try writer.print("{d}", .{val.i64}),
                .u8 => try writer.print("{d}", .{val.u8}),
                .u16 => try writer.print("{d}", .{val.u16}),
                .u32 => try writer.print("{d}", .{val.u32}),
                .u64 => try writer.print("{d}", .{val.u64}),
            }
            if (i < current_dim - 1) try writer.writeAll(", ");
        }
        try writer.writeAll("]");
        return;
    }

    try writer.print("{s}[\n", .{spaces[0..indent]});
    for (0..current_dim) |i| {
        indices[dim_level] = i; // Update the current dimension index
        try print_ndarray_recursive(
            ndarray,
            writer,
            dim_level + 1,
            indent + 2,
            allocator,
            indices, // Pass the updated indices array
        );
        if (i < current_dim - 1) try writer.writeAll(",\n");
    }
    try writer.print("\n{s}]", .{spaces[0..indent]});
}

/// Prints information about an NdArray.
pub fn print_ndarray_info(self: *NdArray, allocator: std.mem.Allocator) !void {
    // Print shape
    const shape_str = try std.fmt.allocPrint(allocator, "{any}", .{self.shape});
    defer allocator.free(shape_str);

    // Compute first and last indices
    const first_index = try allocator.alloc(usize, self.shape.len);
    defer allocator.free(first_index);
    @memset(first_index, 0); // First index is all zeros

    const last_index = try allocator.alloc(usize, self.shape.len);
    defer allocator.free(last_index);

    var remaining = compute_size(self.shape) - 1;
    for (self.shape, 0..) |dim_size, i| {
        last_index[i] = remaining % dim_size;
        remaining /= dim_size;
    }

    // Get first and last values
    const first_val_union = try self.get(first_index);
    const last_val_union = try self.get(last_index);

    // Format numeric values based on dtype
    const first_val_str = try dtypes.format_numeric_value(first_val_union, self.dtype, allocator);
    defer allocator.free(first_val_str);

    const last_val_str = try dtypes.format_numeric_value(last_val_union, self.dtype, allocator);
    defer allocator.free(last_val_str);

    // Print array information with tabs before keywords
    std.debug.print("[{s} ... {s}], shape: {s}, rank: {d}, size: {d}, dtype: {s}", .{
        first_val_str,
        last_val_str,
        shape_str,
        self.shape.len,
        compute_size(self.shape),
        @tagName(self.dtype),
    });
}

pub fn compute_size(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| size *= dim;
    return size;
}
