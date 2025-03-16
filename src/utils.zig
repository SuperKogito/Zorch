const std = @import("std");
const zorch = @import("zorch.zig");

const dtypes = zorch.dtypes;
const logger = zorch.logger;

const NdArray = zorch.NdArray;
const NdarrayError = zorch.errors.NdarrayError;

/// Converts a sequence of bytes from the NdArray's data buffer into a `NumericUnion` value.
///
/// This function interprets the raw bytes in the NdArray's data buffer starting at the specified
/// `byte_offset` and converts them into a `NumericUnion` value based on the NdArray's `dtype`.
/// The `NumericUnion` is a tagged union that can hold any supported numeric type (e.g., `f32`, `i32`, `u64`, etc.).
///
/// # Parameters
/// - `self`: A pointer to the `NdArray` instance whose data buffer is being accessed.
/// - `byte_offset`: The starting index in the data buffer (in bytes) from which to read the value.
///
/// # Returns
/// A `NumericUnion` containing the interpreted value from the data buffer.
///
/// # Behavior
/// - The function reads a sequence of bytes from `self.data` starting at `byte_offset`.
/// - The number of bytes read depends on the `dtype` of the NdArray:
///   - `f32`, `i32`, `u32`: 4 bytes
///   - `f64`, `i64`, `u64`: 8 bytes
///   - `i16`, `u16`: 2 bytes
///   - `i8`, `u8`: 1 byte
/// - The bytes are interpreted using `std.mem.bytesToValue`, which performs a safe reinterpretation
///   of the byte sequence into the corresponding numeric type.
/// - The result is wrapped in a `NumericUnion` tagged with the appropriate type.
///
/// # Example
/// ```zig
/// const ndarray = NdArray.init(...);
/// ndarray.dtype = .f32;
/// ndarray.data = &[_]u8{ 0x40, 0x49, 0x0f, 0xdb }; // Represents 3.14159 in f32
///
/// const value = ndarray.bytes_to_val(0);
/// std.debug.print("Value: {}\n", .{value.f32}); // Output: Value: 3.14159
/// ```
///
/// # Notes
/// - The function assumes that the `byte_offset` is valid and that there are enough bytes remaining
///   in the data buffer to read the required number of bytes for the `dtype`.
/// - If the `byte_offset` is out of bounds or the data buffer is too small, the behavior is undefined.
///
/// # See Also
/// - `NumericUnion`: The tagged union type that holds the interpreted value.
/// - `std.mem.bytesToValue`: The function used to reinterpret bytes as a numeric value.
pub fn bytes_to_val(self: *const NdArray, byte_offset: usize) dtypes.NumericUnion {
    return switch (self.dtype) {
        .f32 => .{ .f32 = std.mem.bytesToValue(f32, self.data[byte_offset..][0..4]) },
        .f64 => .{ .f64 = std.mem.bytesToValue(f64, self.data[byte_offset..][0..8]) },
        .i8 => .{ .i8 = std.mem.bytesToValue(i8, self.data[byte_offset..][0..1]) },
        .i16 => .{ .i16 = std.mem.bytesToValue(i16, self.data[byte_offset..][0..2]) },
        .i32 => .{ .i32 = std.mem.bytesToValue(i32, self.data[byte_offset..][0..4]) },
        .i64 => .{ .i64 = std.mem.bytesToValue(i64, self.data[byte_offset..][0..8]) },
        .u8 => .{ .u8 = std.mem.bytesToValue(u8, self.data[byte_offset..][0..1]) },
        .u16 => .{ .u16 = std.mem.bytesToValue(u16, self.data[byte_offset..][0..2]) },
        .u32 => .{ .u32 = std.mem.bytesToValue(u32, self.data[byte_offset..][0..4]) },
        .u64 => .{ .u64 = std.mem.bytesToValue(u64, self.data[byte_offset..][0..8]) },
    };
}

/// Converts a `NumericUnion` value into bytes and writes them into the NdArray's data buffer at the specified offset.
///
/// This function takes a `NumericUnion` value and writes its byte representation into the NdArray's data buffer
/// starting at the specified `byte_offset`. The number of bytes written depends on the NdArray's `dtype`.
///
/// # Parameters
/// - `self`: A pointer to the `NdArray` instance whose data buffer is being modified.
/// - `byte_offset`: The starting index in the data buffer (in bytes) where the value will be written.
/// - `value`: The `NumericUnion` value to be converted into bytes and written into the buffer.
///
/// # Behavior
/// - The function writes the byte representation of `value` into `self.data` starting at `byte_offset`.
/// - The number of bytes written depends on the `dtype` of the NdArray:
///   - `f32`, `i32`, `u32`: 4 bytes
///   - `f64`, `i64`, `u64`: 8 bytes
///   - `i16`, `u16`: 2 bytes
///   - `i8`, `u8`: 1 byte
/// - The function uses `@memcpy` to safely copy the bytes from the `NumericUnion` value into the data buffer.
/// - The `NumericUnion` is interpreted based on the `dtype` of the NdArray, and only the relevant field is used.
///
/// # Example
/// ```zig
/// const ndarray = NdArray.init(...);
/// ndarray.dtype = .f32;
/// ndarray.data = &[_]u8{0} ** 4; // Initialize a buffer with 4 bytes
///
/// const value = NumericUnion{ .f32 = 3.14159 };
/// ndarray.val_to_bytes(0, value);
///
/// // The buffer now contains the bytes representing 3.14159 in f32 format.
/// std.debug.print("Buffer: {x}\n", .{ndarray.data});
/// ```
///
/// # Notes
/// - The function assumes that the `byte_offset` is valid and that there are enough bytes remaining
///   in the data buffer to write the required number of bytes for the `dtype`.
/// - If the `byte_offset` is out of bounds or the data buffer is too small, the behavior is undefined.
/// - The `value` parameter must match the `dtype` of the NdArray; otherwise, the behavior is undefined.
///
/// # See Also
/// - `NumericUnion`: The tagged union type that holds the value to be written.
/// - `@memcpy`: The built-in function used to copy bytes into the data buffer.
/// - `std.mem.asBytes`: The function used to obtain the byte representation of a value.
pub fn val_to_bytes(self: *NdArray, byte_offset: usize, value: dtypes.NumericUnion) void {
    switch (self.dtype) {
        .f32 => @memcpy(self.data[byte_offset..][0..@sizeOf(f32)], std.mem.asBytes(&value.f32)),
        .f64 => @memcpy(self.data[byte_offset..][0..@sizeOf(f64)], std.mem.asBytes(&value.f64)),
        .i8 => @memcpy(self.data[byte_offset..][0..@sizeOf(i8)], std.mem.asBytes(&value.i8)),
        .i16 => @memcpy(self.data[byte_offset..][0..@sizeOf(i16)], std.mem.asBytes(&value.i16)),
        .i32 => @memcpy(self.data[byte_offset..][0..@sizeOf(i32)], std.mem.asBytes(&value.i32)),
        .i64 => @memcpy(self.data[byte_offset..][0..@sizeOf(i64)], std.mem.asBytes(&value.i64)),
        .u8 => @memcpy(self.data[byte_offset..][0..@sizeOf(u8)], std.mem.asBytes(&value.u8)),
        .u16 => @memcpy(self.data[byte_offset..][0..@sizeOf(u16)], std.mem.asBytes(&value.u16)),
        .u32 => @memcpy(self.data[byte_offset..][0..@sizeOf(u32)], std.mem.asBytes(&value.u32)),
        .u64 => @memcpy(self.data[byte_offset..][0..@sizeOf(u64)], std.mem.asBytes(&value.u64)),
    }
}

/// Allocates and initializes an array of indices for traversing an NdArray.
///
/// # Parameters
/// - `rank`: The number of dimensions (rank) of the NdArray.
/// - `allocator`: The memory allocator to use for the allocation.
///
/// # Returns
/// A slice of `usize` representing the indices, initialized to zero.
///
/// # Errors
/// Returns an error if the allocation fails.
pub fn alloc_indices(rank: usize, allocator: std.mem.Allocator) ![]usize {
    const indices = try allocator.alloc(usize, rank);
    for (0..rank) |i| {
        indices[i] = 0;
    }
    return indices;
}

/// Prints the contents of an NdArray in a human-readable format.
///
/// # Parameters
/// - `ndarray`: A pointer to the NdArray to be printed.
/// - `allocator`: The memory allocator to use for temporary allocations.
///
/// # Errors
/// Returns an error if allocation or printing fails.
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

/// Recursively prints the contents of an NdArray, handling nested dimensions.
///
/// # Parameters
/// - `ndarray`: A pointer to the NdArray to be printed.
/// - `writer`: The output writer (e.g., stdout).
/// - `dim_level`: The current dimension level being processed.
/// - `indent`: The indentation level for pretty-printing.
/// - `allocator`: The memory allocator to use for temporary allocations.
/// - `indices`: The current indices for traversing the NdArray.
///
/// # Errors
/// Returns an error if allocation or printing fails.
pub fn print_ndarray_recursive(ndarray: *const NdArray, writer: anytype, dim_level: usize, indent: usize, allocator: std.mem.Allocator, indices: []usize) !void {
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

    try writer.print("{s}[", .{spaces[0..indent]});
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
    try writer.print("{s}]", .{spaces[0..indent]});
}

/// Converts a flat index into a multi-dimensional index based on the given shape.
///
/// # Parameters
/// - `allocator`: The memory allocator to use for the allocation.
/// - `flat_index`: The flat index to convert.
/// - `shape`: The shape of the NdArray.
///
/// # Returns
/// A slice of `usize` representing the multi-dimensional index.
///
/// # Errors
/// Returns an error if the allocation fails.
pub fn unravel_index(allocator: std.mem.Allocator, flat_index: usize, shape: []const usize) ![]const usize {
    const n_dims = shape.len;
    var indices = try allocator.alloc(usize, n_dims);
    errdefer allocator.free(indices);

    var remaining_index = flat_index;
    for (0.., shape) |i, dim_size| {
        indices[i] = remaining_index % dim_size;
        remaining_index /= dim_size;
    }

    return indices;
}

/// Prints detailed information about an NdArray, including its shape, rank, size, and dtype.
///
/// # Parameters
/// - `self`: A pointer to the NdArray instance.
/// - `allocator`: The memory allocator to use for temporary allocations.
///
/// # Behavior
/// - If the NdArray has 4 or fewer elements, the entire tensor is printed.
/// - Otherwise, only the first and last elements are printed.
///
/// # Errors
/// Returns an error if allocation or printing fails.
pub fn print_ndarray_info(self: *NdArray, allocator: std.mem.Allocator) !void {
    // Print shape
    const shape_str = try std.fmt.allocPrint(allocator, "{any}", .{self.shape});
    defer allocator.free(shape_str);

    const total_size = compute_size(self.shape);

    if (total_size <= 4) {
        // Print the entire tensor
        const values = try get_all_elements(self, allocator);
        defer allocator.free(values);

        // Format all numeric values into a string
        const values_str = try format_numeric_values(values, self.dtype, allocator);
        defer allocator.free(values_str);

        std.debug.print("{s}, shape: {s}, rank: {d}, size: {d}, dtype: {s}", .{
            values_str,
            shape_str,
            self.shape.len,
            total_size,
            @tagName(self.dtype),
        });
    } else {
        // Compute first and last indices
        const first_index = try allocator.alloc(usize, self.shape.len);
        defer allocator.free(first_index);
        @memset(first_index, 0); // First index is all zeros

        const last_index = try allocator.alloc(usize, self.shape.len);
        defer allocator.free(last_index);

        var remaining = total_size - 1;
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
            total_size,
            @tagName(self.dtype),
        });
    }
}

/// Retrieves all elements of an NdArray as a slice of `NumericUnion`.
///
/// # Parameters
/// - `self`: A pointer to the NdArray instance.
/// - `allocator`: The memory allocator to use for the allocation.
///
/// # Returns
/// A slice of `NumericUnion` containing all elements of the NdArray.
///
/// # Errors
/// Returns an error if allocation or element retrieval fails.
fn get_all_elements(self: *NdArray, allocator: std.mem.Allocator) ![]dtypes.NumericUnion {
    const total_size = compute_size(self.shape);
    const elements = try allocator.alloc(dtypes.NumericUnion, total_size);
    const indices = try allocator.alloc(usize, self.shape.len);
    defer allocator.free(indices);
    @memset(indices, 0);

    for (0..total_size) |i| {
        elements[i] = try self.get(indices);
        increment_indices(indices, self.shape);
    }

    return elements;
}

/// Increments the indices for traversing an NdArray in row-major order.
///
/// # Parameters
/// - `indices`: The current indices to increment.
/// - `shape`: The shape of the NdArray.
fn increment_indices(indices: []usize, shape: []const usize) void {
    for (indices, 0..) |*index, i| {
        index.* += 1; // Corrected: Use `index.*` to access the value
        if (index.* < shape[i]) {
            break;
        }
        index.* = 0;
    }
}

/// Formats an array of `NumericUnion` values into a string representation.
///
/// # Parameters
/// - `values`: The array of `NumericUnion` values to format.
/// - `dtype`: The data type of the values.
/// - `allocator`: The memory allocator to use for the allocation.
///
/// # Returns
/// A formatted string representation of the values.
///
/// # Errors
/// Returns an error if allocation or formatting fails.
fn format_numeric_values(values: []const dtypes.NumericUnion, dtype: dtypes.DataType, allocator: std.mem.Allocator) ![]const u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try buffer.appendSlice("[");
    for (values, 0..) |value, i| {
        if (i > 0) {
            try buffer.appendSlice(", ");
        }
        const value_str = try dtypes.format_numeric_value(value, dtype, allocator);
        defer allocator.free(value_str);
        try buffer.appendSlice(value_str);
    }
    try buffer.appendSlice("]");

    return buffer.toOwnedSlice();
}

/// Computes the total number of elements in an NdArray based on its shape.
///
/// # Parameters
/// - `shape`: The shape of the NdArray.
///
/// # Returns
/// The total number of elements.
pub fn compute_size(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| size *= dim;
    return size;
}

/// Initializes an NdArray using Xavier (Glorot) initialization.
///
/// # Parameters
/// - `allocator`: The memory allocator to use for the allocation.
/// - `shape`: The shape of the NdArray.
/// - `dtype`: The data type of the NdArray.
/// - `random_seed`: The seed for the random number generator.
///
/// # Returns
/// A pointer to the initialized NdArray.
///
/// # Errors
/// Returns an error if allocation or initialization fails.
pub fn xavier_initialization(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, random_seed: u64) !*NdArray {
    const size = compute_size(shape);
    const element_size = dtype.sizeInBytes();
    const total_bytes = size * element_size;

    const data = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(data);

    const fan_in = @as(f32, @floatFromInt(shape[0]));
    const fan_out = @as(f32, @floatFromInt(shape[1]));
    const limit = std.math.sqrt(6.0 / (fan_in + fan_out));

    // Initialize a deterministic RNG with the provided seed
    var rng = std.Random.DefaultPrng.init(random_seed);

    switch (dtype) {
        .f32 => {
            const values = std.mem.bytesAsSlice(f32, data);
            for (values) |*v| {
                // Generate a random float in the range [-limit, limit]
                v.* = rng.random().float(f32) * 2.0 * limit - limit;
            }
        },
        .f64 => {
            const values = std.mem.bytesAsSlice(f64, data);
            for (values) |*v| {
                // Generate a random float in the range [-limit, limit]
                v.* = rng.random().float(f64) * 2.0 * limit - limit;
            }
        },
        else => return error.UnsupportedDataType,
    }

    return NdArray.from_bytes(allocator, shape, dtype, data);
}

/// Checks if two shapes are compatible for broadcasting.
///
/// # Parameters
/// - `shape1`: The first shape.
/// - `shape2`: The second shape.
///
/// # Returns
/// `true` if the shapes are compatible for broadcasting, otherwise `false`.
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

/// Computes the resulting shape after broadcasting two shapes.
///
/// # Parameters
/// - `shape1`: The first shape.
/// - `shape2`: The second shape.
/// - `allocator`: The memory allocator to use for the allocation.
///
/// # Returns
/// The resulting broadcasted shape.
///
/// # Errors
/// Returns an error if allocation fails.
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

/// Converts a scalar value to the specified data type.
///
/// # Parameters
/// - `value`: The scalar value to convert.
/// - `dtype`: The target data type.
///
/// # Returns
/// A `NumericUnion` containing the converted value.
///
/// # Errors
/// Returns an error if the conversion fails (e.g., negative value for unsigned types).
pub fn convert_value_to_dtype(value: anytype, dtype: dtypes.DataType) !dtypes.NumericUnion {
    return switch (dtype) {
        .f32 => dtypes.NumericUnion{ .f32 = @floatCast(value) },
        .f64 => dtypes.NumericUnion{ .f64 = @floatCast(value) },
        .i8 => dtypes.NumericUnion{ .i8 = @intFromFloat(@floor(@as(f64, value))) },
        .i16 => dtypes.NumericUnion{ .i16 = @intFromFloat(@floor(@as(f64, value))) },
        .i32 => dtypes.NumericUnion{ .i32 = @intFromFloat(@floor(@as(f64, value))) },
        .i64 => dtypes.NumericUnion{ .i64 = @intFromFloat(@floor(@as(f64, value))) },
        .u8 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return dtypes.NumericUnion{ .u8 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u16 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return dtypes.NumericUnion{ .u16 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u32 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return dtypes.NumericUnion{ .u32 = @intFromFloat(@floor(@as(f64, value))) };
        },
        .u64 => {
            if (value < 0) return error.NegativeValueForUnsignedType;
            return dtypes.NumericUnion{ .u64 = @intFromFloat(@floor(@as(f64, value))) };
        },
    };
}

//=================================================
// Tests
//=================================================

test "xavier_initialization()" {
    const allocator = std.testing.allocator;

    const shape = &[_]usize{ 4, 5 }; // Example shape
    const dtype = dtypes.DataType.f32; // Example data type
    const random_seed: u64 = 12345; // Example seed

    const weights = try xavier_initialization(allocator, shape, dtype, random_seed);
    defer weights.deinit();

    // Print the initialized weights
    // try weights.print();
}
