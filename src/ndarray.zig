const std = @import("std");
const zorch = @import("zorch.zig");

const ops = zorch.ops;
const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;

const NdUnaOp = ops.NdArrayUnaryOperation;
const NdNdBinOp = ops.NdArrayBinaryOperation;
const NdRedOp = ops.NdArrayReductionOperation;
const NdScBinOp = ops.NdArrayScalarBinaryOperation;

/// A structure representing a multi-dimensional array (NdArray).
///
/// This struct contains the following fields:
/// - `shape`: The shape of the array (dimensions).
/// - `dtype`: The data type of the array elements.
/// - `data`: The underlying data buffer.
/// - `strides`: The strides for each dimension.
/// - `allocator`: The memory allocator used for array operations.
/// - `owns_data`: A flag indicating whether the array owns its data.
pub const NdArray = struct {
    shape: []usize,
    dtype: dtypes.DataType,
    data: []u8,
    strides: []usize,
    allocator: std.mem.Allocator,
    owns_data: bool, // Flag to track ownership of data

    /// Initializes a new NdArray with the given shape and data type.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType) !*NdArray {
        const ndarray = try allocator.create(NdArray);
        errdefer allocator.destroy(ndarray);

        ndarray.shape = try allocator.dupe(usize, shape);
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

    /// Deinitializes the NdArray, freeing all associated resources.
    pub fn deinit(self: *NdArray) void {
        if (self.owns_data) {
            self.allocator.free(self.data);
        }
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        self.allocator.destroy(self);
    }

    /// Broadcasts the NdArray to a target shape.
    ///
    /// # Parameters
    /// - `target_shape`: The target shape to broadcast to.
    ///
    /// # Returns
    /// A new NdArray with the broadcasted shape.
    ///
    /// # Errors
    /// Returns an error if the broadcast is invalid or allocation fails.
    pub fn broadcast_to(self: *const NdArray, target_shape: []const usize) !*NdArray {
        // Check if the broadcast is valid
        if (!utils.is_compatible_for_broadcast(self.shape, target_shape)) {
            return error.InvalidBroadcast;
        }

        // Create a new NdArray with the target shape
        var broadcasted_array = try NdArray.init(self.allocator, target_shape, self.dtype);
        errdefer broadcasted_array.deinit();

        const dtype_size = self.dtype.sizeInBytes();
        const total_elements = utils.compute_size(target_shape);

        for (0..total_elements) |dst_idx| {
            // Compute coordinates in the target shape
            var coords = try self.allocator.alloc(usize, target_shape.len);
            defer self.allocator.free(coords);
            var tmp = dst_idx;

            for (0..target_shape.len) |i| {
                var product_after: usize = 1;
                for (target_shape[i + 1 ..]) |dim| {
                    product_after *= dim;
                }
                coords[i] = tmp / product_after;
                tmp = tmp % product_after;
            }

            // Compute source index based on original array's shape and strides
            var src_idx: usize = 0;
            for (0..target_shape.len) |axis| {
                const original_dim = if (axis < self.shape.len) self.shape[axis] else 1;
                const source_coord = if (original_dim != 1) coords[axis] else 0;

                if (axis < self.shape.len) {
                    src_idx += source_coord * self.strides[axis];
                }
            }

            // Ensure src_idx is within bounds of the original array
            const original_size = utils.compute_size(self.shape);
            if (src_idx >= original_size) {
                return error.IndexOutOfBounds;
            }

            // Copy data from the original array to the new array
            @memcpy(
                broadcasted_array.data[dst_idx * dtype_size .. (dst_idx + 1) * dtype_size],
                self.data[src_idx * dtype_size .. (src_idx + 1) * dtype_size],
            );
        }

        return broadcasted_array;
    }

    /// Computes the strides for a given shape.
    ///
    /// # Parameters
    /// - `shape`: The shape of the array.
    /// - `strides`: The strides array to populate.
    pub fn compute_strides(shape: []const usize, strides: []usize) void {
        var stride: usize = 1;
        for (0..shape.len) |i| {
            strides[shape.len - 1 - i] = stride;
            stride *= shape[shape.len - 1 - i];
        }
    }

    /// Slices the NdArray along a specified dimension.
    ///
    /// # Parameters
    /// - `dim`: The dimension to slice.
    /// - `start`: The starting index of the slice.
    /// - `end`: The ending index of the slice.
    ///
    /// # Returns
    /// A new NdArray representing the sliced data.
    ///
    /// # Errors
    /// Returns an error if the slice indices are invalid or allocation fails.
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

    /// Creates an NdArray from a scalar value.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    /// - `value`: The scalar value to initialize the array with.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_value(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, value: anytype) !*NdArray {
        const size = utils.compute_size(shape);
        const element_size = dtype.sizeInBytes();
        const total_bytes = size * element_size;

        const data = try allocator.alloc(u8, total_bytes);
        errdefer allocator.free(data);

        const numeric_value = utils.convert_value_to_dtype(value, dtype);
        const value_bytes = std.mem.asBytes(&numeric_value);

        for (0..size) |i| {
            const offset = i * element_size;
            @memcpy(data[offset .. offset + element_size], value_bytes[0..element_size]);
        }

        return NdArray.from_bytes(allocator, shape, dtype, data);
    }

    /// Creates an NdArray filled with zeros.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn zeros(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType) !*NdArray {
        return NdArray.from_value(allocator, shape, dtype, 0.0);
    }

    /// Creates an NdArray filled with ones.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn ones(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType) !*NdArray {
        return NdArray.from_value(allocator, shape, dtype, 1.0);
    }

    /// Creates an NdArray filled with random values.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn random(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType) !*NdArray {
        const size = utils.compute_size(shape);
        const element_size = dtype.sizeInBytes();
        const total_bytes = size * element_size;

        const data = try allocator.alloc(u8, total_bytes);
        errdefer allocator.free(data);

        switch (dtype) {
            .f32 => {
                const values = std.mem.bytesAsSlice(f32, data);
                for (values) |*v| {
                    v.* = std.crypto.random.float(f32);
                }
            },
            .f64 => {
                const values = std.mem.bytesAsSlice(f64, data);
                for (values) |*v| {
                    v.* = std.crypto.random.float(f64);
                }
            },
            .i32 => {
                const values = std.mem.bytesAsSlice(i32, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(i32);
                }
            },
            .i64 => {
                const values = std.mem.bytesAsSlice(i64, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(i64);
                }
            },
            .u8 => {
                const values = std.mem.bytesAsSlice(u8, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(u8);
                }
            },
            .i8 => {
                const values = std.mem.bytesAsSlice(i8, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(i8);
                }
            },
            .i16 => {
                const values = std.mem.bytesAsSlice(i16, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(i16);
                }
            },
            .u16 => {
                const values = std.mem.bytesAsSlice(u16, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(u16);
                }
            },
            .u32 => {
                const values = std.mem.bytesAsSlice(u32, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(u32);
                }
            },
            .u64 => {
                const values = std.mem.bytesAsSlice(u64, data);
                for (values) |*v| {
                    v.* = std.crypto.random.int(u64);
                }
            },
        }

        return NdArray.from_bytes(allocator, shape, dtype, data);
    }

    /// Creates an NdArray from a byte buffer.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    /// - `data`: The byte buffer containing the array data.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_bytes(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, data: []u8) !*NdArray {
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

    /// Creates an NdArray from raw data.
    ///
    /// # Parameters
    /// - `allocator`: The memory allocator to use for array operations.
    /// - `shape`: The shape of the array.
    /// - `dtype`: The data type of the array elements.
    /// - `data`: The raw data to initialize the array with.
    ///
    /// # Returns
    /// A pointer to the newly created NdArray.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn from_data(allocator: std.mem.Allocator, shape: []const usize, dtype: dtypes.DataType, data: anytype) !*NdArray {
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

    /// Creates a clone of the NdArray.
    ///
    /// # Returns
    /// A new NdArray with the same data and properties.
    ///
    /// # Errors
    /// Returns an error if allocation or initialization fails.
    pub fn clone(self: *NdArray) !*NdArray {
        const cloned = try NdArray.init(self.allocator, self.shape, self.dtype);
        @memcpy(cloned.data, self.data);
        return cloned;
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
    pub fn get(self: *const NdArray, indices: []const usize) !dtypes.NumericUnion {
        for (indices, self.shape) |idx, dim_size| {
            if (idx >= dim_size) return error.IndexOutOfBounds;
        }
        const flat_idx = self.flatten_idx(indices);
        const byte_offset = flat_idx * self.dtype.sizeInBytes();
        return utils.bytes_to_val(self, byte_offset);
    }

    /// Sets the value at the specified indices.
    ///
    /// # Parameters
    /// - `indices`: The indices of the value to set.
    /// - `value`: The value to set.
    ///
    /// # Errors
    /// Returns an error if the indices are invalid or the value type is incompatible.
    pub fn set(self: *NdArray, indices: []const usize, value: dtypes.NumericUnion) !void {
        for (indices, self.shape) |idx, dim_size| {
            if (idx >= dim_size) return error.IndexOutOfBounds;
        }
        const flat_idx = self.flatten_idx(indices);
        const byte_offset = flat_idx * self.dtype.sizeInBytes();
        utils.val_to_bytes(self, byte_offset, value);
    }

    /// Returns the total number of elements in the NdArray.
    pub fn len(self: *NdArray) usize {
        var size: usize = 1;
        for (self.shape) |dim| size *= dim;
        return size;
    }

    /// Sets all elements of the NdArray to a specified value.
    ///
    /// # Parameters
    /// - `value`: The value to set.
    pub fn set_all(self: *NdArray, value: anytype) void {
        const total_size = utils.compute_size(self.shape);
        const element_size = self.dtype.sizeInBytes();
        const numeric_value = utils.convert_value_to_dtype(value, self.dtype) catch |err| {
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

    /// Fills the NdArray with a specified value.
    ///
    /// # Parameters
    /// - `value`: The value to fill the array with.
    ///
    /// # Errors
    /// Returns an error if the value type is incompatible.
    pub fn fill(self: *NdArray, value: anytype) !void {
        self.set_all(value);
    }

    /// Computes the flat index for a given set of indices.
    ///
    /// # Parameters
    /// - `indices`: The indices to flatten.
    ///
    /// # Returns
    /// The flat index corresponding to the given indices.
    fn flatten_idx(self: *const NdArray, indices: []const usize) usize {
        var idx: usize = 0;
        for (indices, self.strides) |i, stride| {
            idx += i * stride;
        }
        return idx;
    }

    /// Reshapes the NdArray to a new shape.
    ///
    /// # Parameters
    /// - `new_shape`: The new shape of the array.
    ///
    /// # Returns
    /// A new NdArray with the reshaped data.
    ///
    /// # Errors
    /// Returns an error if the new shape is incompatible with the current size.
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

    /// Prints the contents of the NdArray.
    ///
    /// # Errors
    /// Returns an error if printing fails.
    pub fn print(self: *NdArray) !void {
        try utils.print_ndarray(self, self.allocator);
    }

    /// Prints detailed information about the NdArray.
    ///
    /// # Errors
    /// Returns an error if printing fails.
    pub fn info(self: *NdArray) !void {
        std.debug.print("NdArray: [", .{});
        try utils.print_ndarray_info(self, self.allocator);
        std.debug.print("]\n", .{});
    }

    /// Casts the NdArray to a new data type.
    ///
    /// # Parameters
    /// - `target_dtype`: The target data type.
    ///
    /// # Returns
    /// A new NdArray with the casted data.
    ///
    /// # Errors
    /// Returns an error if the cast operation fails.
    pub fn cast(self: *NdArray, target_dtype: dtypes.DataType) !*NdArray {
        return try ops.cast_elements(self, target_dtype);
    }

    // ===========================
    // Reduction Operations
    // ===========================
    /// Returns the total number of elements in the NdArray.
    pub fn get_size(self: *NdArray) usize {
        var size: usize = 1;
        for (self.shape) |dim| size *= dim;
        return size;
    }

    /// Computes the indices of the minimum values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the indices (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the indices of the minimum values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn argmin(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.ARGMIN, axis, keepdims);
    }

    /// Computes the indices of the maximum values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the indices (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the indices of the maximum values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn argmax(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.ARGMAX, axis, keepdims);
    }

    /// Computes the minimum values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the minimum (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the minimum values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn min(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.MIN, axis, keepdims);
    }

    /// Computes the maximum values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the maximum (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the maximum values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn max(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.MAX, axis, keepdims);
    }

    /// Computes the sum of values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the sum (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the sum of values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sum(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.SUM, axis, keepdims);
    }

    /// Computes the mean of values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the mean (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the mean of values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn mean(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.MEAN, axis, keepdims);
    }

    /// Computes the product of values along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to compute the product (optional).
    /// - `keepdims`: Whether to keep the reduced dimensions.
    ///
    /// # Returns
    /// A new NdArray containing the product of values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn prod(self: *NdArray, axis: ?usize, keepdims: bool) !*NdArray {
        return try ops.reduce(self, NdRedOp.PROD, axis, keepdims);
    }

    // ===========================
    // Unary Operations
    // ===========================
    /// Applies the natural logarithm element-wise.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the logarithm applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn log(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.LOG, in_place);
    }

    /// Applies the exponential function element-wise.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the exponential applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn exp(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.EXP, in_place);
    }

    /// Applies the hyperbolic tangent function element-wise.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the hyperbolic tangent applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn tanh(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.TANH, in_place);
    }

    /// Applies the Rectified Linear Unit (ReLU) function element-wise.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the ReLU applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn relu(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.RELU, in_place);
    }

    /// Applies the Softmax function along a specified axis.
    ///
    /// # Parameters
    /// - `axis`: The axis along which to apply Softmax (optional).
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the Softmax applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn softmax(self: *NdArray, axis: ?usize, in_place: bool) !*NdArray {
        // Step 1: Compute max along the axis
        const max_x = try self.max(axis, false);
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
        const sum_exp_x = try exp_x.sum(axis, false);
        defer sum_exp_x.deinit();

        // Reshape sum_exp_x to match the dimensionality of exp_x
        const reshaped_sum_exp_x = try sum_exp_x.reshape(&[_]usize{ sum_exp_x.shape[0], 1 });
        defer reshaped_sum_exp_x.deinit();

        // Step 5: Normalize by dividing exp_x by reshaped_sum_exp_x
        const z = try ops.apply_elementwise(exp_x, reshaped_sum_exp_x, self.allocator, NdNdBinOp.DIV, in_place);
        // Do NOT deinit z here! The caller is responsible for deallocating the result.
        return z;
    }

    /// Applies the Sigmoid function element-wise.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the Sigmoid applied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sigmoid(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.SIGMOID, in_place);
    }

    /// Negates the elements of the NdArray.
    ///
    /// # Parameters
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the negated values.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn neg(self: *NdArray, in_place: bool) !*NdArray {
        return try ops.apply_unary(self, NdUnaOp.NEG, in_place);
    }

    // ===============================
    // Binary Operations (with Scalar)
    // ===============================
    /// Adds a scalar value to each element of the NdArray.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to add.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the scalar added.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn add_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.ADD, in_place);
    }

    /// Subtracts a scalar value from each element of the NdArray.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to subtract.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the scalar subtracted.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sub_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.SUB, in_place);
    }

    /// Multiplies each element of the NdArray by a scalar value.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to multiply.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the scalar multiplied.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn mul_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.MUL, in_place);
    }

    /// Divides each element of the NdArray by a scalar value.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to divide by.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the scalar divided.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn div_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.DIV, in_place);
    }

    /// Raises each element of the NdArray to the power of a scalar value.
    ///
    /// # Parameters
    /// - `scalar`: The scalar exponent.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the elements raised to the power of the scalar.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn pow_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.POW, in_place);
    }

    /// Compares each element of the NdArray to a scalar value (greater than).
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn greater_than_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.GT, in_place);
    }

    /// Compares each element of the NdArray to a scalar value (less than).
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn smaller_than_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.LT, in_place);
    }

    /// Compares each element of the NdArray to a scalar value (equal to).
    ///
    /// # Parameters
    /// - `scalar`: The scalar value to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn equal_to_scalar(self: *NdArray, scalar: anytype, in_place: bool) !*NdArray {
        return try ops.apply_binary(self, scalar, NdScBinOp.EQ, in_place);
    }
    // ================================
    // Binary Operations (with NdArray)
    // ================================
    /// Adds another NdArray element-wise.
    ///
    /// # Parameters
    /// - `other`: The NdArray to add.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the result of the addition.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn add(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.ADD, in_place);
    }

    /// Subtracts another NdArray element-wise.
    ///
    /// # Parameters
    /// - `other`: The NdArray to subtract.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the result of the subtraction.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn sub(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.SUB, in_place);
    }

    /// Multiplies another NdArray element-wise.
    ///
    /// # Parameters
    /// - `other`: The NdArray to multiply.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the result of the multiplication.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn mul(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.MUL, in_place);
    }

    /// Divides another NdArray element-wise.
    ///
    /// # Parameters
    /// - `other`: The NdArray to divide by.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the result of the division.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn div(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.DIV, in_place);
    }

    /// Raises each element of the NdArray to the power of the corresponding element in another NdArray.
    ///
    /// # Parameters
    /// - `other`: The NdArray containing the exponents.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with the result of the power operation.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn pow(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        return try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.POW, in_place);
    }

    /// Compares the NdArray to another NdArray element-wise (equal to).
    ///
    /// # Parameters
    /// - `other`: The NdArray to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn equal(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const equal_ndarray = try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.EQ, in_place);
        return equal_ndarray.cast(.i32);
    }

    /// Compares the NdArray to another NdArray element-wise (greater than).
    ///
    /// # Parameters
    /// - `other`: The NdArray to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn greater_than(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const gt_ndarray = try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.GT, in_place);
        return gt_ndarray.cast(.i32);
    }

    /// Compares the NdArray to another NdArray element-wise (less than).
    ///
    /// # Parameters
    /// - `other`: The NdArray to compare against.
    /// - `in_place`: Whether to perform the operation in-place.
    ///
    /// # Returns
    /// A new NdArray with boolean values (1 for true, 0 for false).
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn less_than(self: *NdArray, other: *const NdArray, in_place: bool) !*NdArray {
        const lt_ndarray = try ops.apply_elementwise(self, other, self.allocator, NdNdBinOp.LT, in_place);
        return lt_ndarray.cast(.i32);
    }

    /// Performs matrix multiplication with another NdArray.
    ///
    /// # Parameters
    /// - `b`: The NdArray to multiply with.
    /// - `alpha`: A scaling factor for the result.
    /// - `beta`: A scaling factor for the input.
    /// - `transA`: Whether to transpose the first matrix.
    /// - `transB`: Whether to transpose the second matrix.
    ///
    /// # Returns
    /// A new NdArray with the result of the matrix multiplication.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    pub fn matmul(self: *NdArray, b: *const NdArray, alpha: f32, beta: f32, transA: bool, transB: bool) !*NdArray {
        return try ops.gemm(self, b, alpha, beta, transA, transB, self.allocator);
    }

    /// Transposes the NdArray.
    ///
    /// # Returns
    /// A new NdArray with the transposed shape and strides.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
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

        errdefer {
            self.allocator.free(transposed.shape);
            self.allocator.free(transposed.data);
            self.allocator.free(transposed.strides); // Free strides on error
            self.allocator.destroy(transposed);
        }

        return transposed;
    }
};

// ============================
// Tests for the NdArray struct
// ============================
const expect = std.testing.expect;

test "test min()" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0, -1.0, 0.0, 5.0 };
    const shape = &[_]usize{ 2, 3 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    const min = try original_array.min(null, false);
    defer min.deinit();
}

test "test max()" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0, -1.0, 0.0, 5.0 };
    const shape = &[_]usize{ 2, 3 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    const max = try original_array.max(null, false);
    defer max.deinit();
}

test "test argmin()" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0, -1.0, 0.0, 5.0 };
    const shape = &[_]usize{ 2, 3 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    const argmin = try original_array.argmin(null, false);
    defer argmin.deinit();

    // Get the flat index of the minimum value
    const flat_index = (try argmin.get(&[_]usize{0})).u64;

    // Convert the flat index to multi-dimensional indices
    const idx = try utils.unravel_index(allocator, flat_index, original_array.shape);
    defer allocator.free(idx); // Free the allocated memory

    // Verify the result
    const expected_argmin = 3; // Index of -1.0 in the flattened array
    const actual_argmin = flat_index;
    try std.testing.expectEqual(expected_argmin, actual_argmin);
}

test "test argmax()" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0, -1.0, 0.0, 5.0 };
    const shape = &[_]usize{ 2, 3 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    const argmax = try original_array.argmax(null, false);
    defer argmax.deinit();

    // Verify the result
    const expected_argmax = 5; // Index of 5.0 in the flattened array
    const actual_argmax = (try argmax.get(&[_]usize{0})).u64;
    try std.testing.expectEqual(expected_argmax, actual_argmax);
}

test "broadcast_to valid broadcast: case 1" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{2.0};
    const shape = &[_]usize{ 1, 1 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    // Target shape: 2x3
    const target_shape = [_]usize{ 3, 2 };

    // Broadcast the array
    const broadcasted_array = try original_array.broadcast_to(&target_shape);
    defer broadcasted_array.deinit();
    // try broadcasted_array.print();
}

test "broadcast_to valid broadcast: case 2" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0 };
    const shape = &[_]usize{ 1, 3 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    // Target shape: 2x3
    const target_shape = [_]usize{ 5, 3 };

    // Broadcast the array
    const broadcasted_array = try original_array.broadcast_to(&target_shape);
    defer broadcasted_array.deinit();
    // try broadcasted_array.print();
}

test "broadcast_to valid broadcast: case 3" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1.0, 2.0, 3.0 };
    const shape = &[_]usize{ 3, 1 };
    const original_array = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer original_array.deinit();

    // Target shape: 2x3
    const target_shape = [_]usize{ 3, 5 };

    // Broadcast the array
    const broadcasted_array = try original_array.broadcast_to(&target_shape);
    defer broadcasted_array.deinit();
    // try broadcasted_array.print();
}
test "reduce with keepdims" {
    const allocator = std.testing.allocator;

    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer ndarray.deinit();

    // Reduce along axis 0 with keepdims=true
    const reduced = try ndarray.sum(0, true);
    defer reduced.deinit();

    // Verify the shape is [1, 3]
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, reduced.shape);

    // Verify the values
    try std.testing.expectEqual(@as(f32, 5), (try reduced.get(&[_]usize{ 0, 0 })).f32);
    try std.testing.expectEqual(@as(f32, 7), (try reduced.get(&[_]usize{ 0, 1 })).f32);
    try std.testing.expectEqual(@as(f32, 9), (try reduced.get(&[_]usize{ 0, 2 })).f32);
}
test "ndarray.random()" {
    const allocator = std.heap.page_allocator;

    const shape = [_]usize{ 2, 3 }; // Shape for a 2x3 matrix
    const dtype = .f32; // Using f32 for this test

    // Create an NdArray with random values
    const random_ndarray = try NdArray.random(allocator, &shape, dtype);
    defer random_ndarray.deinit();

    // Print the random values for inspection
    // std.debug.print("Random NdArray:\n", .{});
    const values = std.mem.bytesAsSlice(f32, random_ndarray.data);
    // for (values) |value| {
    //     std.debug.print("{d:.3} ", .{value});
    // }
    // std.debug.print("\n", .{});

    // Check if the data is within expected ranges (for f32, values should be between 0 and 1)
    for (values) |value| {
        if (value < 0.0 or value > 1.0) {
            // std.debug.print("Value out of range: {}\n", .{value});
            return error.InvalidRandomValue;
        }
    }

    // std.debug.print("Test passed: All random values are within the expected range.\n", .{});
}
test "ndarray.init()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.init(allocator, shape, .f32);
    defer ndarray.deinit();

    // Verify the shape
    std.debug.assert(ndarray.shape[0] == 2);
    std.debug.assert(ndarray.shape[1] == 3);

    // Verify the strides
    std.debug.assert(ndarray.strides[0] == 3);
    std.debug.assert(ndarray.strides[1] == 1);
}

test "ndarray.from_value()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

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
        defer ndarray.deinit();
        std.debug.assert(ndarray.shape.len == shape.len);
    }
}

test "ndarray.from_data()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };
    const ndarray = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer ndarray.deinit();

    std.debug.assert(ndarray.shape.len == shape.len);
    for (ndarray.shape, shape) |a, b| {
        std.debug.assert(a == b);
    }
}

test "ndarray.len()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 3 };
    const ndarray = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer ndarray.deinit();

    try expect(ndarray.len() == 6); // Check length
}

test "ndarray.clone()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 2, 2 };
    const original = try NdArray.from_value(allocator, shape, .f32, @as(f32, 7.0));
    defer original.deinit();

    const cloned = try original.clone();
    defer cloned.deinit();

    const value = try cloned.get(&[_]usize{ 1, 1 });
    std.debug.assert(value.f32 == 7.0);
}

test "ndarray.ones()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const ones = try NdArray.ones(allocator, shape, .f32);
    defer ones.deinit();
}

test "ndarray.zeros()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const zeros = try NdArray.zeros(allocator, shape, .f32);
    defer zeros.deinit();
}

test "ndarray.transpose()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    const debug = false;

    // Original data for the ndarray
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };

    // Create the ndarray from data
    const ndarray = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer ndarray.deinit();

    // Perform transpose operation
    const transposed = try ndarray.transpose();
    defer transposed.deinit();

    if (debug) {
        try ndarray.print();
        try transposed.info();
        try transposed.print();
    }

    // Assert the shapes are transposed correctly
    try expect(ndarray.shape[0] == transposed.shape[1]);
    try expect(ndarray.shape[1] == transposed.shape[0]);

    // Verify that transposed values match the expected values
    for (0..ndarray.shape[0]) |i| {
        for (0..ndarray.shape[1]) |j| {
            const original_val = try ndarray.get(&[_]usize{ i, j });
            const transposed_val = try transposed.get(&[_]usize{ j, i });
            if (debug) {
                std.debug.print("x[{}, {}]={} ? x_t[{}, {}]={}\n", .{ i, j, original_val.f32, i, j, transposed_val.f32 });
            }
            try expect(original_val.f32 == transposed_val.f32);
        }
    }
}

test "ndarray.tanh()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 5.0));
    defer a.deinit();
    const res = try a.tanh(false);
    defer res.deinit();

    try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

    const updated_val = try a.get(&[_]usize{ 0, 0 });
    try expect(updated_val.f32 == 42.0);
}

test "ndarray.relu()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Create a 1D array with both negative and positive values.
    const shape = &[_]usize{4};
    const a = try NdArray.from_value(allocator, shape, .f32, 0.0);
    defer a.deinit();
    // Manually set values.
    try a.set(&[_]usize{0}, .{ .f32 = -3.0 });
    try a.set(&[_]usize{1}, .{ .f32 = -0.5 });
    try a.set(&[_]usize{2}, .{ .f32 = 0.0 });
    try a.set(&[_]usize{3}, .{ .f32 = 2.0 });

    // Apply ReLU (non in-place).
    const res = try a.relu(false);
    defer res.deinit();

    // ReLU(x) = max(0, x)
    const expected = [_]f32{ 0.0, 0.0, 0.0, 2.0 };
    for (0.., expected) |i, exp_val| {
        const out = try res.get(&[_]usize{i});
        try expect(@abs(out.f32 - exp_val) < 0.0001);
    }
}

test "ndarray.sigmoid()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var arr = try NdArray.init(allocator, &[_]usize{3}, .f32); // Shape is [3]
    defer arr.deinit();

    // Fill the array with some values
    try arr.set(&[_]usize{0}, .{ .f32 = 0.0 }); // Set the first element
    try arr.set(&[_]usize{1}, .{ .f32 = 1.0 }); // Set the second element
    try arr.set(&[_]usize{2}, .{ .f32 = -1.0 }); // Set the third element

    const result = try ops.apply_unary(arr, .SIGMOID, false);
    defer result.deinit();

    // Check the results
    const val_1 = try result.get(&[_]usize{0});
    const val_2 = try result.get(&[_]usize{1});
    const val_3 = try result.get(&[_]usize{2});
    try expect(@abs(val_1.f32 - 0.5) < 0.0001); // Check the first element
    try expect(@abs(val_2.f32 - 0.73105857863) < 0.0001); // Check the second element
    try expect(@abs(val_3.f32 - 0.26894142137) < 0.0001); // Check the third element
}

test "ndarray.reshape()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Create a 2D array with shape {3, 4} and values:
    // [[1,  2,  3,  4 ],
    //  [5,  6,  7,  8 ],
    //  [9, 10, 11, 12 ]]
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 3, 4 };

    // Initialize the NdArray
    const arr = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer arr.deinit();

    // Define the new shape {6, 2}
    const new_shape = &[_]usize{ 6, 2 };

    // Reshape the array
    const reshaped_arr = try arr.reshape(new_shape);
    defer reshaped_arr.deinit();

    // Assertions to verify the reshape operation
    std.debug.assert(reshaped_arr.shape.len == 2);
    std.debug.assert(reshaped_arr.shape[0] == 6);
    std.debug.assert(reshaped_arr.shape[1] == 2);
}

test "ndarray.softmax()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Create a 2D array with shape {2, 3} and values: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const arr = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer arr.deinit();

    // Apply softmax (non in-place).
    const res = try arr.softmax(1, false);
    defer res.deinit();

    // Compute expected softmax values for each row.
    const exp1_row1 = @exp(@as(f32, 1.0));
    const exp2_row1 = @exp(@as(f32, 2.0));
    const exp3_row1 = @exp(@as(f32, 3.0));
    const sum_row1 = exp1_row1 + exp2_row1 + exp3_row1;
    const expected_row1 = [_]f32{ exp1_row1 / sum_row1, exp2_row1 / sum_row1, exp3_row1 / sum_row1 };

    const exp1_row2 = @exp(@as(f32, 4.0));
    const exp2_row2 = @exp(@as(f32, 5.0));
    const exp3_row2 = @exp(@as(f32, 6.0));
    const sum_row2 = exp1_row2 + exp2_row2 + exp3_row2;
    const expected_row2 = [_]f32{ exp1_row2 / sum_row2, exp2_row2 / sum_row2, exp3_row2 / sum_row2 };

    // Verify the results for the first row
    for (0..3) |i| {
        const out = try res.get(&[_]usize{ 0, i });
        try std.testing.expectApproxEqAbs(expected_row1[i], out.f32, 0.0001);
    }

    // Verify the results for the second row
    for (0..3) |i| {
        const out = try res.get(&[_]usize{ 1, i });
        try std.testing.expectApproxEqAbs(expected_row2[i], out.f32, 0.0001);
    }
}

test "ndarray.reduce.axis(null)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    var arr = try NdArray.init(allocator, &[_]usize{3}, .f32); // Shape is [3]
    defer arr.deinit();

    // Fill the array with some values
    try arr.set(&[_]usize{0}, .{ .f32 = 1.0 });
    try arr.set(&[_]usize{1}, .{ .f32 = 2.0 });
    try arr.set(&[_]usize{2}, .{ .f32 = 3.0 });

    // Test SUM
    const sum_result = try ops.reduce(arr, .SUM, null, false);
    defer sum_result.deinit();
    const sum_val = try sum_result.get(&[_]usize{0});
    try expect(sum_val.f32 == 6.0);

    // Test MIN
    const min_result = try ops.reduce(arr, .MIN, null, false);
    defer min_result.deinit();
    const min_val = try min_result.get(&[_]usize{0});
    try expect(min_val.f32 == 1.0);

    // Test MAX
    const max_result = try ops.reduce(arr, .MAX, null, false);
    defer max_result.deinit();
    const max_val = try max_result.get(&[_]usize{0});
    try expect(max_val.f32 == 3.0);

    // Test MEAN
    const mean_result = try ops.reduce(arr, .MEAN, null, false);
    defer mean_result.deinit();
    const mean_val = try mean_result.get(&[_]usize{0});
    try expect(mean_val.f32 == 2.0);

    // Test PROD
    const prod_result = try ops.reduce(arr, .PROD, null, false);
    defer prod_result.deinit();
    const prod_val = try prod_result.get(&[_]usize{0});
    try expect(prod_val.f32 == 6.0);
}

test "ndarray.reduce.axis(0)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const arr = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer arr.deinit();

    // Test SUM along axis 1
    const sum_result = try ops.reduce(arr, .SUM, 0, false);
    defer sum_result.deinit();

    const sum_0 = try sum_result.get(&[_]usize{0});
    const sum_1 = try sum_result.get(&[_]usize{1});
    const sum_2 = try sum_result.get(&[_]usize{2});
    try expect(sum_0.f32 == 5.0);
    try expect(sum_1.f32 == 7.0);
    try expect(sum_2.f32 == 9.0);

    // Test MAX along axis 1
    const max_result = try ops.reduce(arr, .MAX, 0, false);
    defer max_result.deinit();

    const max_0 = try max_result.get(&[_]usize{0});
    const max_1 = try max_result.get(&[_]usize{1});
    const max_2 = try max_result.get(&[_]usize{2});
    try expect(max_0.f32 == 4.0);
    try expect(max_1.f32 == 5.0);
    try expect(max_2.f32 == 6.0);

    // Test MIN along axis 1
    const min_result = try ops.reduce(arr, .MIN, 0, false);
    defer min_result.deinit();

    const min_0 = try min_result.get(&[_]usize{0});
    const min_1 = try min_result.get(&[_]usize{1});
    const min_2 = try min_result.get(&[_]usize{2});
    try expect(min_0.f32 == 1.0);
    try expect(min_1.f32 == 2.0);
    try expect(min_2.f32 == 3.0);
}

test "ndarray.reduce.axis(1)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const arr = try NdArray.from_data(allocator, shape, .f32, data_a);
    defer arr.deinit();

    // Test SUM along axis 1
    const sum_result = try ops.reduce(arr, .SUM, 1, false);
    defer sum_result.deinit();

    const sum_0 = try sum_result.get(&[_]usize{0});
    const sum_1 = try sum_result.get(&[_]usize{1});
    try expect(sum_0.f32 == 6.0);
    try expect(sum_1.f32 == 15.0);

    // Test MAX along axis 1
    const max_result = try ops.reduce(arr, .MAX, 1, false);
    defer max_result.deinit();

    const max_0 = try max_result.get(&[_]usize{0});
    const max_1 = try max_result.get(&[_]usize{1});
    try expect(max_0.f32 == 3.0);
    try expect(max_1.f32 == 6.0);

    // Test MIN along axis 1
    const min_result = try ops.reduce(arr, .MIN, 1, false);
    defer min_result.deinit();

    const min_0 = try min_result.get(&[_]usize{0});
    const min_1 = try min_result.get(&[_]usize{1});
    try expect(min_0.f32 == 1.0);
    try expect(min_1.f32 == 4.0);
}

test "ndarray.set()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();

    try a.set(&[_]usize{ 0, 0 }, .{ .f32 = 42.0 });

    const updated_val = try a.get(&[_]usize{ 0, 0 });
    try expect(updated_val.f32 == 42.0);
}

test "ndarray.get()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();

    const value = try a.get(&[_]usize{ 0, 0 });
    try expect(value.f32 == 1.0);
}

test "ndarray.fill()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 0.0));
    defer a.deinit();

    try a.fill(42.0);

    const updated_val = try a.get(&[_]usize{ 0, 0 });
    try expect(updated_val.f32 == 42.0);
}

test "ndarray.add()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();
    const b = try NdArray.from_value(allocator, shape, .f32, @as(f32, 2.0));
    defer b.deinit();

    const sum = try a.add(b, false);
    defer sum.deinit();

    const sum_val = try sum.get(&[_]usize{ 0, 0 });
    try expect(sum_val.f32 == 3.0);
}

test "ndarray.sub()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();
    const b = try NdArray.from_value(allocator, shape, .f32, @as(f32, 2.0));
    defer b.deinit();

    const sub = try a.sub(b, false);
    defer sub.deinit();

    const sub_val = try sub.get(&[_]usize{ 0, 0 });
    try expect(sub_val.f32 == -1.0);
}

test "ndarray.mul()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();
    const b = try NdArray.from_value(allocator, shape, .f32, @as(f32, 2.0));
    defer b.deinit();

    const mul = try a.mul(b, false);
    defer mul.deinit();

    const mul_val = try mul.get(&[_]usize{ 0, 0 });
    try expect(mul_val.f32 == 2.0);
}

test "ndarray.div()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 1.0));
    defer a.deinit();
    const b = try NdArray.from_value(allocator, shape, .f32, @as(f32, 2.0));
    defer b.deinit();

    const div = try a.div(b, false);
    defer div.deinit();

    const div_val = try div.get(&[_]usize{ 0, 0 });
    try expect(div_val.f32 == 0.5);
}

test "ndarray.pow()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const shape = &[_]usize{ 5, 3 };
    const a = try NdArray.from_value(allocator, shape, .f32, @as(f32, 2.0));
    defer a.deinit();
    const b = try NdArray.from_value(allocator, shape, .f32, @as(f32, 3.0));
    defer b.deinit();

    const pow = try a.pow(b, false);
    defer pow.deinit();

    const pow_val = try pow.get(&[_]usize{ 0, 0 });
    try expect(pow_val.f32 == 8.0);
}

test "ndarray.add_scalar()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const test_value = 5;
    const test_scalar = 3;
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const shape = &[_]usize{ 5, 3 };
        const value = switch (dtype) {
            .f32 => @as(f32, test_value),
            .f64 => @as(f64, test_value),
            .i32 => @as(i32, test_value),
            .i64 => @as(i64, test_value),
            .i16 => @as(i16, test_value),
            .i8 => @as(i8, test_value),
            .u32 => @as(u32, test_value),
            .u64 => @as(u64, test_value),
            .u16 => @as(u16, test_value),
            .u8 => @as(u8, test_value),
        };

        const a = try NdArray.from_value(allocator, shape, dtype, value);
        defer a.deinit();

        // Test non-inplace operation
        const result = try a.add_scalar(test_scalar, false);
        defer result.deinit();

        if (debug) {
            try result.info();
        }

        const result_val = try result.get(&[_]usize{ 0, 0 });
        const expected_result = switch (dtype) {
            .f32 => dtypes.ScalarValue{ .f32 = @as(f32, test_value) + @as(f32, test_scalar) },
            .f64 => dtypes.ScalarValue{ .f64 = @as(f64, test_value) + @as(f64, test_scalar) },
            .i32 => dtypes.ScalarValue{ .i32 = @as(i32, test_value) + @as(i32, test_scalar) },
            .i64 => dtypes.ScalarValue{ .i64 = @as(i64, test_value) + @as(i64, test_scalar) },
            .i16 => dtypes.ScalarValue{ .i16 = @as(i16, test_value) + @as(i16, test_scalar) },
            .i8 => dtypes.ScalarValue{ .i8 = @as(i8, test_value) + @as(i8, test_scalar) },
            .u32 => dtypes.ScalarValue{ .u32 = @as(u32, test_value) + @as(u32, test_scalar) },
            .u64 => dtypes.ScalarValue{ .u64 = @as(u64, test_value) + @as(u64, test_scalar) },
            .u16 => dtypes.ScalarValue{ .u16 = @as(u16, test_value) + @as(u16, test_scalar) },
            .u8 => dtypes.ScalarValue{ .u8 = @as(u8, test_value) + @as(u8, test_scalar) },
        };

        switch (dtype) {
            .f32 => try expect(result_val.f32 == expected_result.f32),
            .f64 => try expect(result_val.f64 == expected_result.f64),
            .i32 => try expect(result_val.i32 == expected_result.i32),
            .i64 => try expect(result_val.i64 == expected_result.i64),
            .i16 => try expect(result_val.i16 == expected_result.i16),
            .i8 => try expect(result_val.i8 == expected_result.i8),
            .u32 => try expect(result_val.u32 == expected_result.u32),
            .u64 => try expect(result_val.u64 == expected_result.u64),
            .u16 => try expect(result_val.u16 == expected_result.u16),
            .u8 => try expect(result_val.u8 == expected_result.u8),
        }

        // Test inplace operation
        const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
        defer a_copy.deinit();

        const inplace_result = try a_copy.add_scalar(test_scalar, true);
        defer if (inplace_result != a_copy) inplace_result.deinit();

        if (debug) {
            try inplace_result.info();
        }

        const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
        switch (dtype) {
            .f32 => try expect(inplace_val.f32 == expected_result.f32),
            .f64 => try expect(inplace_val.f64 == expected_result.f64),
            .i32 => try expect(inplace_val.i32 == expected_result.i32),
            .i64 => try expect(inplace_val.i64 == expected_result.i64),
            .i16 => try expect(inplace_val.i16 == expected_result.i16),
            .i8 => try expect(inplace_val.i8 == expected_result.i8),
            .u32 => try expect(inplace_val.u32 == expected_result.u32),
            .u64 => try expect(inplace_val.u64 == expected_result.u64),
            .u16 => try expect(inplace_val.u16 == expected_result.u16),
            .u8 => try expect(inplace_val.u8 == expected_result.u8),
        }

        // Ensure that the inplace operation modified the original array
        try expect(inplace_result == a_copy);
    }
}

test "ndarray.sub_scalar()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const test_value = 5;
    const test_scalar = 3;
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const shape = &[_]usize{ 5, 3 };
        const value = switch (dtype) {
            .f32 => @as(f32, test_value),
            .f64 => @as(f64, test_value),
            .i32 => @as(i32, test_value),
            .i64 => @as(i64, test_value),
            .i16 => @as(i16, test_value),
            .i8 => @as(i8, test_value),
            .u32 => @as(u32, test_value),
            .u64 => @as(u64, test_value),
            .u16 => @as(u16, test_value),
            .u8 => @as(u8, test_value),
        };

        const a = try NdArray.from_value(allocator, shape, dtype, value);
        defer a.deinit();

        // Test non-inplace operation
        const result = try a.sub_scalar(test_scalar, false);
        defer result.deinit();

        if (debug) {
            try result.info();
        }

        const result_val = try result.get(&[_]usize{ 0, 0 });
        const expected_result = switch (dtype) {
            .f32 => dtypes.ScalarValue{ .f32 = @as(f32, test_value) - @as(f32, test_scalar) },
            .f64 => dtypes.ScalarValue{ .f64 = @as(f64, test_value) - @as(f64, test_scalar) },
            .i32 => dtypes.ScalarValue{ .i32 = @as(i32, test_value) - @as(i32, test_scalar) },
            .i64 => dtypes.ScalarValue{ .i64 = @as(i64, test_value) - @as(i64, test_scalar) },
            .i16 => dtypes.ScalarValue{ .i16 = @as(i16, test_value) - @as(i16, test_scalar) },
            .i8 => dtypes.ScalarValue{ .i8 = @as(i8, test_value) - @as(i8, test_scalar) },
            .u32 => blk: {
                if (test_value >= test_scalar) {
                    break :blk dtypes.ScalarValue{ .u32 = @as(u32, test_value) - @as(u32, test_scalar) };
                } else {
                    std.debug.print("Warning: Skipping u32 subtraction test because test_value < test_scalar\n", .{});
                    break :blk dtypes.ScalarValue{ .u32 = 0 }; // Return a dummy value
                }
            },
            .u64 => blk: {
                if (test_value >= test_scalar) {
                    break :blk dtypes.ScalarValue{ .u64 = @as(u64, test_value) - @as(u64, test_scalar) };
                } else {
                    std.debug.print("Warning: Skipping u64 subtraction test because test_value < test_scalar\n", .{});
                    break :blk dtypes.ScalarValue{ .u64 = 0 }; // Return a dummy value
                }
            },
            .u16 => blk: {
                if (test_value >= test_scalar) {
                    break :blk dtypes.ScalarValue{ .u16 = @as(u16, test_value) - @as(u16, test_scalar) };
                } else {
                    std.debug.print("Warning: Skipping u16 subtraction test because test_value < test_scalar\n", .{});
                    break :blk dtypes.ScalarValue{ .u16 = 0 }; // Return a dummy value
                }
            },
            .u8 => blk: {
                if (test_value >= test_scalar) {
                    break :blk dtypes.ScalarValue{ .u8 = @as(u8, test_value) - @as(u8, test_scalar) };
                } else {
                    std.debug.print("Warning: Skipping u8 subtraction test because test_value < test_scalar\n", .{});
                    break :blk dtypes.ScalarValue{ .u8 = 0 }; // Return a dummy value
                }
            },
        };

        switch (dtype) {
            .f32 => try expect(result_val.f32 == expected_result.f32),
            .f64 => try expect(result_val.f64 == expected_result.f64),
            .i32 => try expect(result_val.i32 == expected_result.i32),
            .i64 => try expect(result_val.i64 == expected_result.i64),
            .i16 => try expect(result_val.i16 == expected_result.i16),
            .i8 => try expect(result_val.i8 == expected_result.i8),
            .u32 => try expect(result_val.u32 == expected_result.u32),
            .u64 => try expect(result_val.u64 == expected_result.u64),
            .u16 => try expect(result_val.u16 == expected_result.u16),
            .u8 => try expect(result_val.u8 == expected_result.u8),
        }

        // Test inplace operation
        const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
        defer a_copy.deinit();

        const inplace_result = try a_copy.sub_scalar(test_scalar, true);
        defer if (inplace_result != a_copy) inplace_result.deinit();

        if (debug) {
            try inplace_result.info();
        }

        const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
        switch (dtype) {
            .f32 => try expect(inplace_val.f32 == expected_result.f32),
            .f64 => try expect(inplace_val.f64 == expected_result.f64),
            .i32 => try expect(inplace_val.i32 == expected_result.i32),
            .i64 => try expect(inplace_val.i64 == expected_result.i64),
            .i16 => try expect(inplace_val.i16 == expected_result.i16),
            .i8 => try expect(inplace_val.i8 == expected_result.i8),
            .u32 => try expect(inplace_val.u32 == expected_result.u32),
            .u64 => try expect(inplace_val.u64 == expected_result.u64),
            .u16 => try expect(inplace_val.u16 == expected_result.u16),
            .u8 => try expect(inplace_val.u8 == expected_result.u8),
        }

        // Ensure that the inplace operation modified the original array
        try expect(inplace_result == a_copy);
    }
}

test "ndarray.mul_scalar(3.0)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const test_value = 5.0;
    const test_scalar = 3.0;
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const shape = &[_]usize{ 5, 3 };
        const value = switch (dtype) {
            .f32 => @as(f32, test_value),
            .f64 => @as(f64, test_value),
            .i32 => @as(i32, test_value),
            .i64 => @as(i64, test_value),
            .i16 => @as(i16, test_value),
            .i8 => @as(i8, test_value),
            .u32 => @as(u32, test_value),
            .u64 => @as(u64, test_value),
            .u16 => @as(u16, test_value),
            .u8 => @as(u8, test_value),
        };

        const a = try NdArray.from_value(allocator, shape, dtype, value);
        defer a.deinit();

        // Test non-inplace operation
        const result = try a.mul_scalar(test_scalar, false);
        defer result.deinit();

        if (debug) {
            try result.info();
        }

        const result_val = try result.get(&[_]usize{ 0, 0 });
        const expected_result = switch (dtype) {
            .f32 => dtypes.ScalarValue{ .f32 = @as(f32, test_value) * @as(f32, test_scalar) },
            .f64 => dtypes.ScalarValue{ .f64 = @as(f64, test_value) * @as(f64, test_scalar) },
            .i32 => dtypes.ScalarValue{ .i32 = @as(i32, test_value) * @as(i32, test_scalar) },
            .i64 => dtypes.ScalarValue{ .i64 = @as(i64, test_value) * @as(i64, test_scalar) },
            .i16 => dtypes.ScalarValue{ .i16 = @as(i16, test_value) * @as(i16, test_scalar) },
            .i8 => dtypes.ScalarValue{ .i8 = @as(i8, test_value) * @as(i8, test_scalar) },
            .u32 => dtypes.ScalarValue{ .u32 = @as(u32, test_value) * @as(u32, test_scalar) },
            .u64 => dtypes.ScalarValue{ .u64 = @as(u64, test_value) * @as(u64, test_scalar) },
            .u16 => dtypes.ScalarValue{ .u16 = @as(u16, test_value) * @as(u16, test_scalar) },
            .u8 => dtypes.ScalarValue{ .u8 = @as(u8, test_value) * @as(u8, test_scalar) },
        };

        switch (dtype) {
            .f32 => try expect(result_val.f32 == expected_result.f32),
            .f64 => try expect(result_val.f64 == expected_result.f64),
            .i32 => try expect(result_val.i32 == expected_result.i32),
            .i64 => try expect(result_val.i64 == expected_result.i64),
            .i16 => try expect(result_val.i16 == expected_result.i16),
            .i8 => try expect(result_val.i8 == expected_result.i8),
            .u32 => try expect(result_val.u32 == expected_result.u32),
            .u64 => try expect(result_val.u64 == expected_result.u64),
            .u16 => try expect(result_val.u16 == expected_result.u16),
            .u8 => try expect(result_val.u8 == expected_result.u8),
        }

        // Test inplace operation
        const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
        defer a_copy.deinit();

        const inplace_result = try a_copy.mul_scalar(test_scalar, true);
        defer if (inplace_result != a_copy) inplace_result.deinit();

        if (debug) {
            try inplace_result.info();
        }

        const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
        switch (dtype) {
            .f32 => try expect(inplace_val.f32 == expected_result.f32),
            .f64 => try expect(inplace_val.f64 == expected_result.f64),
            .i32 => try expect(inplace_val.i32 == expected_result.i32),
            .i64 => try expect(inplace_val.i64 == expected_result.i64),
            .i16 => try expect(inplace_val.i16 == expected_result.i16),
            .i8 => try expect(inplace_val.i8 == expected_result.i8),
            .u32 => try expect(inplace_val.u32 == expected_result.u32),
            .u64 => try expect(inplace_val.u64 == expected_result.u64),
            .u16 => try expect(inplace_val.u16 == expected_result.u16),
            .u8 => try expect(inplace_val.u8 == expected_result.u8),
        }

        // Ensure that the inplace operation modified the original array
        try expect(inplace_result == a_copy);
    }
}

test "ndarray.mul_scalar(0.005)" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const test_value = 5.0;
    const test_scalar = 0.005;
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const shape = &[_]usize{ 5, 3 };
        const value = switch (dtype) {
            .f32 => @as(f32, test_value),
            .f64 => @as(f64, test_value),
            .i32 => @as(i32, test_value),
            .i64 => @as(i64, test_value),
            .i16 => @as(i16, test_value),
            .i8 => @as(i8, test_value),
            .u32 => @as(u32, test_value),
            .u64 => @as(u64, test_value),
            .u16 => @as(u16, test_value),
            .u8 => @as(u8, test_value),
        };

        const a = try NdArray.from_value(allocator, shape, dtype, value);
        defer a.deinit();

        // Test non-inplace operation
        const result = try a.mul_scalar(test_scalar, false);
        defer result.deinit();

        if (debug) {
            try result.info();
        }

        const result_val = try result.get(&[_]usize{ 0, 0 });
        const expected_result = switch (dtype) {
            .f32 => dtypes.ScalarValue{ .f32 = @as(f32, test_value) * @as(f32, test_scalar) },
            .f64 => dtypes.ScalarValue{ .f64 = @as(f64, test_value) * @as(f64, test_scalar) },
            .i32 => dtypes.ScalarValue{ .i32 = @as(i32, test_value) * @as(i32, @intFromFloat(test_scalar)) },
            .i64 => dtypes.ScalarValue{ .i64 = @as(i64, test_value) * @as(i64, @intFromFloat(test_scalar)) },
            .i16 => dtypes.ScalarValue{ .i16 = @as(i16, test_value) * @as(i16, @intFromFloat(test_scalar)) },
            .i8 => dtypes.ScalarValue{ .i8 = @as(i8, test_value) * @as(i8, @intFromFloat(test_scalar)) },
            .u32 => dtypes.ScalarValue{ .u32 = @as(u32, test_value) * @as(u32, @intFromFloat(test_scalar)) },
            .u64 => dtypes.ScalarValue{ .u64 = @as(u64, test_value) * @as(u64, @intFromFloat(test_scalar)) },
            .u16 => dtypes.ScalarValue{ .u16 = @as(u16, test_value) * @as(u16, @intFromFloat(test_scalar)) },
            .u8 => dtypes.ScalarValue{ .u8 = @as(u8, test_value) * @as(u8, @intFromFloat(test_scalar)) },
        };

        switch (dtype) {
            .f32 => try expect(result_val.f32 == expected_result.f32),
            .f64 => try expect(result_val.f64 == expected_result.f64),
            .i32 => try expect(result_val.i32 == expected_result.i32),
            .i64 => try expect(result_val.i64 == expected_result.i64),
            .i16 => try expect(result_val.i16 == expected_result.i16),
            .i8 => try expect(result_val.i8 == expected_result.i8),
            .u32 => try expect(result_val.u32 == expected_result.u32),
            .u64 => try expect(result_val.u64 == expected_result.u64),
            .u16 => try expect(result_val.u16 == expected_result.u16),
            .u8 => try expect(result_val.u8 == expected_result.u8),
        }

        // Test inplace operation
        const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
        defer a_copy.deinit();

        const inplace_result = try a_copy.mul_scalar(test_scalar, true);
        defer if (inplace_result != a_copy) inplace_result.deinit();

        if (debug) {
            try inplace_result.info();
        }

        const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
        switch (dtype) {
            .f32 => try expect(inplace_val.f32 == expected_result.f32),
            .f64 => try expect(inplace_val.f64 == expected_result.f64),
            .i32 => try expect(inplace_val.i32 == expected_result.i32),
            .i64 => try expect(inplace_val.i64 == expected_result.i64),
            .i16 => try expect(inplace_val.i16 == expected_result.i16),
            .i8 => try expect(inplace_val.i8 == expected_result.i8),
            .u32 => try expect(inplace_val.u32 == expected_result.u32),
            .u64 => try expect(inplace_val.u64 == expected_result.u64),
            .u16 => try expect(inplace_val.u16 == expected_result.u16),
            .u8 => try expect(inplace_val.u8 == expected_result.u8),
        }

        // Ensure that the inplace operation modified the original array
        try expect(inplace_result == a_copy);
    }
}

test "ndarray.div_scalar()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const test_value = 5;
    const test_scalar = 3;
    const data_types = &[_]dtypes.DataType{
        .f32, .f64, .i32, .i64, .i16, .i8, .u32, .u64, .u16, .u8,
    };

    for (data_types) |dtype| {
        const shape = &[_]usize{ 5, 3 };
        const value = switch (dtype) {
            .f32 => @as(f32, test_value),
            .f64 => @as(f64, test_value),
            .i32 => @as(i32, test_value),
            .i64 => @as(i64, test_value),
            .i16 => @as(i16, test_value),
            .i8 => @as(i8, test_value),
            .u32 => @as(u32, test_value),
            .u64 => @as(u64, test_value),
            .u16 => @as(u16, test_value),
            .u8 => @as(u8, test_value),
        };

        const a = try NdArray.from_value(allocator, shape, dtype, value);
        defer a.deinit();

        // Test non-inplace operation
        const result = try a.div_scalar(test_scalar, false);
        defer result.deinit();

        if (debug) {
            try result.info();
        }

        const result_val = try result.get(&[_]usize{ 0, 0 });
        const expected_result = switch (dtype) {
            .f32 => dtypes.ScalarValue{ .f32 = @as(f32, test_value) / @as(f32, test_scalar) },
            .f64 => dtypes.ScalarValue{ .f64 = @as(f64, test_value) / @as(f64, test_scalar) },
            .i32 => dtypes.ScalarValue{ .i32 = @divTrunc(@as(i32, test_value), @as(i32, test_scalar)) },
            .i64 => dtypes.ScalarValue{ .i64 = @divTrunc(@as(i64, test_value), @as(i64, test_scalar)) },
            .i16 => dtypes.ScalarValue{ .i16 = @divTrunc(@as(i16, test_value), @as(i16, test_scalar)) },
            .i8 => dtypes.ScalarValue{ .i8 = @divTrunc(@as(i8, test_value), @as(i8, test_scalar)) },
            .u32 => dtypes.ScalarValue{ .u32 = @as(u32, test_value) / @as(u32, test_scalar) },
            .u64 => dtypes.ScalarValue{ .u64 = @as(u64, test_value) / @as(u64, test_scalar) },
            .u16 => dtypes.ScalarValue{ .u16 = @as(u16, test_value) / @as(u16, test_scalar) },
            .u8 => dtypes.ScalarValue{ .u8 = @as(u8, test_value) / @as(u8, test_scalar) },
        };

        switch (dtype) {
            .f32 => try expect(result_val.f32 == expected_result.f32),
            .f64 => try expect(result_val.f64 == expected_result.f64),
            .i32 => try expect(result_val.i32 == expected_result.i32),
            .i64 => try expect(result_val.i64 == expected_result.i64),
            .i16 => try expect(result_val.i16 == expected_result.i16),
            .i8 => try expect(result_val.i8 == expected_result.i8),
            .u32 => try expect(result_val.u32 == expected_result.u32),
            .u64 => try expect(result_val.u64 == expected_result.u64),
            .u16 => try expect(result_val.u16 == expected_result.u16),
            .u8 => try expect(result_val.u8 == expected_result.u8),
        }

        // Test inplace operation
        const a_copy = try NdArray.from_value(allocator, shape, dtype, test_value);
        defer a_copy.deinit();

        const inplace_result = try a_copy.div_scalar(test_scalar, true);
        defer if (inplace_result != a_copy) inplace_result.deinit();

        if (debug) {
            try inplace_result.info();
        }

        const inplace_val = try inplace_result.get(&[_]usize{ 0, 0 });
        switch (dtype) {
            .f32 => try expect(inplace_val.f32 == expected_result.f32),
            .f64 => try expect(inplace_val.f64 == expected_result.f64),
            .i32 => try expect(inplace_val.i32 == expected_result.i32),
            .i64 => try expect(inplace_val.i64 == expected_result.i64),
            .i16 => try expect(inplace_val.i16 == expected_result.i16),
            .i8 => try expect(inplace_val.i8 == expected_result.i8),
            .u32 => try expect(inplace_val.u32 == expected_result.u32),
            .u64 => try expect(inplace_val.u64 == expected_result.u64),
            .u16 => try expect(inplace_val.u16 == expected_result.u16),
            .u8 => try expect(inplace_val.u8 == expected_result.u8),
        }

        // Ensure that the inplace operation modified the original array
        try expect(inplace_result == a_copy);
    }
}

test "ndarray.matmul()" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const debug = false;
    const a_shape = &[_]usize{ 3, 2 };
    const a = try NdArray.from_value(allocator, a_shape, .f32, @as(f32, 1.0));
    defer a.deinit();

    const b_shape = &[_]usize{ 2, 4 };
    const b = try NdArray.from_value(allocator, b_shape, .f32, @as(f32, 2.0));
    defer b.deinit();

    const matmul_result = try a.matmul(b, 1.0, 1.0, false, false);
    defer matmul_result.deinit();

    if (debug) {
        try a.print();
        try b.print();
        try matmul_result.print();
    }

    const matmul_val = try matmul_result.get(&[_]usize{ 0, 0 });
    try expect(matmul_val.f32 == 4.0);
}

test "NdArray.cast: convert f32 to i32" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarray
    const data = [_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
    defer ndarray.deinit();

    // Cast the ndarray to i32
    const casted = try ndarray.cast(.i32);
    defer casted.deinit();

    // Expected result after casting to i32 (truncates floating-point values)
    const expected_data = &[_]i32{ 1, 2, 3, 4, 5, 6 };

    // Verify the result using a loop
    const total_size = utils.compute_size(shape);
    for (0..total_size) |i| {
        const row = i / shape[1]; // Calculate the row index
        const col = i % shape[1]; // Calculate the column index
        const casted_value = try casted.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.i32);
    }
}

test "NdArray.cast: convert f32 to f64" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarray
    const data = [_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
    defer ndarray.deinit();

    // Cast the ndarray to f64
    const casted = try ndarray.cast(.f64);
    defer casted.deinit();

    // Expected result after casting to f64
    const expected_data = &[_]f64{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };

    // Verify the result using a loop
    const total_size = utils.compute_size(shape);
    for (0..total_size) |i| {
        const row = i / shape[1]; // Calculate the row index
        const col = i % shape[1]; // Calculate the column index
        const casted_value = try casted.get(&[_]usize{ row, col });

        // Use approxEqAbs to compare floating-point values with a tolerance
        const tolerance = 1e-6; // Adjust the tolerance as needed
        const is_equal = std.math.approxEqAbs(f64, expected_data[i], casted_value.f64, tolerance);

        // Log the values for debugging
        // std.debug.print("Expected: {d}, Got: {d}, Tolerance: {d}\n", .{ expected_data[i], casted_value.f64, tolerance });

        // Check if the values are approximately equal
        try std.testing.expect(is_equal);
    }
}

test "NdArray.cast: convert i32 to i64" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarray
    const data = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const shape = &[_]usize{ 2, 3 };

    // Create the ndarray from data
    const ndarray = try NdArray.from_data(allocator, shape, .i32, &data);
    defer ndarray.deinit();

    // Cast the ndarray to i64
    const casted = try ndarray.cast(.i64);
    defer casted.deinit();

    // Expected result after casting to i64
    const expected_data = &[_]i64{ 1, 2, 3, 4, 5, 6 };

    // Verify the result using a loop
    const total_size = utils.compute_size(shape);
    for (0..total_size) |i| {
        const row = i / shape[1]; // Calculate the row index
        const col = i % shape[1]; // Calculate the column index
        const casted_value = try casted.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.i64);
    }
}

test "NdArray.cast: convert f64 to f32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = [_]f64{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
    const shape = &[_]usize{ 2, 3 };

    const ndarray = try NdArray.from_data(allocator, shape, .f64, &data);
    defer ndarray.deinit();

    const casted = try ndarray.cast(.f32);
    defer casted.deinit();

    const expected_data = &[_]f32{ 1.5, 2.7, 3.2, 4.8, 5.1, 6.9 };
    const total_size = utils.compute_size(shape);

    for (0..total_size) |i| {
        const row = i / shape[1];
        const col = i % shape[1];
        const casted_value = try casted.get(&[_]usize{ row, col });
        const tolerance = 1e-6;
        const is_equal = std.math.approxEqAbs(f32, expected_data[i], casted_value.f32, tolerance);
        try std.testing.expect(is_equal);
    }
}

test "NdArray.cast: convert i32 to f64" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = [_]i32{ 1, -2, 3, 4, -5, 6 };
    const shape = &[_]usize{ 2, 3 };

    const ndarray = try NdArray.from_data(allocator, shape, .i32, &data);
    defer ndarray.deinit();

    const casted = try ndarray.cast(.f64);
    defer casted.deinit();

    const expected_data = &[_]f64{ 1.0, -2.0, 3.0, 4.0, -5.0, 6.0 };
    const total_size = utils.compute_size(shape);

    for (0..total_size) |i| {
        const row = i / shape[1];
        const col = i % shape[1];
        const casted_value = try casted.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.f64);
    }
}

test "NdArray.slice: slice rows" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };

    const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
    defer ndarray.deinit();

    const sliced = try ndarray.slice(0, 1, 3); // Slice rows 1 to 3
    defer sliced.deinit();

    const expected_data = &[_]f32{ 4, 5, 6, 7, 8, 9 };
    const sliced_shape = &[_]usize{ 2, 3 };
    const total_size = utils.compute_size(sliced_shape);

    for (0..total_size) |i| {
        const row = i / sliced_shape[1];
        const col = i % sliced_shape[1];
        const casted_value = try sliced.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.f32);
    }
}

test "NdArray.slice: slice columns" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const shape = &[_]usize{ 4, 3 };

    const ndarray = try NdArray.from_data(allocator, shape, .f32, &data);
    defer ndarray.deinit();

    const sliced = try ndarray.slice(1, 1, 2); // Slice columns 1 to 2
    defer sliced.deinit();

    const expected_data = &[_]f32{ 2, 5, 8, 11 };
    const sliced_shape = &[_]usize{ 4, 1 };
    const total_size = utils.compute_size(sliced_shape);

    for (0..total_size) |i| {
        const row = i / sliced_shape[1];
        const col = i % sliced_shape[1];
        const casted_value = try sliced.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.f32);
    }
}

test "NdArray.equal: element-wise equality comparison" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarrays
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const data_b = [_]f32{ 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0 };
    const shape = &[_]usize{ 4, 3 };

    // Create the ndarrays from data
    const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
    const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
    defer ndarray_a.deinit();
    defer ndarray_b.deinit();

    // Perform the equality comparison
    const result = try ndarray_a.equal(ndarray_b, false);
    defer result.deinit();

    // Expected result (1 where elements are equal, 0 otherwise)
    const expected_data = &[_]i32{ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 };
    const total_size = utils.compute_size(shape);

    for (0..total_size) |i| {
        const row = i / result.shape[1];
        const col = i % result.shape[1];
        const casted_value = try result.get(&[_]usize{ row, col });
        try std.testing.expectEqual(expected_data[i], casted_value.i32);
    }
}

test "NdArray.greater_than: element-wise greater than comparison" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarrays
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 6, 0, 11 };
    const data_b = [_]f32{ 1, 1, 4, 3, 6, 5, 1, 9 };
    const shape = &[_]usize{ 4, 2 };

    // Create the ndarrays from data
    const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
    const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
    defer ndarray_a.deinit();
    defer ndarray_b.deinit();

    // Perform the greater-than comparison
    const result = try ndarray_a.greater_than(ndarray_b, false);
    defer result.deinit();

    // Expected result (1 where elements in A are greater than B, 0 otherwise)
    const expected_data = &[_]i32{
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
        1, 1, 1,
    };

    const total_size = utils.compute_size(result.shape);
    for (0..total_size) |i| {
        const row = i / result.shape[1];
        const col = i % result.shape[1];
        const casted_value = try result.get(&[_]usize{ row, col });
        // std.debug.print("{} ? {} \n", .{ expected_data[i], casted_value.i32 });
        try std.testing.expectEqual(expected_data[i], casted_value.i32);
    }
}

test "NdArray.less_than: element-wise less than comparison" {
    // Initialize an allocator for the test
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Original data for the ndarrays
    const data_a = [_]f32{ 1, 2, 3, 4, 5, 8 };
    const data_b = [_]f32{ 1, 3, 2, 5, 8, 8.5 };
    const shape = &[_]usize{ 3, 2 };

    // Create the ndarrays from data
    const ndarray_a = try NdArray.from_data(allocator, shape, .f32, &data_a);
    const ndarray_b = try NdArray.from_data(allocator, shape, .f32, &data_b);
    defer ndarray_a.deinit();
    defer ndarray_b.deinit();

    // Perform the less-than comparison
    const result = try ndarray_a.less_than(ndarray_b, false);
    defer result.deinit();

    // Expected result (1 where elements in A are less than B, 0 otherwise)
    const expected_data = &[_]i32{ 0, 1, 0, 1, 1, 1, 1 };

    const total_size = utils.compute_size(result.shape);
    for (0..total_size) |i| {
        const row = i / result.shape[1];
        const col = i % result.shape[1];
        const casted_value = try result.get(&[_]usize{ row, col });
        // std.debug.print("{} ? {} \n", .{ expected_data[i], casted_value.i32 });
        try std.testing.expectEqual(expected_data[i], casted_value.i32);
    }
}
