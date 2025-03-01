const std = @import("std");

// Define a union type to hold any TEST_SCALAR TEST_VALUE
pub const ScalarValue = union(enum) {
    f32: f32,
    f64: f64,
    i32: i32,
    i64: i64,
    i16: i16,
    i8: i8,
    u32: u32,
    u64: u64,
    u16: u16,
    u8: u8,
};

// Define supported data types
pub const DType = enum {
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,

    // Convert DType to DataType (if needed)
    pub fn toDataType(self: DType) DataType {
        return switch (self) {
            .f32 => DataType.f32,
            .f64 => DataType.f64,
            .i8 => DataType.i8,
            .i16 => DataType.i16,
            .i32 => DataType.i32,
            .i64 => DataType.i64,
            .u8 => DataType.u8,
            .u16 => DataType.u16,
            .u32 => DataType.u32,
            .u64 => DataType.u64,
        };
    }
};

// Define a simple DataType enum
pub const DataType = enum {
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,

    // Get the size of the data type in bytes
    pub fn sizeInBytes(self: DataType) usize {
        return switch (self) {
            .f32 => @sizeOf(f32),
            .f64 => @sizeOf(f64),
            .i8 => @sizeOf(i8),
            .i16 => @sizeOf(i16),
            .i32 => @sizeOf(i32),
            .i64 => @sizeOf(i64),
            .u8 => @sizeOf(u8),
            .u16 => @sizeOf(u16),
            .u32 => @sizeOf(u32),
            .u64 => @sizeOf(u64),
        };
    }

    // Convert DataType to DType (if needed)
    pub fn toDType(self: DataType) DType {
        return switch (self) {
            .f32 => DType.f32,
            .f64 => DType.f64,
            .i8 => DType.i8,
            .i16 => DType.i16,
            .i32 => DType.i32,
            .i64 => DType.i64,
            .u8 => DType.u8,
            .u16 => DType.u16,
            .u32 => DType.u32,
            .u64 => DType.u64,
        };
    }
};

// Union to store different numeric types
pub const NumericUnion = union(DataType) {
    f32: f32,
    f64: f64,
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
};

/// Converts a NumericUnion to a formatted string based on the dtype.
pub fn format_numeric_value(num: NumericUnion, dtype: DataType, allocator: std.mem.Allocator) ![]const u8 {
    return switch (dtype) {
        .f32 => std.fmt.allocPrint(allocator, "{d:.3}", .{num.f32}),
        .f64 => std.fmt.allocPrint(allocator, "{d:.6}", .{num.f64}),
        .i8 => std.fmt.allocPrint(allocator, "{d}", .{num.i8}),
        .i16 => std.fmt.allocPrint(allocator, "{d}", .{num.i16}),
        .i32 => std.fmt.allocPrint(allocator, "{d}", .{num.i32}),
        .i64 => std.fmt.allocPrint(allocator, "{d}", .{num.i64}),
        .u8 => std.fmt.allocPrint(allocator, "{d}", .{num.u8}),
        .u16 => std.fmt.allocPrint(allocator, "{d}", .{num.u16}),
        .u32 => std.fmt.allocPrint(allocator, "{d}", .{num.u32}),
        .u64 => std.fmt.allocPrint(allocator, "{d}", .{num.u64}),
    };
}
