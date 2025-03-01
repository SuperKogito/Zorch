const std = @import("std");
const dtypes = @import("dtypes.zig");
const NdArray = @import("ndarray.zig").NdArray;

pub const DataType = dtypes.DataType;
pub const NumericUnion = dtypes.NumericUnion;

pub fn bytes_to_val(self: *const NdArray, byte_offset: usize) NumericUnion {
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

pub fn val_to_bytes(self: *NdArray, byte_offset: usize, value: NumericUnion) void {
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
