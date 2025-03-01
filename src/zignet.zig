pub const tensor = @import("tensor.zig");

// Create a math namespace
pub const math = struct {
    pub const ndarray = struct {
        pub const dtypes = @import("math/ndarray/dtypes.zig");
        pub const ndarray = @import("math/ndarray/ndarray.zig");
    };
};

// Create a math namespace
pub const nn = struct {
    pub const layers = @import("nn/layers.zig");
    pub const losses = @import("nn/losses.zig");
    pub const optimizers = @import("nn/optimizers.zig");
    pub const sequential = @import("nn/sequential.zig");
};

// Direct exports for easier access
pub const dtypes = math.ndarray.dtypes;
pub const ndarray = math.ndarray.ndarray;
pub const DType = dtypes.DType;
pub const NdArray = ndarray.NdArray;
