const std = @import("std");
const zorch = @import("zorch.zig");

const utils = zorch.utils;
const dtypes = zorch.dtypes;
const logger = zorch.logger;

const Tensor = zorch.Tensor;
const TensorError = zorch.errors.TensorError;
const NdArray = zorch.NdArray;
const NdarrayError = zorch.errors.NdarrayError;
const DEBUG = false;

// Operation interface
pub const Operation = struct {
    notation: []const u8,
    forward: *const fn (allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor,
    backward: *const fn (tensor: *const Tensor, grad: *NdArray) TensorError!void,
};

// Helper function to initialize an operation with an empty cache
fn init_op(notation: []const u8, forward: anytype, backward: anytype) Operation {
    return Operation{
        .notation = notation,
        .forward = forward,
        .backward = backward,
    };
}

fn update_grad(existing_grad: *NdArray, backpropagated_grad: *NdArray) NdarrayError!void {
    // std.debug.print("\n--- Starting update_grad ---\n", .{});
    // std.debug.print("Existing gradient shape: {any}\n", .{existing_grad.shape});
    // std.debug.print("Backpropagated gradient shape: {any}\n", .{backpropagated_grad.shape});

    const target_shape = existing_grad.shape;
    var working_grad = try backpropagated_grad.clone();
    errdefer working_grad.deinit();

    // std.debug.print("Cloned working_grad shape: {any}\n", .{working_grad.shape});

    // Handle excess dimensions by summing
    while (working_grad.shape.len > target_shape.len) {
        // std.debug.print("Excess dimensions detected. Summing along axis 0...\n", .{});
        const summed = try working_grad.sum(0, true);
        working_grad.deinit();
        working_grad = summed;
        // std.debug.print("After summing, working_grad shape: {any}\n", .{working_grad.shape});
        try working_grad.print();
    }

    // Handle mismatched dimensions by summing
    for (0..target_shape.len) |axis| {
        if (working_grad.shape[axis] != target_shape[axis]) {
            // std.debug.print("Mismatch detected on axis {}. Working shape: {}, Target shape: {}\n", .{ axis, working_grad.shape[axis], target_shape[axis] });
            if (working_grad.shape[axis] % target_shape[axis] != 0) {
                // std.debug.print("Incompatible shapes: {} is not divisible by {}\n", .{ working_grad.shape[axis], target_shape[axis] });
                return error.IncompatibleShapes;
            }
            // std.debug.print("Summing along axis {}...\n", .{axis});
            const summed = try working_grad.sum(axis, true);
            working_grad.deinit();
            working_grad = summed;
            // std.debug.print("After summing, working_grad shape: {any}\n", .{working_grad.shape});
            try working_grad.print();
        }
    }

    // Broadcast to target shape if needed
    if (!utils.is_compatible_for_broadcast(working_grad.shape, target_shape)) {
        // std.debug.print("Broadcasting working_grad to target shape...\n", .{});
        const broadcasted = try working_grad.broadcast_to(target_shape);
        working_grad.deinit();
        working_grad = broadcasted;
        // std.debug.print("After broadcasting, working_grad shape: {any}\n", .{working_grad.shape});
        try working_grad.print();
    }

    // std.debug.print("       Backpropagated grad  : ", .{});
    // try backpropagated_grad.print();
    // std.debug.print("       Previous operations grad  : ", .{});
    // try existing_grad.print();
    // std.debug.print("       Current operation grad    : ", .{});
    // try working_grad.print();

    // Accumulate the gradient instead of overwriting
    // std.debug.print("Adding working_grad to existing_grad...\n", .{});
    _ = try existing_grad.add(working_grad, true);

    // Deinitialize working_grad after successful addition
    working_grad.deinit();

    // std.debug.print("--- update_grad completed successfully ---\n", .{});
}

// Available operations
pub const mse_op = init_op("Mse", &MseOp.forward, &MseOp.backward);
pub const add_op = init_op("Add", &AddOp.forward, &AddOp.backward);
pub const mul_op = init_op("Mul", &MulOp.forward, &MulOp.backward);
pub const sub_op = init_op("Sub", &SubOp.forward, &SubOp.backward);
pub const div_op = init_op("Div", &DivOp.forward, &DivOp.backward);
pub const transpose_op = init_op("Transpose", &TransposeOp.forward, &TransposeOp.backward);
pub const tanh_op = init_op("Tanh", &TanhOp.forward, &TanhOp.backward);
pub const relu_op = init_op("Relu", &ReLUOp.forward, &ReLUOp.backward);
pub const sigmoid_op = init_op("Sigmoid", &SigmoidOp.forward, &SigmoidOp.backward);
pub const pow_op = init_op("Pow", &PowOp.forward, &PowOp.backward);
pub const gemm_op = init_op("Gemm", &GemmOp.forward, &GemmOp.backward);
pub const softmax_op = init_op("Softmax", &SoftmaxOp.forward, &SoftmaxOp.backward);
// pub const min_op = init_op("Min", &MinOp.forward, &MinOp.backward);
// pub const max_op = init_op("Max", &MaxOp.forward, &MaxOp.backward);
// pub const mean_op = init_op("Mean", &MeanOp.forward, &MeanOp.backward);
pub const clone_op = init_op("Mean", &CloneOp.forward, &CloneOp.backward);

// MSE Operation
pub const MseOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        if (DEBUG) {
            std.debug.print("* ENTER MSE. FORWARD \n", .{});
        }

        const pred = params[0];
        const target = params[1];

        if (DEBUG) {
            std.debug.print("│    y: ", .{});
            try pred.info();

            std.debug.print("│    ŷ: ", .{});
            try target.info();
        }

        const diff = try pred.data.sub(target.data, false);
        defer diff.deinit();

        const squared_diff = try diff.mul(diff, false);
        defer squared_diff.deinit();
        const mse = try squared_diff.mean(null, false);

        if (DEBUG) {
            std.debug.print("│    mse(y, ŷ): ", .{});
            try mse.info();
            std.debug.print("└── EXIT MSE. FORWARD\n\n", .{});
        }

        return try Tensor.init(
            allocator,
            mse.shape,
            mse.dtype,
            mse,
            &[_]*Tensor{ pred, target },
            @as(?*Operation, @constCast(&mse_op)),
            "mse-res",
            pred.requires_grad or target.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        if (DEBUG) {
            std.debug.print("\n+ ENTER MSE. BACKWARD \n", .{});
        }

        const pred = tensor.parents[0];
        const target = tensor.parents[1];

        if (pred.grad) |x_grad| {
            const n = utils.compute_size(pred.shape);
            const diff = try pred.data.sub(target.data, false); // pred - target
            defer diff.deinit();

            const pred_grad = try diff.mul_scalar(2.0 / @as(f32, @floatFromInt(n)), false); // 2 * (pred - target) / n
            defer pred_grad.deinit();

            const current_grad = try pred_grad.mul(grad, false); // 2 * (pred - target) / n
            defer current_grad.deinit();

            std.debug.print("│    Child Grad: ", .{});
            try grad.info();

            if (DEBUG) {
                std.debug.print("│    Backprop. Grad: ", .{});
                try current_grad.info();

                std.debug.print("│    Pred.     Grad: ", .{});
                try x_grad.info();
            }

            try update_grad(x_grad, current_grad);
            if (DEBUG) {
                std.debug.print("│    Final     Grad: ", .{});
                try x_grad.info();
            }
        }
        if (target.grad) |y_grad| {
            std.debug.print("Warning: Gradient is not implemented for target tensor in MSE operation.", .{});
            _ = y_grad;
        }
        if (DEBUG) {
            std.debug.print("└── EXIT MSE. BACKWARD \n", .{});
        }
    }
};

// Add Operation
pub const AddOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.add(y.data, false);
        errdefer z.deinit();

        // Log messages
        logger.info("main", "This is an info message with args: {d}", .{42});
        logger.err("main", "This is an error message", .{});
        logger.warn("main", "This is a warning message", .{});
        logger.debug("main", "This is a debug message", .{});

        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&add_op)),
            "add-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        // std.debug.print("\n+── ENTER ADD. BACKWARD \n", .{});

        const x = tensor.parents[0];
        const y = tensor.parents[1];

        // std.debug.print("│    Child Grad: ", .{});
        try grad.info();
        // Update gradient for x
        if (x.grad) |x_grad| {
            // std.debug.print("\n│    Before Update [Add] dE/dX = ", .{});
            try x_grad.print();
            try update_grad(x_grad, grad);
            // std.debug.print("│    After Update [Add] dE/dX = ", .{});
            try x_grad.print();
        }

        // Update gradient for y
        if (y.grad) |y_grad| {
            // std.debug.print("\n│    Before Update [Add] dE/dY = ", .{});
            try y_grad.print();
            try update_grad(y_grad, grad);
            // std.debug.print("│    After Update [Add] dE/dY = ", .{});
            try y_grad.print();
        }
        // std.debug.print("\n└── EXIT ADD. BACKWARD \n", .{});
    }
};

// Multiply Operation
pub const MulOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.mul(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&mul_op)),
            "mul-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        // std.debug.print("+── ENTER MUL. BACKWARD \n", .{});

        const x = tensor.parents[0];
        const y = tensor.parents[1];

        // std.debug.print("│    Child Grad: ", .{});
        try grad.info();

        // Compute gradient for x
        if (x.requires_grad) {
            const grad_x = try y.data.mul(grad, false);
            defer grad_x.deinit();

            // std.debug.print("\n│    Before Update [Mul] dE/dX = ", .{});
            // try x.grad.?.info();
            // try grad_x.info();
            // try y.info();

            try update_grad(x.grad.?, grad_x);

            // std.debug.print("│    After Update [Mul] dE/dX = ", .{});
            //vtry x.grad.?.print();
        }

        // Compute gradient for y
        if (y.requires_grad) {
            const grad_y = try x.data.mul(grad, false);
            defer grad_y.deinit();

            // std.debug.print("\n│    Before Update [Mul] dE/dY = ", .{});
            //try y.grad.?.print();

            try update_grad(y.grad.?, grad_y);

            // std.debug.print("│    After Update [Mul] dE/dY = ", .{});
            //try y.grad.?.print();
        }

        // std.debug.print("\n└── EXIT MUL. BACKWARD \n", .{});
    }
};

// Subtract Operation
pub const SubOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.sub(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&sub_op)),
            "sub-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];
        if (x.grad) |x_grad| {
            try update_grad(x_grad, grad);
        }
        if (y.grad) |y_grad| {
            const neg_grad = try grad.neg(false);
            defer neg_grad.deinit();

            try update_grad(y_grad, neg_grad);
        }
    }
};

// Divide Operation
pub const DivOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.div(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&div_op)),
            "div-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];

        if (x.grad) |x_grad| {
            const grad_x = try grad.div(y.data, false);
            defer grad_x.deinit();

            try update_grad(x_grad, grad_x);
        }
        if (y.grad) |y_grad| {
            const y_squared = try y.data.mul(y.data, false);
            defer y_squared.deinit();

            const grad_y = try x.data.mul(grad, false);
            defer grad_y.deinit();

            const neg_grad_y = try grad_y.neg(false);
            defer neg_grad_y.deinit();

            const final_grad_y = try neg_grad_y.div(y_squared, false);
            defer final_grad_y.deinit();

            try update_grad(y_grad, final_grad_y);
        }
    }
};

// Transpose Operation
pub const TransposeOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.transpose();
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&transpose_op)),
            "transpose-res",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        if (x.grad) |x_grad| {
            const grad_t = try grad.transpose();
            defer grad_t.deinit();

            try update_grad(x_grad, grad_t);
        }
    }
};

// ReLU Operation
pub const ReLUOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const z = try x.data.relu(false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&relu_op)),
            "relu-res",
            x.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        if (x.grad) |x_grad| {
            const mask = try x.data.greater_than_scalar(0.0, false);
            defer mask.deinit();

            const grad_relu = try grad.mul(mask, false);
            defer grad_relu.deinit();

            try update_grad(x_grad, grad_relu);
        }
    }
};

// Tanh Operation
pub const TanhOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.tanh(false);

        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&tanh_op)),
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        // Accumulate the gradient in x.grad
        if (x.grad) |x_grad| {
            // Compute tanh(x) during the backward pass
            const y_data = try x.data.tanh(false);
            errdefer y_data.deinit();

            // Compute 1 - tanh(x)^2
            const z_squared = try y_data.mul(y_data, false);
            errdefer z_squared.deinit();

            const neg_z_squared = try z_squared.neg(false);
            errdefer neg_z_squared.deinit();

            const one_minus_z_squared = try neg_z_squared.add_scalar(1.0, true);
            errdefer one_minus_z_squared.deinit();

            // Compute the gradient: grad * (1 - tanh(x)^2)
            const grad_tanh = try grad.mul(one_minus_z_squared, false);
            errdefer grad_tanh.deinit();

            try update_grad(x_grad, grad_tanh);
        }
    }
};

// Sigmoid Operation
pub const SigmoidOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y_data = try x.data.sigmoid(false);
        const y = try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&sigmoid_op)),
            "",
            x.requires_grad,
        );
        return y;
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        if (x.grad) |x_grad| {
            const z = try x.data.sigmoid(false); // z = sigmoid(x)
            defer z.deinit();

            const neg_z = try z.neg(false); // neg_z = -z
            defer neg_z.deinit();

            const one_minus_z = try neg_z.add_scalar(1.0, false); // one_minus_z = 1 - z
            defer one_minus_z.deinit();

            const z_times_one_minus_z = try z.mul(one_minus_z, false); // z * (1 - z)
            defer z_times_one_minus_z.deinit();

            const grad_sigmoid = try grad.mul(z_times_one_minus_z, false); // grad * z * (1 - z)
            defer grad_sigmoid.deinit();

            try update_grad(x_grad, grad_sigmoid); // Accumulate gradient
        }
    }
};

// Power Operation
pub const PowOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        const y = params[1];
        const z = try x.data.pow(y.data, false);
        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&div_op)),
            "Pow-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const n = tensor.parents[1];

        // Compute gradient for x
        if (x.grad) |x_grad| {
            const n_minus_1 = try n.data.sub_scalar(1.0, false);
            defer n_minus_1.deinit(); // Deallocate n_minus_1 after use

            const x_pow_n_minus_1 = try x.data.pow(n_minus_1, false);
            defer x_pow_n_minus_1.deinit(); // Deallocate x_pow_n_minus_1 after use

            const n_times_x_pow_n_minus_1 = try n.data.mul(x_pow_n_minus_1, false);
            defer n_times_x_pow_n_minus_1.deinit(); // Deallocate n_times_x_pow_n_minus_1 after use

            const grad_x = try grad.mul(n_times_x_pow_n_minus_1, false);
            defer grad_x.deinit(); // Deallocate grad_x after use

            try update_grad(x_grad, grad_x);
        }

        // Compute gradient for n
        if (n.grad) |n_grad| {
            const x_pow_n = try x.data.pow(n.data, false);
            defer x_pow_n.deinit(); // Deallocate x_pow_n after use

            const log_x = try x.data.log(false);
            defer log_x.deinit(); // Deallocate log_x after use

            const z = try x_pow_n.mul(log_x, false);
            defer z.deinit(); // Deallocate z after use

            const grad_n = try grad.mul(z, false);
            defer grad_n.deinit(); // Deallocate grad_n after use

            try update_grad(n_grad, grad_n);
        }
        // // std.debug.print("pow.backward() \n", .{});
    }
};

// GEMM Operation (General Matrix Multiplication)
pub const GemmOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;
        const x = params[0];
        const y = params[1];
        // try x.info();
        // try y.info();
        // if (x.grad) |x_grad| {
        //     try x_grad.info();
        // }
        // if (y.grad) |y_grad| {
        //     try y_grad.info();
        // }

        // Debug prints to verify shapes
        const z = try x.data.matmul(y.data, alpha, beta, false, false);

        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{ x, y },
            @as(?*Operation, @constCast(&gemm_op)),
            "matmul-res",
            x.requires_grad or y.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];
        const y = tensor.parents[1];

        // Debug prints to verify shapes
        // // std.debug.print("x shape: {any}\n", .{x.shape});
        // // std.debug.print("y shape: {any}\n", .{y.shape});
        // // std.debug.print("grad shape: {any}\n", .{grad.shape});

        // Compute gradients for x
        if (x.grad) |x_grad| {
            const grad_x = try grad.matmul(y.data, 1.0, 0.0, false, true);
            defer grad_x.deinit();

            try update_grad(x_grad, grad_x);
        }

        // Compute gradients for y
        if (y.grad) |y_grad| {
            const grad_y = try x.data.matmul(grad, 1.0, 0.0, true, false);
            defer grad_y.deinit();

            try update_grad(y_grad, grad_y);
        }
    }
};

// Clone Operation
pub const CloneOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];
        // // std.debug.print("====== \n", .{});
        // try x.info();

        const z = try x.data.clone();
        // try z.info();

        if (x.data.data.len != z.data.len) {
            // // std.debug.print("Buffer size mismatch: input={}, cloned={}\n", .{ x.data.data.len, z.data.len });
            return error.BufferSizeMismatch;
        }

        return try Tensor.init(
            allocator,
            z.shape,
            z.dtype,
            z,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&clone_op)),
            "clone-res",
            x.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        if (x.grad) |x_grad| {
            try update_grad(x_grad, grad);
        }
    }
};

// Softmax Operation
pub const SoftmaxOp = struct {
    pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
        const x = params[0];

        // Ensure the axis is valid for the input tensor's shape
        const axis: usize = if (x.data.shape.len > 1) 1 else 0;
        defer x.deinit();

        // Compute softmax
        const y_data = try x.data.softmax(axis, false);
        defer y_data.deinit();

        // Create a new Tensor from the softmax result
        return try Tensor.from_ndarray(
            allocator,
            y_data,
            &[_]*Tensor{x},
            @as(?*Operation, @constCast(&softmax_op)),
            "",
            x.requires_grad,
        );
    }

    pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
        const x = tensor.parents[0];

        // Accumulate gradient
        if (x.grad) |x_grad| {
            // Ensure the axis is valid for the input tensor's shape
            const axis: usize = if (x.data.shape.len > 1) 1 else 0;

            const softmax_x = try x.data.softmax(axis, false);
            errdefer softmax_x.deinit();

            // Compute (1 - softmax_x)
            const neg_softmax_x = try softmax_x.neg(false);
            errdefer neg_softmax_x.deinit();

            const one_minus_softmax_x = try neg_softmax_x.add_scalar(1.0, true);
            errdefer one_minus_softmax_x.deinit();

            // Compute softmax_x * (1 - softmax_x)
            const z = try softmax_x.mul(one_minus_softmax_x, false);
            errdefer z.deinit();

            // Multiply by upstream gradient
            const grad_softmax = try grad.mul(z, false);
            errdefer grad_softmax.deinit();

            try update_grad(x_grad, grad_softmax);
        }
    }
};

// // Max Operation
// pub const MaxOp = struct {
//     pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
//         const x = params[0];
//         const y_data = try x.data.max(false);
//         defer y_data.deinit();

//         return try Tensor.from_ndarray(
//             allocator,
//             y_data,
//             &[_]*Tensor{x},
//             @as(?*Operation, @constCast(&max_op)), // Use `max_op` instead of `tanh_op`
//             "",
//             x.requires_grad,
//         );
//     }

//     pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
//         const x = tensor.parents[0];

//         // Accumulate the gradient in x.grad
//         if (x.grad) |x_grad| {
//             // Compute the gradient of the max operation
//             const grad_max = try x.data.maxBackward(grad, false); // Use `maxBackward` to compute the gradient
//             try update_grad(x_grad, grad_max);
//         }
//     }
// };

// // Mean Operation
// pub const MeanOp = struct {
//     pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
//         const x = params[0];
//         const y_data = try x.data.mean(null); // Use `mean` instead of `min`
//         const y = try Tensor.from_ndarray(
//             allocator,
//             y_data,
//             &[_]*Tensor{x},
//             @as(?*Operation, @constCast(&mean_op)), // Use `mean_op` instead of `tanh_op`
//             "Mean-res",
//             x.requires_grad,
//         );
//         return y;
//     }

//     pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
//         // // std.debug.print("BACKWARD mean\n", .{});

//         const x = tensor.parents[0];
//         const z_size = x.data.get_size();
//         const grad_val: f32 = 1.0 / @as(f32, @floatFromInt(z_size));
//         // Compute the gradient of the mean operation
//         const grad_mean = try NdArray.from_value(x.allocator, x.shape, x.data.dtype, grad_val);
//         const grad_x = try grad_mean.mul(grad, false);

//         // Accumulate the gradient in x.grad
//         if (x.grad) |x_grad| {
//             try update_grad(x_grad, grad_x);
//         }
//     }
// };

// // Min Operation
// pub const MinOp = struct {
//     pub fn forward(allocator: std.mem.Allocator, params: []const *Tensor) TensorError!*Tensor {
//         const x = params[0];
//         const y_data = try x.data.min(false); // Compute the minimum value
//         const y = try Tensor.from_ndarray(
//             allocator,
//             y_data,
//             &[_]*Tensor{x},
//             @as(?*Operation, @constCast(&min_op)), // Use `min_op` for the operation
//             "",
//             x.requires_grad,
//         );
//         return y;
//     }

//     pub fn backward(tensor: *const Tensor, grad: *NdArray) TensorError!void {
//         const x = tensor.parents[0];
//         // Accumulate the gradient in x.grad
//         if (x.grad) |x_grad| {
//             // Compute the gradient of the min operation
//             const grad_min = try x.data.min_backward(grad, false); // Use `minBackward` to compute the gradient
//             try update_grad(x_grad, grad_min);
//         }
//     }
// };

// // ==========================
// // Tests for autograd structs
// // ==========================
const testing = std.testing;

// // Test Add Operation
// test "Add operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 2.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try add_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 5.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try add_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Multiply Operation
// test "Multiply operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 2.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try mul_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 6.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try mul_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 3.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 2.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Subtract Operation
// test "Subtract operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 5.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try sub_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 2.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try sub_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, -1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Divide Operation
// test "Divide operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 6.0, true);
//     const y = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 3.0, true);

//     // Forward pass
//     const z = try div_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 2.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try div_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0 / 3.0, 1e-6);

//     const y_grad_value = try y.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, -6.0 / 9.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Power Operation
// test "Power operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     const x_val = 2.0;
//     const n_val = 3.0;

//     // Create tensors
//     const x = try Tensor.from_value(allocator, &[_]usize{3}, dtypes.DataType.f32, x_val, true);
//     defer x.deinit(); // Ensure cleanup
//     const n = try Tensor.from_value(allocator, &[_]usize{3}, dtypes.DataType.f32, n_val, true);
//     defer n.deinit(); // Ensure cleanup

//     // Forward pass
//     const y = try pow_op.forward(allocator, &[_]*Tensor{ x, n });
//     defer y.deinit(); // Ensure cleanup

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{0}); // Get the first element of the result
//     try std.testing.expectApproxEqAbs(y_value.f32, 8.0, 1e-6); // Check if y = 2^3 = 8

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{3}, dtypes.DataType.f32);
//     defer grad.deinit(); // Ensure cleanup
//     try pow_op.backward(y, grad); // Compute gradients

//     // Gradient check for x
//     const expected_grad_x = n_val * std.math.pow(f32, x_val, n_val - 1); // dL/dx = 3 * 2^2 = 12
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Get the gradient of x

//     // Check gradient for x
//     try std.testing.expectApproxEqAbs(x_grad_value.f32, expected_grad_x, 1e-6);

//     // Gradient check for n
//     const expected_grad_n = std.math.pow(f32, x_val, n_val) * std.math.log(f32, std.math.e, x_val); // dL/dn = 8 * ln(2)
//     const n_grad_value = try n.grad.?.get(&[_]usize{0}); // Get the gradient of n

//     // Check gradient for n
//     try std.testing.expectApproxEqAbs(n_grad_value.f32, expected_grad_n, 1e-6);
// }

// // Test Transpose Operation
// test "Transpose operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);

//     // Forward pass
//     const y = try transpose_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{ 0, 1 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, 3.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 1.0, 1.0, 1.0 });
//     try transpose_op.backward(y, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 1, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 1.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test Tanh Operation
// test "Tanh operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, 0.5, true);

//     // Forward pass
//     const y = try tanh_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const expected_tanh = std.math.tanh(@as(f32, 0.5)); // Use runtime tanh for f32
//     const y_value = try y.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, expected_tanh, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try tanh_op.backward(y, grad);

//     // Gradient check
//     const expected_grad = 1.0 - expected_tanh * expected_tanh;
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, expected_grad, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test ReLU Operation
// test "ReLU operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x = try Tensor.from_value(allocator, &[_]usize{2}, dtypes.DataType.f32, -1.0, true);

//     // Forward pass
//     const y = try relu_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const y_value = try y.data.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, 0.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{2}, dtypes.DataType.f32);
//     try relu_op.backward(y, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{0}); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 0.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// // Test GEMM Operation
// test "GEMM operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensors
//     const x = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0 }, true);
//     const y = try Tensor.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 5.0, 6.0, 7.0, 8.0 }, true);

//     // Forward pass
//     const z = try gemm_op.forward(allocator, &[_]*Tensor{ x, y });

//     // Forward pass check
//     const z_value = try z.data.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(z_value.f32, 19.0, 1e-6);

//     // Backward pass
//     const grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, dtypes.DataType.f32, &[_]f32{ 1.0, 1.0, 1.0, 1.0 });
//     try gemm_op.backward(z, grad);

//     // Gradient check
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, 11.0, 1e-6);
//     const y_grad_value = try y.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_grad_value.f32, 4.0, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
//     z.deinit();
// }

// // Test Softmax Operation
// test "Softmax operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create input tensor
//     const data_a = [_]f32{ 1, 2, 3, 4, 5, 6 };
//     const shape = &[_]usize{ 2, 3 };
//     const x = try Tensor.from_data(allocator, shape, dtypes.DataType.f32, &data_a, true);
//     errdefer x.deinit(); // Clean up x

//     // Forward pass
//     const res = try softmax_op.forward(allocator, &[_]*Tensor{x});
//     errdefer res.deinit(); // Clean up res

//     // Compute expected softmax values for each row.
//     const exp1_row1 = @exp(@as(f32, 1.0));
//     const exp2_row1 = @exp(@as(f32, 2.0));
//     const exp3_row1 = @exp(@as(f32, 3.0));
//     const sum_row1 = exp1_row1 + exp2_row1 + exp3_row1;
//     const expected_row1 = [_]f32{ exp1_row1 / sum_row1, exp2_row1 / sum_row1, exp3_row1 / sum_row1 };

//     const exp1_row2 = @exp(@as(f32, 4.0));
//     const exp2_row2 = @exp(@as(f32, 5.0));
//     const exp3_row2 = @exp(@as(f32, 6.0));
//     const sum_row2 = exp1_row2 + exp2_row2 + exp3_row2;
//     const expected_row2 = [_]f32{ exp1_row2 / sum_row2, exp2_row2 / sum_row2, exp3_row2 / sum_row2 };

//     try res.print();
//     // Verify the results for the first row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 0, i });
//         try std.testing.expectApproxEqAbs(expected_row1[i], out.f32, 0.0001);
//     }

//     // Verify the results for the second row
//     for (0..3) |i| {
//         const out = try res.get(&[_]usize{ 1, i });
//         try std.testing.expectApproxEqAbs(expected_row2[i], out.f32, 0.0001);
//     }

//     // Backward pass
//     const upstream_grad_data = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }; // Upstream gradient
//     const upstream_grad = try NdArray.from_data(allocator, shape, dtypes.DataType.f32, &upstream_grad_data);
//     defer upstream_grad.deinit(); // Clean up upstream_grad

//     try softmax_op.backward(res, upstream_grad);

//     // Verify the gradients
//     if (x.grad) |x_grad| {
//         // Compute expected gradients manually
//         const softmax_row1 = expected_row1;
//         const softmax_row2 = expected_row2;

//         // Gradient for the first row
//         const grad_row1 = [_]f32{
//             softmax_row1[0] * (1 - softmax_row1[0]) * upstream_grad_data[0],
//             softmax_row1[1] * (1 - softmax_row1[1]) * upstream_grad_data[1],
//             softmax_row1[2] * (1 - softmax_row1[2]) * upstream_grad_data[2],
//         };

//         // Gradient for the second row
//         const grad_row2 = [_]f32{
//             softmax_row2[0] * (1 - softmax_row2[0]) * upstream_grad_data[3],
//             softmax_row2[1] * (1 - softmax_row2[1]) * upstream_grad_data[4],
//             softmax_row2[2] * (1 - softmax_row2[2]) * upstream_grad_data[5],
//         };

//         // Verify the gradients for the first row
//         for (0..3) |i| {
//             const grad = try x_grad.get(&[_]usize{ 0, i });
//             try std.testing.expectApproxEqAbs(grad_row1[i], grad.f32, 0.0001);
//         }

//         // Verify the gradients for the second row
//         for (0..3) |i| {
//             const grad = try x_grad.get(&[_]usize{ 1, i });
//             try std.testing.expectApproxEqAbs(grad_row2[i], grad.f32, 0.0001);
//         }
//     } else {
//         try std.testing.expect(false); // Fail if x.grad is null
//     }
// }

// // Test Sigmoid Operation
// test "Sigmoid operation forward and backward" {
//     const allocator = std.heap.page_allocator;

//     // Create tensor
//     const x_value: f32 = 0.5;
//     const x = try Tensor.from_value(allocator, &[_]usize{ 3, 1 }, dtypes.DataType.f32, x_value, true);

//     // Forward pass
//     const y = try sigmoid_op.forward(allocator, &[_]*Tensor{x});

//     // Forward pass check
//     const expected_sigmoid = 1.0 / (1.0 + @exp(-x_value));
//     const y_value = try y.data.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(y_value.f32, expected_sigmoid, 1e-6);

//     // Backward pass
//     const grad = try NdArray.ones(allocator, &[_]usize{ 3, 1 }, dtypes.DataType.f32);
//     try sigmoid_op.backward(y, grad);

//     // Gradient check
//     const expected_grad = expected_sigmoid * (1.0 - expected_sigmoid); // ~0.235003
//     const x_grad_value = try x.grad.?.get(&[_]usize{ 0, 0 }); // Handle the error here
//     try testing.expectApproxEqAbs(x_grad_value.f32, expected_grad, 1e-6);

//     // Clean up
//     x.deinit();
//     y.deinit();
// }

// test "Tensor Operations Gradient Test" {
//     const allocator = std.testing.allocator;

//     // Create tensors
//     const a = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 2.0, true);
//     a.label = try std.fmt.allocPrint(allocator, "a", .{});
//     defer a.deinit();

//     const b = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 3.0, true);
//     b.label = try std.fmt.allocPrint(allocator, "b", .{});
//     defer b.deinit();

//     const c = try Tensor.from_data(allocator, &[_]usize{ 1, 3 }, .f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 }, true);
//     c.label = try std.fmt.allocPrint(allocator, "c", .{});

//     defer c.deinit();

//     // Operations
//     const d = try a.add(b);
//     defer d.deinit();

//     const e = try c.mul(d);
//     defer e.deinit();

//     const f = try e.sub(b);
//     defer f.deinit();

//     const g = try f.div(c);
//     defer g.deinit();

//     // Forward pass
//     std.debug.print("Forward Pass:\n", .{});
//     std.debug.print("a =", .{});
//     try a.print();
//     std.debug.print("b =", .{});
//     try b.print();
//     std.debug.print("c =", .{});
//     try c.print();

//     std.debug.print("d = a + b =", .{});
//     try d.print();
//     std.debug.print("e = c * d =", .{});
//     try e.print();
//     std.debug.print("f = e - b =", .{});
//     try f.print();
//     std.debug.print("g = f / c =", .{});
//     try g.print();

//     // Backward pass
//     try g.backward(null);

//     // Print gradients
//     std.debug.print("\nBackward Pass:\n", .{});
//     std.debug.print("dg/da =", .{});
//     try a.grad.?.print();
//     std.debug.print("dg/db =", .{});
//     try b.grad.?.print();
//     std.debug.print("dg/dc =", .{});
//     try c.grad.?.print();
// }

// test "Comprehensive Tensor Operations Gradient Test" {
//     const allocator = std.testing.allocator;

//     // Create tensors
//     const a = try Tensor.from_data(allocator, &[_]usize{ 1, 1 }, .f32, &[_]f32{4.2}, true);
//     a.label = try std.fmt.allocPrint(allocator, "a", .{});
//     defer a.deinit();

//     const b = try Tensor.from_data(allocator, &[_]usize{ 1, 3 }, .f32, &[_]f32{ 0.0, 5.0, 6.0 }, true);
//     b.label = try std.fmt.allocPrint(allocator, "b", .{});
//     defer b.deinit();

//     const c = try Tensor.from_data(allocator, &[_]usize{ 1, 3 }, .f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 }, true);
//     c.label = try std.fmt.allocPrint(allocator, "c", .{});
//     defer c.deinit();

//     // Operations
//     const d = try a.mul(b); // d = a * b
//     d.label = try std.fmt.allocPrint(allocator, "d", .{});
//     defer d.deinit();

//     const e = try d.add(c); // e = d + c
//     e.label = try std.fmt.allocPrint(allocator, "e", .{});
//     defer e.deinit();

//     const f = try e.div(b); // f = e / b
//     f.label = try std.fmt.allocPrint(allocator, "f", .{});
//     defer f.deinit();

//     const g = try f.pow(2.0); // g = f^2
//     g.label = try std.fmt.allocPrint(allocator, "g", .{});
//     defer g.deinit();

//     const h = try g.relu(); // h = ReLU(g)
//     h.label = try std.fmt.allocPrint(allocator, "h", .{});
//     defer h.deinit();

//     const i = try h.sigmoid(); // i = Sigmoid(h)
//     i.label = try std.fmt.allocPrint(allocator, "i", .{});
//     defer i.deinit();

//     // Forward pass
//     std.debug.print("Forward Pass:\n", .{});
//     std.debug.print("a =", .{});
//     try a.print();
//     std.debug.print("b =", .{});
//     try b.print();
//     std.debug.print("c =", .{});
//     try c.print();

//     std.debug.print("d = a * b =", .{});
//     try d.print();
//     std.debug.print("e = d + c =", .{});
//     try e.print();
//     std.debug.print("f = e / b =", .{});
//     try f.print();
//     std.debug.print("g = f^2 =", .{});
//     try g.print();
//     std.debug.print("h = ReLU(g) =", .{});
//     try h.print();
//     std.debug.print("i = Sigmoid(h) =", .{});
//     try i.print();

//     // Backward pass
//     try i.backward(null);

//     // Print gradients
//     std.debug.print("\nBackward Pass:\n", .{});
//     std.debug.print("di/da =", .{});
//     try a.grad.?.print();
//     std.debug.print("di/db =", .{});
//     try b.grad.?.print();
//     std.debug.print("di/dc =", .{});
//     try c.grad.?.print();
// }

// Test case for multiple valid paths in computation graph
// test "Multiple Path Computation Graph Gradient Test" {
//     const allocator = std.testing.allocator;

//     // Create tensors
//     const F = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 2.0, true);
//     F.label = try std.fmt.allocPrint(allocator, "F", .{});
//     defer F.deinit();

//     const B = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 3.0, true);
//     B.label = try std.fmt.allocPrint(allocator, "B", .{});
//     defer B.deinit();

//     const A = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 4.0, true);
//     A.label = try std.fmt.allocPrint(allocator, "A", .{});
//     defer A.deinit();

//     // Operations
//     const D = try F.add(B);
//     D.label = try std.fmt.allocPrint(allocator, "D = F + B", .{});
//     defer D.deinit();

//     const C = try A.mul(D);
//     C.label = try std.fmt.allocPrint(allocator, "C = A * D", .{});
//     defer C.deinit();

//     const E = try C.add(D);
//     E.label = try std.fmt.allocPrint(allocator, "E = C + D", .{});
//     defer E.deinit();

//     // Forward pass
//     std.debug.print("Forward Pass:\n", .{});
//     std.debug.print("F =", .{});
//     try F.print();
//     std.debug.print("B =", .{});
//     try B.print();
//     std.debug.print("A =", .{});
//     try A.print();

//     std.debug.print("D = F + B =", .{});
//     try D.print();
//     std.debug.print("C = A * D =", .{});
//     try C.print();
//     std.debug.print("E = C + D =", .{});
//     try E.print();

//     // Backward pass
//     try E.backward(null);

//     // Print gradients
//     std.debug.print("\nBackward Pass:\n", .{});
//     std.debug.print("dE/dA =", .{});
//     try A.grad.?.print();
//     std.debug.print("dE/dB =", .{});
//     try B.grad.?.print();
//     std.debug.print("dE/dF =", .{});
//     try F.grad.?.print();
// }

// // Test case for gradient accumulation
// test "Gradient Accumulation Test" {
//     const allocator = std.testing.allocator;

//     // Create tensors
//     const A = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 2.0, true);
//     // A.label = try std.fmt.allocPrint(allocator, "A", .{});
//     defer A.deinit();

//     const B = try Tensor.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 3.0, true);
//     // B.label = try std.fmt.allocPrint(allocator, "B", .{});
//     defer B.deinit();

//     // Operations
//     const C = try A.mul(B);
//     // C.label = try std.fmt.allocPrint(allocator, "C = A * B", .{});
//     defer C.deinit();

//     const D = try A.mul(C);
//     // D.label = try std.fmt.allocPrint(allocator, "D = A * C", .{});
//     defer D.deinit();

//     const E = try D.add(C);
//     // E.label = try std.fmt.allocPrint(allocator, "E = D + C", .{});
//     defer E.deinit();

//     // Forward pass
//     std.debug.print("Forward Pass:\n", .{});
//     std.debug.print("A =", .{});
//     try A.print();
//     std.debug.print("B =", .{});
//     try B.print();

//     std.debug.print("C = A * B =", .{});
//     try C.print();
//     std.debug.print("D = A * C =", .{});
//     try D.print();
//     std.debug.print("E = D + C =", .{});
//     try E.print();

//     // Backward pass
//     try E.backward(null);

//     // Print gradients
//     std.debug.print("\nBackward Pass:\n", .{});
//     std.debug.print("dE/dA =", .{});
//     try A.grad.?.print();
//     std.debug.print("dE/dB =", .{});
//     try B.grad.?.print();

//     // Verify gradients
//     const expected_dE_dA = 15.0;
//     const expected_dE_dB = 6.0;

//     const dE_dA = try A.grad.?.get(&[_]usize{ 0, 0 });
//     try std.testing.expectApproxEqAbs(expected_dE_dA, dE_dA.f32, 1e-6);

//     const dE_dB = try B.grad.?.get(&[_]usize{ 0, 0 });
//     try std.testing.expectApproxEqAbs(expected_dE_dB, dE_dB.f32, 1e-6);
// }

// test "update_grad - Case 1: Same Shape" {
//     const allocator = std.testing.allocator;
//     std.debug.print("\n--- Running Test Case 1: Same Shape ---\n", .{});

//     // Create existing gradient (2x2 matrix)
//     const existing_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1, 2, 3, 4 });
//     defer existing_grad.deinit();
//     std.debug.print("Existing gradient:\n", .{});
//     try existing_grad.print();

//     // Create backpropagated gradient (2x2 matrix)
//     const backprop_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 5, 6, 7, 8 });
//     defer backprop_grad.deinit();
//     std.debug.print("Backpropagated gradient:\n", .{});
//     try backprop_grad.print();

//     // Update the existing gradient
//     try update_grad(existing_grad, backprop_grad);

//     // Print the result
//     std.debug.print("Updated gradient:\n", .{});
//     try existing_grad.print();
//     std.debug.print("**********************************\n", .{});
//     std.debug.print("**********************************\n", .{});
// }

// test "update_grad - Case 2: Scalar Broadcasting" {
//     const allocator = std.testing.allocator;
//     std.debug.print("\n--- Running Test Case 2: Scalar Broadcasting ---\n", .{});

//     // Create existing gradient (scalar)
//     const existing_grad = try NdArray.from_value(allocator, &[_]usize{ 1, 1 }, .f32, 2.0);
//     defer existing_grad.deinit();
//     std.debug.print("Existing gradient:\n", .{});
//     try existing_grad.print();

//     // Create backpropagated gradient (2x2 matrix)
//     const backprop_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1, 2, 3, 4 });
//     defer backprop_grad.deinit();
//     std.debug.print("Backpropagated gradient:\n", .{});
//     try backprop_grad.print();

//     // Update the existing gradient
//     try update_grad(existing_grad, backprop_grad);

//     // Print the result
//     std.debug.print("Updated gradient:\n", .{});
//     try existing_grad.print();
//     std.debug.print("**********************************\n", .{});
//     std.debug.print("**********************************\n", .{});
// }

// test "update_grad - Case 3: Column Vector Broadcasting" {
//     const allocator = std.testing.allocator;
//     std.debug.print("\n--- Running Test Case 3: Column Vector Broadcasting ---\n", .{});

//     // Create existing gradient (column vector)
//     const existing_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 1 }, .f32, &[_]f32{ 1.0, 2.0 });
//     defer existing_grad.deinit();
//     std.debug.print("Existing gradient:\n", .{});
//     try existing_grad.print();

//     // Create backpropagated gradient (2x2 matrix)
//     const backprop_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 });
//     defer backprop_grad.deinit();
//     std.debug.print("Backpropagated gradient:\n", .{});
//     try backprop_grad.print();

//     // Update the existing gradient
//     try update_grad(existing_grad, backprop_grad);

//     // Print the result
//     std.debug.print("Updated gradient:\n", .{});
//     try existing_grad.print();
//     std.debug.print("**********************************\n", .{});
//     std.debug.print("**********************************\n", .{});
// }

// test "update_grad - Case 4: Row Vector Broadcasting" {
//     const allocator = std.testing.allocator;
//     std.debug.print("\n--- Running Test Case 4: Row Vector Broadcasting ---\n", .{});

//     // Create existing gradient (row vector)
//     const existing_grad = try NdArray.from_data(allocator, &[_]usize{ 1, 2 }, .f32, &[_]f32{ 1.0, 2.0 });
//     defer existing_grad.deinit();
//     std.debug.print("Existing gradient:\n", .{});
//     try existing_grad.print();

//     // Create backpropagated gradient (2x2 matrix)
//     const backprop_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 3.0, 4.0, 5.0, 6.0 });
//     defer backprop_grad.deinit();
//     std.debug.print("Backpropagated gradient:\n", .{});
//     try backprop_grad.print();

//     // Update the existing gradient
//     try update_grad(existing_grad, backprop_grad);

//     // Print the result
//     std.debug.print("Updated gradient:\n", .{});
//     try existing_grad.print();
//     std.debug.print("**********************************\n", .{});
//     std.debug.print("**********************************\n", .{});
// }

// test "update_grad - Case 5: Mismatched Dimensions (Incompatible Shapes)" {
//     const allocator = std.testing.allocator;
//     std.debug.print("\n--- Running Test Case 5: Mismatched Dimensions (Incompatible Shapes) ---\n", .{});

//     // Create existing gradient (2x2 matrix)
//     const existing_grad = try NdArray.from_data(allocator, &[_]usize{ 2, 2 }, .f32, &[_]f32{ 1, 2, 3, 4 });
//     defer existing_grad.deinit();
//     std.debug.print("Existing gradient:\n", .{});
//     try existing_grad.print();

//     // Create backpropagated gradient (3x3 matrix, incompatible shape)
//     const backprop_grad = try NdArray.from_data(allocator, &[_]usize{ 3, 3 }, .f32, &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 });
//     defer backprop_grad.deinit();
//     std.debug.print("Backpropagated gradient:\n", .{});
//     try backprop_grad.print();

//     // Attempt to update the existing gradient (should return an error)
//     if (update_grad(existing_grad, backprop_grad)) |_| {
//         std.debug.print("Error: Expected IncompatibleShapes error but got success\n", .{});
//     } else |err| {
//         std.debug.print("Caught expected error: {}\n", .{err});
//         std.debug.print("**********************************\n", .{});
//         std.debug.print("**********************************\n", .{});
//     }
// }
