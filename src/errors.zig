/// A custom error set for Tensor operations.
///
/// This error set defines various errors that can occur during tensor operations, such as:
/// - `OutOfMemory`: Memory allocation failed.
/// - `ShapeMismatch`: Tensor shapes do not match for the operation.
/// - `UnsupportedOperation`: The operation is not supported.
/// - `DataSizeMismatch`: The size of the data does not match the expected size.
/// - `TypeMismatch`: The data type of the tensor is incompatible with the operation.
/// - `UnsupportedDataType`: The data type is not supported for the operation.
/// - `InvalidShape`: The tensor shape is invalid for the operation.
/// - `InvalidIndex`: The provided index is invalid.
/// - `InvalidSlice`: The provided slice is invalid.
/// - `ShapeMismatchForInplace`: Shapes do not match for in-place operations.
/// - `IndexOutOfBounds`: The index is out of bounds.
/// - `InvalidInput`: The input is invalid for the operation.
/// - `InvalidAxis`: The specified axis is invalid.
/// - `InvalidAxisSize`: The size of the axis is invalid.
/// - `InvalidAxisOrder`: The order of axes is invalid.
/// - `Overflow`: An arithmetic overflow occurred.
/// - `ReshapeError`: An error occurred during tensor reshaping.
/// - `PrintError`: An error occurred while printing the tensor.
/// - `AccessDenied`: Access to a resource was denied.
/// - `Unexpected`: An unexpected error occurred.
/// - `DiskQuota`: The disk quota has been exceeded.
/// - `FileTooBig`: The file is too large.
/// - `InputOutput`: An input/output error occurred.
/// - `NoSpaceLeft`: No space is left on the device.
/// - `DeviceBusy`: The device is busy.
/// - `InvalidArgument`: An invalid argument was provided.
/// - `BrokenPipe`: A broken pipe error occurred.
/// - `SystemResources`: Insufficient system resources.
/// - `OperationAborted`: The operation was aborted.
/// - `NotOpenForWriting`: The resource is not open for writing.
/// - `LockViolation`: A lock violation occurred.
/// - `WouldBlock`: The operation would block.
/// - `ConnectionResetByPeer`: The connection was reset by the peer.
/// - `ProcessNotFound`: The process was not found.
/// - `NoDevice`: The device was not found.
/// - `BufferSizeMismatch`: The buffer size does not match the expected size.
/// - `CycleDetected`: A cycle was detected in the computation graph.
/// - `InvalidBroadcast`: The broadcast operation is invalid.
/// - `IncompatibleShapes`: The tensor shapes are incompatible for the operation.
pub const TensorError = error{
    OutOfMemory,
    ShapeMismatch,
    UnsupportedOperation,
    DataSizeMismatch,
    TypeMismatch,
    UnsupportedDataType,
    InvalidShape,
    InvalidIndex,
    InvalidSlice,
    ShapeMismatchForInplace,
    IndexOutOfBounds,
    InvalidInput,
    InvalidAxis,
    InvalidAxisSize,
    InvalidAxisOrder,
    Overflow,
    ReshapeError,
    PrintError,
    AccessDenied,
    Unexpected,
    DiskQuota,
    FileTooBig,
    InputOutput,
    NoSpaceLeft,
    DeviceBusy,
    InvalidArgument,
    BrokenPipe,
    SystemResources,
    OperationAborted,
    NotOpenForWriting,
    LockViolation,
    WouldBlock,
    ConnectionResetByPeer,
    ProcessNotFound,
    NoDevice,
    BufferSizeMismatch,
    CycleDetected,
    InvalidBroadcast,
    IncompatibleShapes,
};

/// A custom error set for NdArray operations.
///
/// This error set defines various errors that can occur during NdArray operations, such as:
/// - `OutOfMemory`: Memory allocation failed.
/// - `ShapeMismatch`: NdArray shapes do not match for the operation.
/// - `UnsupportedOperation`: The operation is not supported.
/// - `DataSizeMismatch`: The size of the data does not match the expected size.
/// - `TypeMismatch`: The data type of the NdArray is incompatible with the operation.
/// - `UnsupportedDataType`: The data type is not supported for the operation.
/// - `InvalidShape`: The NdArray shape is invalid for the operation.
/// - `InvalidIndex`: The provided index is invalid.
/// - `InvalidSlice`: The provided slice is invalid.
/// - `ShapeMismatchForInplace`: Shapes do not match for in-place operations.
/// - `IndexOutOfBounds`: The index is out of bounds.
/// - `InvalidInput`: The input is invalid for the operation.
/// - `InvalidAxis`: The specified axis is invalid.
/// - `InvalidAxisSize`: The size of the axis is invalid.
/// - `InvalidAxisOrder`: The order of axes is invalid.
/// - `Overflow`: An arithmetic overflow occurred.
/// - `ReshapeError`: An error occurred during NdArray reshaping.
/// - `PrintError`: An error occurred while printing the NdArray.
/// - `AccessDenied`: Access to a resource was denied.
/// - `Unexpected`: An unexpected error occurred.
/// - `DiskQuota`: The disk quota has been exceeded.
/// - `FileTooBig`: The file is too large.
/// - `InputOutput`: An input/output error occurred.
/// - `NoSpaceLeft`: No space is left on the device.
/// - `DeviceBusy`: The device is busy.
/// - `InvalidArgument`: An invalid argument was provided.
/// - `BrokenPipe`: A broken pipe error occurred.
/// - `SystemResources`: Insufficient system resources.
/// - `OperationAborted`: The operation was aborted.
/// - `NotOpenForWriting`: The resource is not open for writing.
/// - `LockViolation`: A lock violation occurred.
/// - `WouldBlock`: The operation would block.
/// - `ConnectionResetByPeer`: The connection was reset by the peer.
/// - `ProcessNotFound`: The process was not found.
/// - `NoDevice`: The device was not found.
/// - `BufferSizeMismatch`: The buffer size does not match the expected size.
/// - `CycleDetected`: A cycle was detected in the computation graph.
/// - `InvalidBroadcast`: The broadcast operation is invalid.
/// - `IncompatibleShapes`: The NdArray shapes are incompatible for the operation.
pub const NdarrayError = error{
    OutOfMemory,
    ShapeMismatch,
    UnsupportedOperation,
    DataSizeMismatch,
    TypeMismatch,
    UnsupportedDataType,
    InvalidShape,
    InvalidIndex,
    InvalidSlice,
    ShapeMismatchForInplace,
    IndexOutOfBounds,
    InvalidInput,
    InvalidAxis,
    InvalidAxisSize,
    InvalidAxisOrder,
    Overflow,
    ReshapeError,
    PrintError,
    AccessDenied,
    Unexpected,
    DiskQuota,
    FileTooBig,
    InputOutput,
    NoSpaceLeft,
    DeviceBusy,
    InvalidArgument,
    BrokenPipe,
    SystemResources,
    OperationAborted,
    NotOpenForWriting,
    LockViolation,
    WouldBlock,
    ConnectionResetByPeer,
    ProcessNotFound,
    NoDevice,
    BufferSizeMismatch,
    CycleDetected,
    InvalidBroadcast,
    IncompatibleShapes,
};
