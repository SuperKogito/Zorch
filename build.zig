const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Define the zignet module
    const zignetModule = b.addModule("zignet", .{
        .root_source_file = b.path("src/zignet.zig"), // Main entry file
    });

    // Create the executable
    const exe = b.addExecutable(.{
        .name = "zignet",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/main.zig"),
    });

    exe.root_module.addImport("zignet", zignetModule);
    b.installArtifact(exe);

    // Create run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Add tests
    const tests = b.addTest(.{
        .root_source_file = b.path("src/zignet.zig"), // Change this if your tests are elsewhere
        .target = target,
        .optimize = optimize,
    });
    const test_run = b.addRunArtifact(tests);
    const tests_step = b.step("test", "Run tests");
    tests_step.dependOn(&test_run.step);
}
