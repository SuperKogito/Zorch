const std = @import("std");

/// Configuration for the logger.
///
/// This struct contains the following fields:
/// - `timestamp_offset`: The offset in seconds for timestamps (default: 3600).
/// - `use_timestamps`: Whether to include timestamps in logs (default: true).
/// - `use_colors`: Whether to use colored log output (default: true).
pub const LoggerConfig = struct {
    timestamp_offset: i64 = 3600, // Default offset in seconds (1 hour)
    use_timestamps: bool = true, // Enable timestamps in logs
    use_colors: bool = true, // Enable colored log output
};
var logger_config: LoggerConfig = .{};

/// Initializes the logger with a custom configuration.
///
/// # Parameters
/// - `config`: The configuration to apply to the logger.
pub fn init_logger(config: LoggerConfig) void {
    logger_config = config;
}

/// ANSI color codes for colored log output.
///
/// This struct contains the following fields:
/// - `red`: ANSI code for red text.
/// - `yellow`: ANSI code for yellow text.
/// - `blue`: ANSI code for blue text.
/// - `magenta`: ANSI code for magenta text.
/// - `reset`: ANSI code to reset text color.
/// ANSI color codes for colored log output.
const Color = struct {
    const red = "\x1b[31m";
    const yellow = "\x1b[33m";
    const blue = "\x1b[34m";
    const magenta = "\x1b[35m";
    const reset = "\x1b[0m";
};

/// Formats the log level as a colored string.
///
/// # Parameters
/// - `level`: The log level to format.
///
/// # Returns
/// A formatted string representing the log level.
fn format_log_level(level: std.log.Level) []const u8 {
    return if (logger_config.use_colors) switch (level) {
        .err => Color.red ++ "ERROR" ++ Color.reset,
        .warn => Color.yellow ++ "WARN" ++ Color.reset,
        .info => Color.blue ++ "INFO" ++ Color.reset,
        .debug => Color.magenta ++ "DEBUG" ++ Color.reset,
    } else switch (level) {
        .err => "ERROR",
        .warn => "WARN",
        .info => "INFO",
        .debug => "DEBUG",
    };
}

/// Internal function to handle log output.
///
/// # Parameters
/// - `level`: The log level.
/// - `scope`: The scope of the log message.
/// - `message`: The log message format string.
/// - `args`: Arguments to format into the message.
fn log_message(comptime level: std.log.Level, scope: []const u8, comptime message: []const u8, args: anytype) void {
    std.debug.lockStdErr();
    defer std.debug.unlockStdErr();
    const stderr = std.io.getStdErr().writer();

    // Print timestamp if enabled
    if (logger_config.use_timestamps) {
        const timestamp = std.time.timestamp() + logger_config.timestamp_offset;
        const time = Time.from_unix_timestamp(timestamp);
        nosuspend stderr.print("{d:0>4}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2}:{d:0>2} ", .{
            time.year, time.month, time.day, time.hour, time.minute, time.second,
        }) catch return;
    }

    // Print log level, scope, and message
    nosuspend stderr.print("{s} [{s}] " ++ message ++ "\n", .{ format_log_level(level), scope } ++ args) catch return;
}

/// Public logging functions for different log levels.
///
/// # Parameters
/// - `scope`: The scope of the log message.
/// - `message`: The log message format string.
/// - `args`: Arguments to format into the message.
pub fn log(level: std.log.Level, scope: []const u8, comptime message: []const u8, args: anytype) void {
    log_message(level, scope, message, args);
}

pub fn err(scope: []const u8, comptime message: []const u8, args: anytype) void {
    log_message(.err, scope, message, args);
}

pub fn warn(scope: []const u8, comptime message: []const u8, args: anytype) void {
    log_message(.warn, scope, message, args);
}

pub fn info(scope: []const u8, comptime message: []const u8, args: anytype) void {
    log_message(.info, scope, message, args);
}

pub fn debug(scope: []const u8, comptime message: []const u8, args: anytype) void {
    log_message(.debug, scope, message, args);
}

/// Time representation and conversion from Unix timestamp.
///
/// This struct contains the following fields:
/// - `year`: The year.
/// - `month`: The month.
/// - `day`: The day.
/// - `hour`: The hour.
/// - `minute`: The minute.
/// - `second`: The second.
pub const Time = struct {
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,

    /// Converts a Unix timestamp to a Time struct.
    ///
    /// # Parameters
    /// - `timestamp`: The Unix timestamp to convert.
    ///
    /// # Returns
    /// A `Time` struct representing the converted timestamp.
    pub fn from_unix_timestamp(timestamp: i64) Time {
        const days = @divFloor(timestamp, 86400);
        const seconds_in_day = @mod(timestamp, 86400);
        const year_info = get_year(days);
        const month_day = get_month_day(year_info.days_into_year, year_info.year);

        return Time{
            .year = year_info.year,
            .month = month_day.month,
            .day = month_day.day,
            .hour = @intCast(@divFloor(seconds_in_day, 3600)),
            .minute = @intCast(@divFloor(@mod(seconds_in_day, 3600), 60)),
            .second = @intCast(@mod(seconds_in_day, 60)),
        };
    }
};

/// Computes the year and remaining days from a total number of days since 1970.
///
/// # Parameters
/// - `days`: The total number of days since 1970.
///
/// # Returns
/// A struct containing the year and the number of days into that year.
fn get_year(days: i64) struct { year: u16, days_into_year: i64 } {
    var year: u16 = 1970;
    var remaining_days: i64 = days;

    while (true) {
        const days_in_year: i64 = if (is_leap_year(year)) 366 else 365;
        if (remaining_days < days_in_year) break;
        remaining_days -= days_in_year;
        year += 1;
    }

    return .{ .year = year, .days_into_year = remaining_days };
}

/// Computes the month and day from the number of days into a year.
///
/// # Parameters
/// - `days_into_year`: The number of days into the year.
/// - `year`: The year.
///
/// # Returns
/// A struct containing the month and day.
fn get_month_day(days_into_year: i64, year: u16) struct { month: u8, day: u8 } {
    const days_per_month = [_]u8{ 31, 28 + @as(u8, @intFromBool(is_leap_year(year))), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    var month: u8 = 1;
    var remaining_days: i64 = days_into_year;

    while (true) {
        const days_this_month = days_per_month[month - 1];
        if (remaining_days < days_this_month) break;
        remaining_days -= days_this_month;
        month += 1;
    }

    return .{ .month = month, .day = @intCast(remaining_days + 1) };
}

/// Checks if a year is a leap year.
///
/// # Parameters
/// - `year`: The year to check.
///
/// # Returns
/// `true` if the year is a leap year, otherwise `false`.
fn is_leap_year(year: u16) bool {
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0);
}
