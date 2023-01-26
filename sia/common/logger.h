/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <sstream>
#include <string>

namespace sia {

/// A base class to define logging.  Override to use a custom logger.
class LoggerInterface {
 public:
  LoggerInterface() = default;
  virtual ~LoggerInterface() = default;
  virtual void debug(const std::string& msg) const = 0;
  virtual void info(const std::string& msg) const = 0;
  virtual void warn(const std::string& msg) const = 0;
  virtual void error(const std::string& msg) const = 0;
  virtual void critical(const std::string& msg) const = 0;
};

class Logger {
 public:
  /// Set the inherited LoggerInterface to use a custom logger
  static void setCustomLogger(LoggerInterface& interface);

  // Sia classes should use the following routines internally
  static void debug(const std::string& msg);
  static void info(const std::string& msg);
  static void warn(const std::string& msg);
  static void error(const std::string& msg);
  static void critical(const std::string& msg);

 private:
  static Logger& instance();
  Logger();
  LoggerInterface* m_interface;
};

/// Formats an expression inplace using stringstream
#define SIA_FMT(expression) \
  ((std::ostringstream&)(std::ostringstream() << expression)).str()

/// Logs an expression
#define SIA_DEBUG(expression) Logger::debug(SIA_FMT(expression))

/// Logs an expression
#define SIA_INFO(expression) Logger::info(SIA_FMT(expression))

/// Logs an expression
#define SIA_WARN(expression) Logger::warn(SIA_FMT(expression))

/// Logs an expression
#define SIA_ERROR(expression) Logger::error(SIA_FMT(expression))

/// Logs an expression
#define SIA_CRITICAL(expression) Logger::critical(SIA_FMT(expression))

}  // namespace sia
