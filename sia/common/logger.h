/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include "sia/common/format.h"

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

/// The default logging interface unless a custom one is initialized.
class DefaultLogger : public LoggerInterface {
 public:
  DefaultLogger() = default;
  virtual ~DefaultLogger() = default;
  virtual void debug(const std::string& msg) const;
  virtual void info(const std::string& msg) const;
  virtual void warn(const std::string& msg) const;
  virtual void error(const std::string& msg) const;
  virtual void critical(const std::string& msg) const;
};

class Logger {
 public:
  /// Set the inherited LoggerInterface to use a custom logger
  template <typename T = DefaultLogger>
  static void init() {
    instance().m_interface = std::make_shared<T>();
  }

  // Sia classes should use the following routines internally
  static void debug(const std::string& msg);
  static void info(const std::string& msg);
  static void warn(const std::string& msg);
  static void error(const std::string& msg);
  static void critical(const std::string& msg);

 private:
  static Logger& instance();
  Logger();
  std::shared_ptr<LoggerInterface> m_interface;
};

/// Logs an expression
#define SIA_DEBUG(expression) sia::Logger::debug(SIA_FMT(expression));

/// Logs an expression
#define SIA_INFO(expression) sia::Logger::info(SIA_FMT(expression));

/// Logs an expression
#define SIA_WARN(expression) sia::Logger::warn(SIA_FMT(expression));

/// Logs an expression
#define SIA_ERROR(expression) sia::Logger::error(SIA_FMT(expression));

/// Logs an expression
#define SIA_CRITICAL(expression) sia::Logger::critical(SIA_FMT(expression));

}  // namespace sia
