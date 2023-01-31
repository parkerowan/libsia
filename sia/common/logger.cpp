/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/common/logger.h"
#include <iostream>
#include "sia/common/exception.h"

namespace sia {

class DefaultInterface : public LoggerInterface {
 public:
  DefaultInterface() = default;
  virtual ~DefaultInterface() = default;
  virtual void debug(const std::string& msg) const;
  virtual void info(const std::string& msg) const;
  virtual void warn(const std::string& msg) const;
  virtual void error(const std::string& msg) const;
  virtual void critical(const std::string& msg) const;
};

void DefaultInterface::debug(const std::string& msg) const {
  std::cout << "SIA [DEBUG] " << msg << "\n";
}

void DefaultInterface::info(const std::string& msg) const {
  std::cout << "SIA [INFO] " << msg << "\n";
}

void DefaultInterface::warn(const std::string& msg) const {
  std::cout << "SIA [WARN] " << msg << "\n";
}

void DefaultInterface::error(const std::string& msg) const {
  std::cout << "SIA [ERROR] " << msg << "\n";
}

void DefaultInterface::critical(const std::string& msg) const {
  std::cout << "SIA [CRITICAL] " << msg << "\n";
}

Logger& Logger::instance() {
  static Logger logger;
  return logger;
}

void Logger::setCustomLogger(LoggerInterface& interface) {
  instance().m_interface = &interface;
}

void Logger::debug(const std::string& msg) {
  SIA_THROW_IF_NOT(instance().m_interface != nullptr,
                   "LoggerInterface is not defined");
  instance().m_interface->debug(msg);
}

void Logger::info(const std::string& msg) {
  SIA_THROW_IF_NOT(instance().m_interface != nullptr,
                   "LoggerInterface is not defined");
  instance().m_interface->info(msg);
}

void Logger::warn(const std::string& msg) {
  SIA_THROW_IF_NOT(instance().m_interface != nullptr,
                   "LoggerInterface is not defined");
  instance().m_interface->warn(msg);
}

void Logger::error(const std::string& msg) {
  SIA_THROW_IF_NOT(instance().m_interface != nullptr,
                   "LoggerInterface is not defined");
  instance().m_interface->error(msg);
}

void Logger::critical(const std::string& msg) {
  SIA_THROW_IF_NOT(instance().m_interface != nullptr,
                   "LoggerInterface is not defined");
  instance().m_interface->critical(msg);
}

DefaultInterface default_interface{};

Logger::Logger() : m_interface(&default_interface) {}

}  // namespace sia
