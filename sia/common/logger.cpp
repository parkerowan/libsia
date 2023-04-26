/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/common/logger.h"
#include <iostream>
#include "sia/common/exception.h"

namespace sia {

void DefaultLogger::debug(const std::string& msg) const {
  std::cout << "SIA [DEBUG] " << msg << "\n";
}

void DefaultLogger::info(const std::string& msg) const {
  std::cout << "SIA [INFO] " << msg << "\n";
}

void DefaultLogger::warn(const std::string& msg) const {
  std::cout << "SIA [WARN] " << msg << "\n";
}

void DefaultLogger::error(const std::string& msg) const {
  std::cout << "SIA [ERROR] " << msg << "\n";
}

void DefaultLogger::critical(const std::string& msg) const {
  std::cout << "SIA [CRITICAL] " << msg << "\n";
}

Logger& Logger::instance() {
  static Logger logger;
  return logger;
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

Logger::Logger() : m_interface(std::make_shared<DefaultLogger>()) {}

}  // namespace sia
