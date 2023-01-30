/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

#include <chrono>
#include <thread>

class MyCustomLogger : public sia::LoggerInterface {
 public:
  MyCustomLogger() = default;

  void debug(const std::string& msg) const override {
    std::cout << "Custom debug " << msg << "\n";
  }
  void info(const std::string& msg) const override {
    std::cout << "Custom info " << msg << "\n";
  }
  void warn(const std::string& msg) const override {
    std::cout << "Custom warn " << msg << "\n";
  }
  void error(const std::string& msg) const override {
    std::cout << "Custom error " << msg << "\n";
  }
  void critical(const std::string& msg) const override {
    std::cout << "Custom critical " << msg << "\n";
  }
};

TEST(Common, Logger) {
  sia::Logger::debug("My debug");
  sia::Logger::info("My info");
  sia::Logger::warn("My warn");
  sia::Logger::error("My error");
  sia::Logger::critical("My critical");
}

TEST(Common, LoggerInterface) {
  MyCustomLogger custom_logger{};
  sia::Logger::setCustomLogger(custom_logger);

  sia::Logger::debug("My debug");
  sia::Logger::info("My info");
  sia::Logger::warn("My warn");
  sia::Logger::error("My error");
  sia::Logger::critical("My critical");
}

TEST(Common, Metrics) {
  sia::BaseMetrics metrics{};
  std::this_thread::sleep_for(std::chrono::microseconds(1));
  metrics.clockElapsedUs();
  EXPECT_GT(metrics.elapsed_us, 0);
}
