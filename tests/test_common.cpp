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
  sia::Logger::init<MyCustomLogger>();

  sia::Logger::debug("My debug");
  sia::Logger::info("My info");
  sia::Logger::warn("My warn");
  sia::Logger::error("My error");
  sia::Logger::critical("My critical");
}

TEST(Common, Format) {
  // Format an expression in place
  std::string msg = SIA_FMT("My message" << 0);

  EXPECT_EQ(msg, "My message0");
}

TEST(Common, LoggerMacros) {
  // Format an expression in place
  std::string msg = SIA_FMT("My message" << 0);

  // Make sure the convenience logging works
  SIA_DEBUG("My debug " << 0);
  SIA_INFO("My info " << 0);
  SIA_WARN("My warn " << 0);
  SIA_ERROR("My error " << 0);
  SIA_CRITICAL("My critical " << 0);
}

TEST(Common, DefaultInterface) {
  // Now reset the logger to use the default interface
  sia::Logger::init();

  // Make sure the convenience logging still works
  SIA_DEBUG("My debug " << 0);
  SIA_INFO("My info " << 0);
  SIA_WARN("My warn " << 0);
  SIA_ERROR("My error " << 0);
  SIA_CRITICAL("My critical " << 0);
}

TEST(Common, Metrics) {
  sia::BaseMetrics metrics{};
  std::this_thread::sleep_for(std::chrono::microseconds(1));
  metrics.clockElapsedUs();
  EXPECT_GT(metrics.elapsed_us, 0);
}
