/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2020, Open Source Robotics Foundation, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

#include <cassert>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>

#include <console_bridge/console.h>

class OutputHandlerString : public console_bridge::OutputHandler
{
public:
  OutputHandlerString()
  {
  }

  ~OutputHandlerString() override
  {
  }

  void log(const std::string & text, console_bridge::LogLevel level, const char *filename, int line) override
  {
    (void)line;
    (void)filename;
    text_ = text;
    log_level_ = level;
  }

  std::string text_;
  console_bridge::LogLevel log_level_{console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_NONE};
};

//////////////////////////////////////////////////
TEST(ConsoleTest, MacroExpansionTest_ItShouldCompile)
{
  if (true)
    CONSOLE_BRIDGE_logDebug("Testing Log");

  if (true)
    CONSOLE_BRIDGE_logDebug("Testing Log");
  else
  {
      assert(true);
  }

  if (true)
  {
    CONSOLE_BRIDGE_logDebug("Testing Log");
  }
  else
  {
    CONSOLE_BRIDGE_logDebug("Testing Log");
  }
}

//////////////////////////////////////////////////
TEST(ConsoleTest, StdoutStderrOutput)
{
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);
  EXPECT_NO_THROW(
    CONSOLE_BRIDGE_logDebug("Testing Log"));
  EXPECT_NO_THROW(
    CONSOLE_BRIDGE_logInform("Testing Log"));
  EXPECT_NO_THROW(
    CONSOLE_BRIDGE_logWarn("Testing Log"));
  EXPECT_NO_THROW(
    CONSOLE_BRIDGE_logError("Testing Log"));
}

//////////////////////////////////////////////////
TEST(ConsoleTest, MultipleArguments)
{
  // This tests that multiple arguments to the CONSOLE_BRIDGE_* macros get
  // formatted and output properly.

  OutputHandlerString string_oh;
  console_bridge::useOutputHandler(&string_oh);
  EXPECT_EQ(&string_oh, console_bridge::getOutputHandler());

  CONSOLE_BRIDGE_logError("no extra parameters");
  EXPECT_EQ(string_oh.text_, "no extra parameters");

  CONSOLE_BRIDGE_logError("one integer: %d", 42);
  EXPECT_EQ(string_oh.text_, "one integer: 42");

  CONSOLE_BRIDGE_logError("two floats: %.2f, %.2f", 42.01, 1/3.0);
  EXPECT_EQ(string_oh.text_, "two floats: 42.01, 0.33");
}

TEST(ConsoleTest, BasicOutputHandler)
{
  // This tests that we can install a custom OutputHandler and log to it.

  OutputHandlerString string_oh;
  console_bridge::useOutputHandler(&string_oh);
  EXPECT_EQ(&string_oh, console_bridge::getOutputHandler());
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);

  CONSOLE_BRIDGE_logDebug("Debug");

  EXPECT_EQ(string_oh.text_, "Debug");
  EXPECT_EQ(string_oh.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);
}

TEST(ConsoleTest, LogLevelTooLow)
{
  // This tests that the custom OutputHandler log() method is *not* invoked if
  // the log level set in console_bridge is higher than the message log level.
  OutputHandlerString string_oh;
  console_bridge::useOutputHandler(&string_oh);
  EXPECT_EQ(&string_oh, console_bridge::getOutputHandler());
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);

  CONSOLE_BRIDGE_logDebug("Debug");

  EXPECT_EQ(string_oh.text_, "");
  EXPECT_EQ(string_oh.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_NONE);
}

TEST(ConsoleTest, SwapHandlers)
{
  // This tests the ability to swap output handlers from one to another.

  OutputHandlerString string_oh1;
  OutputHandlerString string_oh2;

  console_bridge::useOutputHandler(&string_oh1);
  EXPECT_EQ(&string_oh1, console_bridge::getOutputHandler());
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);

  CONSOLE_BRIDGE_logInform("Info1");

  console_bridge::useOutputHandler(&string_oh2);
  EXPECT_EQ(&string_oh2, console_bridge::getOutputHandler());

  CONSOLE_BRIDGE_logInform("Info2");

  EXPECT_EQ(string_oh1.text_, "Info1");
  EXPECT_EQ(string_oh1.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);

  EXPECT_EQ(string_oh2.text_, "Info2");
  EXPECT_EQ(string_oh2.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);
}

TEST(ConsoleTest, RestoreHandler)
{
  // This tests the console_bridge::restorePreviousOutputHandler() function.

  OutputHandlerString string_oh1;
  OutputHandlerString string_oh2;

  console_bridge::useOutputHandler(&string_oh1);
  EXPECT_EQ(&string_oh1, console_bridge::getOutputHandler());
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);

  console_bridge::useOutputHandler(&string_oh2);
  EXPECT_EQ(&string_oh2, console_bridge::getOutputHandler());

  CONSOLE_BRIDGE_logInform("Info2");

  console_bridge::restorePreviousOutputHandler();

  CONSOLE_BRIDGE_logInform("Info1");

  EXPECT_EQ(string_oh1.text_, "Info1");
  EXPECT_EQ(string_oh1.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);

  EXPECT_EQ(string_oh2.text_, "Info2");
  EXPECT_EQ(string_oh2.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_INFO);
}

TEST(ConsoleTest, NoOutputHandler)
{
  // This tests that calling console_bridge::noOutputHandler() results in
  // no output, even when our custom OutputHandler is "installed".

  OutputHandlerString string_oh;
  console_bridge::useOutputHandler(&string_oh);
  EXPECT_EQ(&string_oh, console_bridge::getOutputHandler());
  console_bridge::setLogLevel(console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);

  CONSOLE_BRIDGE_logDebug("Debug");
  EXPECT_EQ(string_oh.text_, "Debug");
  EXPECT_EQ(string_oh.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);

  string_oh.text_ = "";
  string_oh.log_level_ = console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_NONE;

  console_bridge::noOutputHandler();
  EXPECT_EQ(nullptr, console_bridge::getOutputHandler());

  CONSOLE_BRIDGE_logDebug("Debug");
  EXPECT_EQ(string_oh.text_, "");
  EXPECT_EQ(string_oh.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_NONE);

  console_bridge::restorePreviousOutputHandler();
  CONSOLE_BRIDGE_logDebug("Debug2");

  EXPECT_EQ(string_oh.text_, "Debug2");
  EXPECT_EQ(string_oh.log_level_, console_bridge::LogLevel::CONSOLE_BRIDGE_LOG_DEBUG);
}

TEST(ConsoleTest, TestLogLevel)
{
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_DEBUG);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::CONSOLE_BRIDGE_LOG_DEBUG);

  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_INFO);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::CONSOLE_BRIDGE_LOG_INFO);

  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_WARN);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::CONSOLE_BRIDGE_LOG_WARN);

  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_ERROR);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::CONSOLE_BRIDGE_LOG_ERROR);

  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_NONE);
  EXPECT_EQ(console_bridge::getLogLevel(), console_bridge::CONSOLE_BRIDGE_LOG_NONE);
}

TEST(ConsoleTest, TestOutputHandlerFileBadFilename) {
  console_bridge::OutputHandlerFile handler("/really/hoping/this/path/doesnt/exist.txt");
  EXPECT_NO_THROW(
    handler.log("This should not crash", console_bridge::CONSOLE_BRIDGE_LOG_WARN, "file.cpp", 42));

  console_bridge::useOutputHandler(&handler);
  EXPECT_NO_THROW(
    CONSOLE_BRIDGE_logError("This also should not crash, nor actually log anything"));
  // ~OutputHandlerFile() should not fail to close a non-existent file handle
}

class FileHandlerTest : public ::testing::Test {
public:
  FileHandlerTest() : log_filename_("tmp.txt") {}

  virtual void SetUp()
  {
    // Needs to be reset to avoid side effects from other tests
    console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_WARN);
  }

  virtual void TearDown()
  {
    remove(log_filename());
  }

  std::string getTextFromLogFile() {
    std::ifstream f(log_filename_);
    std::stringstream result;
    result << f.rdbuf();
    return result.str();
  }

  const char * log_filename() { return log_filename_.c_str(); }

private:
  std::string log_filename_;
};

TEST_F(FileHandlerTest, TestInformDoesntLog) {
  // Use scoping to call ~OutputHandlerFile() and force in to flush contents and close file
  {
    const std::string text = "Some logging text";
    console_bridge::OutputHandlerFile handler(log_filename());
    console_bridge::useOutputHandler(&handler);
    CONSOLE_BRIDGE_logInform("This shouldn't log to file because it's only inform");
  }

  const std::string result = getTextFromLogFile();
  EXPECT_TRUE(result.empty()) << "Log file was not empty, it contained:\n\n" << result;
}

TEST_F(FileHandlerTest, TestErrorLogs) {
  const std::string text = "Some logging text";

  // Use scoping to call ~OutputHandlerFile() and force in to flush contents and close file
  {
    console_bridge::OutputHandlerFile handler(log_filename());
    console_bridge::useOutputHandler(&handler);
    CONSOLE_BRIDGE_logError(text.c_str());
  }
  const std::string expected_text = "Error:   " + text;
  const std::string result = getTextFromLogFile();

  // Just checking that expected text is in the file, not checking full log statement.
  EXPECT_NE(result.find(expected_text), result.npos)
    << "Log file did not contain expected text, instead it contained:\n\n: " << result;
}

TEST_F(FileHandlerTest, TestInformLogsWithLogLevel) {
  const std::string text = "Some logging text";
  console_bridge::setLogLevel(console_bridge::CONSOLE_BRIDGE_LOG_INFO);

  // Use scoping to call ~OutputHandlerFile() and force in to flush contents and close file
  {
    console_bridge::OutputHandlerFile handler(log_filename());
    console_bridge::useOutputHandler(&handler);
    CONSOLE_BRIDGE_logInform(text.c_str());
  }

  const std::string expected_text = "Info:    " + text;
  const std::string result = getTextFromLogFile();

  // Just checking that expected text is in the file, not checking full log statement.
  EXPECT_NE(result.find(expected_text), result.npos)
    << "Log file did not contain expected text, instead it contained:\n\n: " << result;
}
