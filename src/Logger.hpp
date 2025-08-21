#pragma once

#include <fstream>
#include <string>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <sstream>

#include <cstdint>
#include <string_view>
#include <string>

constexpr uint32_t fnv1a(std::string_view s) {
    uint32_t h = 2166136261u;
    for (char c : s)
        h = (h ^ uint32_t(c)) * 16777619u;
    return h;
}

// ------------------------------------------------------------------
// This captures the call-site through default arguments:
struct LogSite {
    uint32_t id;

    constexpr LogSite(const char* file = __builtin_FILE(), unsigned line   = __builtin_LINE()) : id(fnv1a(file) ^ line) {}
};

class Logger {
  public:
      enum Level { INFO, WARNING, ERROR };

      static Logger& get_instance() {
        static Logger instance; 
        return instance;
      }

      Logger(const Logger&) = delete;
      Logger& operator=(const Logger&) = delete;

      static void log_info(const std::string& message, LogSite site={}) {
        if (get_instance().logging_level >= Logger::logging_info) {
          log(Level::INFO, "[" + std::to_string(site.id) + "] " + std::string(message));
        }
      }

      static void log_warning(const std::string& message, LogSite site={}) {
        if (get_instance().logging_level >= Logger::logging_warnings) {
          log(Level::WARNING, "[" + std::to_string(site.id) + "] " + std::string(message));
        }
      }

      static void log_error(const std::string& message, LogSite site={}) {
        if (get_instance().logging_level >= Logger::logging_errors) {
          log(Level::ERROR, "[" + std::to_string(site.id) + "] " + std::string(message));
        }
      }

      static std::string read_log() {
        Logger& instance = get_instance();
        
        if (!instance.log_file.is_open()) {
          return "";
        }

        instance.log_file.flush(); 

        std::ifstream in(instance.log_file_path);  
        if (!in.is_open()) {
          std::cerr << "Failed to read log file: " << instance.log_file_path << std::endl;
          return "";
        }

        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
      }

  private:
      static constexpr int logging_disabled = 0;
      static constexpr int logging_errors = 1;
      static constexpr int logging_warnings = 2;
      static constexpr int logging_info = 3;

      int logging_level;
      std::string log_file_path;
      std::ofstream log_file;

      Logger() {
        const char* level = std::getenv("QUTILS_LOG_LEVEL");
        const char* filename = std::getenv("QUTILS_LOG_FILE");

        std::string level_str = "NONE";
        if (level != nullptr && filename != nullptr) {
          level_str = level;
        }

        if (filename != nullptr) {
          log_file_path = filename;
          log_file.open(log_file_path, std::ios::app);
          if (!log_file.is_open()) {
            std::cerr << "Failed to open log file: " << log_file_path << std::endl;
          }
        }

        if (level_str == "NONE" || !log_file.is_open()) {
          logging_level = logging_disabled;
        } else if (level_str == "ERROR" || level_str == "ERRORS") {
          logging_level = logging_errors;
        } else if (level_str == "WARNING" || level_str == "WARNINGS") {
          logging_level = logging_warnings;
        } else {
          logging_level = logging_info;
        }
      }

      static void log(Level level, const std::string& message) {
        Logger& instance = get_instance();

        if (instance.log_file.is_open()) {
          instance.log_file << current_time() << " [" << level_to_string(level) << "] " << message << "\n";
        }
      }

      static std::string level_to_string(Level level) {
        switch (level) {
          case INFO:    return "INFO";
          case WARNING: return "WARNING";
          case ERROR:   return "ERROR";
        }
        return "UNKNOWN";
      }

      static std::string current_time() {
        std::time_t now = std::time(nullptr);
        char buf[100];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        return std::string(buf);
      }
};
